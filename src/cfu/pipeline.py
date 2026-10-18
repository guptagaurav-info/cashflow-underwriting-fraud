"""End-to-end: ingest -> simulate book -> classify transactions (with agent routing) -> SQL features ->
entity resolution -> fraud gate -> PD -> decision + memos -> report."""
from __future__ import annotations

import json
import time
import warnings

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from . import entity_resolution as er
from . import ingest, paysim, simulate
from .classify import TransactionAgent, TxnClassifier, classify_with_routing, merchant_key, routing_report
from .config import load_config, p
from .decision import LABELS, debt_capacity_offer, decide, fmt, memo, portfolio_outcome, write_memo
from .models import BUREAU, CASHFLOW, FRAUD, PD_MONOTONE, CalibratedGBM, fraud_models, pd_metrics, rule_flags

warnings.filterwarnings("ignore")
PROC = p("data/processed")
FIG = p("reports/figures")


def _save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, dpi=130); plt.close(fig)


def stage_public_classifier(cfg) -> dict:
    """Benchmark on the public US bank-transaction dataset: random split vs merchant-holdout split."""
    df = pd.read_csv(p(cfg["data"]["txn_categories"]["path"]))
    groups = merchant_key(df.description)
    splits = {"random_split": train_test_split(np.arange(len(df)), test_size=0.2, random_state=0, stratify=df.category),
              "merchant_holdout_split": next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(df, groups=groups))}
    out, curves = {"n": len(df), "classes": int(df.category.nunique()), "merchants": int(groups.nunique())}, {}
    for name, (tr, te) in splits.items():
        clf = TxnClassifier().fit(df.description.iloc[tr], df.category.iloc[tr])
        r = classify_with_routing(clf, TransactionAgent(clf), df.iloc[te][["description"]], cfg["classifier"]["auto_accept_confidence"])
        out[name] = routing_report(df.category.iloc[te], r)
        ok = (r.model_label.to_numpy() == df.category.iloc[te].to_numpy())
        th = np.linspace(0.3, 0.99, 30)
        curves[name] = [(float((r.confidence >= t).mean()), float(ok[r.confidence >= t].mean())) for t in th]
        print(f"[classify-public] {name}: model acc={out[name]['model_only_accuracy']:.3f} "
              f"auto-accept {out[name]['auto_accept_coverage']:.1%} @ {out[name]['auto_accept_accuracy']:.3f}")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, c in curves.items():
        ax.plot([x for x, _ in c], [y for _, y in c], "o-", ms=3, label=name.replace("_", " "))
    ax.set(xlabel="Auto-accept coverage", ylabel="Accuracy of auto-accepted rows",
           title="Confidence routing: coverage vs accuracy")
    ax.legend(); _save(fig, "classifier_coverage_accuracy.png")
    return out


def stage_smb_classify(cfg, tx, apps) -> tuple[pd.DataFrame, dict]:
    """Train on a 'hand-labelled' sample of 1,000 training-period accounts; classify every other account."""
    prim = apps[apps.is_primary]
    train_ids = prim.loc[prim.app_date <= cfg["split"]["train_until"], "app_id"].astype(int)
    labelled = set(train_ids.sample(1000, random_state=0))
    lab = tx[tx.app_id.isin(labelled)]
    clf = TxnClassifier().fit(lab.description, lab.category)
    rest = tx[~tx.app_id.isin(labelled)].reset_index(drop=True)
    agent = TransactionAgent(clf)
    ids = rest.app_id.unique()
    parts = []
    for chunk in np.array_split(ids, max(1, len(ids) // 500)):       # bounded memory: ~500 accounts at a time
        m = rest.app_id.isin(chunk).to_numpy()
        r_ = classify_with_routing(clf, agent, rest[m].reset_index(drop=True), cfg["classifier"]["auto_accept_confidence"])
        r_.index = np.where(m)[0]
        parts.append(r_)
    routed = pd.concat(parts).sort_index()
    rest = rest.assign(category_pred=routed.label.to_numpy(), tool=routed.tool.to_numpy(), needs_review=routed.needs_review.to_numpy())
    lab = lab.assign(category_pred=lab.category, tool="human_label", needs_review=False)
    allx = pd.concat([lab, rest], ignore_index=True)
    rep = routing_report(rest.category, routed)
    rep["recall_by_class_model_vs_routed"] = {
        c: {"model": float((routed.model_label[rest.category == c] == c).mean()),
            "routed": float((routed.label[rest.category == c] == c).mean())}
        for c in ["revenue", "transfer_in", "loan_proceeds", "rent", "debt_mca", "nsf_fee"]}
    # business impact: error in "true revenue" per account
    t_rev = rest[rest.category == "revenue"].groupby("app_id").amount.sum()
    for col, lab_ in (("model", routed.model_label), ("routed", routed.label)):
        p_rev = rest[lab_.to_numpy() == "revenue"].groupby("app_id").amount.sum().reindex(t_rev.index).fillna(0)
        ape = (p_rev / t_rev - 1).abs()
        rep[f"true_revenue_abs_pct_error_{col}"] = {"median": float(ape.median()), "p95": float(ape.quantile(.95)),
                                                     "share_over_5pct": float((ape > .05).mean())}
    print(f"[classify-smb] acc model={rep['model_only_accuracy']:.4f} routed={rep['routed_accuracy']:.4f} "
          f"| loan_proceeds recall {rep['recall_by_class_model_vs_routed']['loan_proceeds']}")
    return allx, rep


def stage_features(cfg, allx, apps, stmts) -> pd.DataFrame:
    allx[["app_id", "date", "amount", "balance", "description", "category_pred", "seq"]].to_parquet(PROC / "txn_classified.parquet")
    a = apps.copy(); a["app_id"] = a["app_id"].astype("float").astype("Int64")
    a.to_parquet(PROC / "apps_flat.parquet")
    con = duckdb.connect(str(p(cfg["data"]["warehouse"])))
    con.execute("SET enable_progress_bar = false")
    for k, v in {"txn_path": PROC / "txn_classified.parquet", "app_path": PROC / "apps_flat.parquet",
                 "stmt_path": PROC / "sim_statements.parquet"}.items():
        con.execute(f"SET VARIABLE {k} = '{v}'")
    con.execute(f"SET VARIABLE factor = {cfg['decision']['factor_rate']}")
    con.execute(f"SET VARIABLE term = {cfg['decision']['offer_term_months']}")
    for f in ("01_cashflow_features.sql", "02_document_fraud_features.sql"):
        con.execute(p(f"sql/{f}").read_text()); print(f"[features] ran sql/{f}")
    cf = con.sql("SELECT * FROM cf_features").df()
    doc = con.sql("SELECT * FROM doc_features").df()
    con.close()
    ent, pairs = er.resolve(apps)
    er_eval = er.evaluate(apps, ent)
    net = er.network_features(apps, ent)
    print(f"[entity-resolution] {er_eval}")
    df = cf.merge(doc, on="app_id").merge(net, on="app_id")
    df["log_requested_amount"] = np.log1p(df.requested_amount)
    return df, er_eval


def main():
    t0 = time.time()
    cfg = load_config()
    for d in (PROC, FIG, p("reports/memos")):
        d.mkdir(parents=True, exist_ok=True)
    for old in p("reports/memos").glob("app_*"):
        old.unlink()
    out: dict = {}
    ingest.run(cfg)
    out["paysim"] = paysim.run(cfg)
    out["classifier_public"] = stage_public_classifier(cfg)

    if not (PROC / "sim_transactions.parquet").exists():
        simulate.run(cfg)
    tx = pd.read_parquet(PROC / "sim_transactions.parquet")
    apps = pd.read_parquet(PROC / "sim_applications.parquet")
    stmts = pd.read_parquet(PROC / "sim_statements.parquet")
    labels = pd.read_parquet(PROC / "sim_labels.parquet")
    out["book"] = {"applications": int(len(labels)), "transactions": int(len(tx)), "network_applications": int(len(apps)),
                   "default_rate": float(labels.is_default.mean()), "fraud_rate": float(labels.is_fraud.mean()),
                   "default_rate_by_fraud_type": labels.groupby("fraud_type").is_default.mean().round(3).to_dict(),
                   "n_by_fraud_type": labels.fraud_type.value_counts().to_dict()}

    allx, out["classifier_smb"] = stage_smb_classify(cfg, tx, apps)
    df, out["entity_resolution"] = stage_features(cfg, allx, apps, stmts)
    df = df.merge(labels[["app_id", "is_default", "is_fraud", "fraud_type"]], on="app_id")
    df.to_parquet(PROC / "model_table.parquet")
    out["book"]["median_dscr_existing_debt_holders"] = float(df.loc[df.debt_service_avg_6m > 0, "dscr_existing_6m"].median())

    is_tr = pd.to_datetime(df.app_date) <= pd.Timestamp(cfg["split"]["train_until"])
    tr, te = df[is_tr].reset_index(drop=True), df[~is_tr].reset_index(drop=True)
    out["split"] = {"train": {"n": len(tr), "default_rate": float(tr.is_default.mean()), "fraud_rate": float(tr.is_fraud.mean())},
                    "test": {"n": len(te), "default_rate": float(te.is_default.mean()), "fraud_rate": float(te.is_fraud.mean())}}

    # ---------------- fraud gate
    out["fraud"], fobj = fraud_models(tr, te)
    te["fraud_score"] = fobj["scores"]["supervised_gbm"]
    tr["fraud_score"] = fobj["gbm"].predict_proba(tr)
    flags_te = rule_flags(te)
    out["fraud"]["recall_by_fraud_type_at_threshold"] = (
        te.assign(caught=te.fraud_score >= cfg["decision"]["fraud_refer_min"]).query("is_fraud == 1")
        .groupby("fraud_type").caught.mean().round(3).to_dict())
    out["fraud"]["false_positive_rate_at_threshold"] = float((te.loc[te.is_fraud == 0, "fraud_score"] >= cfg["decision"]["fraud_refer_min"]).mean())
    print("[fraud]", {k: round(v["pr_auc"], 3) for k, v in out["fraud"].items() if isinstance(v, dict) and "pr_auc" in v})

    # ---------------- PD (trained on non-fraud only)
    clean = tr[tr.is_fraud == 0]
    pd_base = CalibratedGBM(BUREAU, monotone={"bureau_score": -1}).fit(clean, clean.is_default)
    pd_cf = CalibratedGBM(BUREAU + CASHFLOW, monotone=PD_MONOTONE).fit(clean, clean.is_default)
    te_clean = te[te.is_fraud == 0]
    out["pd"] = {"bureau_only": {"test_nonfraud": pd_metrics(te_clean.is_default, pd_base.predict_proba(te_clean)),
                                 "test_all": pd_metrics(te.is_default, pd_base.predict_proba(te))},
                 "bureau_plus_cashflow": {"test_nonfraud": pd_metrics(te_clean.is_default, pd_cf.predict_proba(te_clean)),
                                          "test_all": pd_metrics(te.is_default, pd_cf.predict_proba(te))}}
    te["pd"] = pd_cf.predict_proba(te)
    te["pd_bureau"] = pd_base.predict_proba(te)
    print("[pd] AUC bureau-only {:.3f} -> +cash-flow {:.3f} (non-fraud test)".format(
        out["pd"]["bureau_only"]["test_nonfraud"]["auc"], out["pd"]["bureau_plus_cashflow"]["test_nonfraud"]["auc"]))
    contrib = pd_cf.contributions(te)
    imp = contrib.abs().mean().sort_values(ascending=False)
    out["pd"]["mean_abs_shap_top15"] = imp.head(15).round(4).to_dict()
    # fraudsters' PD: doctored statements make weak businesses look healthy
    out["pd"]["mean_pd_by_fraud_type"] = te.groupby("fraud_type").pd.mean().round(3).to_dict()
    out["pd"]["observed_default_by_fraud_type"] = te.groupby("fraud_type").is_default.mean().round(3).to_dict()

    # ---------------- decisions
    dec = decide(te, cfg)
    te = te.merge(dec, on="app_id")
    out["decision_mix"] = te.decision.value_counts(normalize=True).round(4).to_dict()
    gated = (te.decision == "APPROVE").to_numpy()
    # PD-only policy: same PD cutoff, no fraud gate; offers sized from the (possibly doctored) statements
    off_nogate = te.apply(lambda r: debt_capacity_offer(r, cfg), axis=1).to_numpy()
    no_gate = ((te.pd <= cfg["decision"]["pd_approve_max"]).to_numpy()) & (off_nogate >= 5000)
    out["policy_comparison_test"] = {
        "pd_only_no_fraud_gate": portfolio_outcome(te, no_gate, off_nogate, cfg),
        "fraud_gate_then_pd": portfolio_outcome(te, gated, te.offer_amount.to_numpy(), cfg),
        "bureau_only_pd": portfolio_outcome(te, ((te.pd_bureau <= cfg["decision"]["pd_approve_max"]) & (off_nogate >= 5000)).to_numpy(), off_nogate, cfg),
    }
    print("[policy]", json.dumps(out["policy_comparison_test"], default=float))

    # ---------------- memos for a few illustrative applications
    picks = []
    for d_ in ["APPROVE", "DECLINE", "REFER_FRAUD", "REFER_CREDIT"]:
        c = te[te.decision == d_]
        if len(c):
            picks.append(c.sample(1, random_state=3).index[0])
    st = te.loc[te.fraud_type == "stacking"]
    if len(st):
        picks.append(st.index[0])
    for i in dict.fromkeys(picks):
        row = te.loc[i]
        c = contrib.loc[i].sort_values(ascending=False)
        reasons = [f"{LABELS.get(f, f)}: {fmt(f, row[f])}" for f in c.index[:3] if c[f] > 0]
        flags = [k for k, v in flags_te.loc[i].items() if v]
        data, page = memo(row, reasons, flags, cfg)
        write_memo(p(f"reports/memos/app_{int(row.app_id)}.json"), p(f"reports/memos/app_{int(row.app_id)}.html"), data, page)
    out["memos"] = sorted(str(x.name) for x in p("reports/memos").glob("*.html"))

    # ---------------- figures
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for name, col in [("bureau only", "pd_bureau"), ("bureau + cash flow", "pd")]:
        f_, t_, _ = roc_curve(te_clean.is_default, te.loc[te.is_fraud == 0, col])
        auc = out["pd"]["bureau_only" if col == "pd_bureau" else "bureau_plus_cashflow"]["test_nonfraud"]["auc"]
        ax.plot(f_, t_, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=.8); ax.set(xlabel="FPR", ylabel="TPR", title="12-month default — out-of-time test")
    ax.legend(); _save(fig, "pd_roc.png")

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for k, s in fobj["scores"].items():
        pr, rc, _ = precision_recall_curve(te.is_fraud, s)
        ax.plot(rc, pr, label=f"{k.replace('_', ' ')} (PR-AUC {out['fraud'][k]['pr_auc']:.2f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Fraud detection — out-of-time test"); ax.legend()
    _save(fig, "fraud_pr.png")

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    imp.head(12)[::-1].rename(index=lambda x: LABELS.get(x, x)).plot.barh(ax=ax)
    ax.set(title="PD drivers (mean |SHAP|, log-odds)"); _save(fig, "pd_shap.png")

    fig, ax = plt.subplots(figsize=(6.4, 4))
    g = te.groupby("fraud_type").agg(mean_pd=("pd", "mean"), observed=("is_default", "mean"))
    g.plot.bar(ax=ax); ax.set(title="Doctored statements fool the PD model", ylabel="rate", xlabel="")
    ax.tick_params(axis="x", rotation=20); _save(fig, "pd_vs_fraud_type.png")

    out["runtime_sec"] = round(time.time() - t0, 1)
    with open(p("reports/metrics.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[done] {out['runtime_sec']}s -> reports/metrics.json")


if __name__ == "__main__":
    main()
