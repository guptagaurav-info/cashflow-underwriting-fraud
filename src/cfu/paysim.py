"""Fraud benchmark on PaySim (6.36M simulated mobile-money transactions, labelled fraud) - and a data-validity check.

Question asked: can ledger arithmetic (old_balance - amount = new_balance) be used as a fraud signal here,
the way it is on bank statements? Answer from the data: NO. ~90% of LEGITIMATE PaySim transactions fail
origin-balance reconciliation while fraudulent ones almost never do - an artefact of how the simulator
maintains balances, not a fraud mechanism. A model fed those fields learns the artefact and is unstable
out of time. So the benchmark model uses transaction attributes only, and the reconciliation table is
reported as the reason the feature was rejected. Out-of-time split on `step` (hour): train <= 500, test > 500.
"""
import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import load_config, p

BASE = ["amount", "log_amount", "is_transfer", "hour_of_day", "orig_old_balance", "dest_old_balance",
        "amount_to_orig_balance", "orig_drained"]


def load(cfg) -> pd.DataFrame:
    path = p(cfg["data"]["paysim"]["path"])
    duckdb.execute("SET enable_progress_bar = false")
    return duckdb.sql(f"""
        SELECT step, CASE type WHEN 'TRANSFER' THEN 1 ELSE 0 END AS is_transfer, amount, LN(1 + amount) AS log_amount,
               step % 24 AS hour_of_day, oldbalanceOrg AS orig_old_balance, oldbalanceDest AS dest_old_balance,
               amount / NULLIF(oldbalanceOrg, 0) AS amount_to_orig_balance,
               CASE WHEN oldbalanceOrg > 0 AND newbalanceOrig = 0 THEN 1 ELSE 0 END AS orig_drained,
               oldbalanceOrg - amount - newbalanceOrig AS orig_balance_error,
               oldbalanceDest + amount - newbalanceDest AS dest_balance_error,
               isFraud AS is_fraud
        FROM read_csv('{path}', header=true)
        WHERE type IN ('TRANSFER', 'CASH_OUT')      -- fraud only occurs in these two types
    """).df()


def precision_at_review_rate(y, s, rate=0.005):
    k = max(1, int(rate * len(y)))
    top = np.argsort(-s)[:k]
    return float(np.mean(np.asarray(y)[top])), float(np.asarray(y)[top].sum() / max(np.sum(y), 1))


def run(cfg=None) -> dict:
    cfg = cfg or load_config()
    df = load(cfg)
    tr, te = df[df.step <= 500], df[df.step > 500]
    out = {"n_train": len(tr), "n_test": len(te), "fraud_rate_train": float(tr.is_fraud.mean()),
           "fraud_rate_test": float(te.is_fraud.mean())}
    # validity check on the reconciliation fields
    rec = df.assign(orig_recon_break=df.orig_balance_error.abs() > 0.01,
                    dest_recon_break=df.dest_balance_error.abs() > 0.01)
    out["reconciliation_break_rate_by_class"] = (rec.groupby("is_fraud")[["orig_recon_break", "dest_recon_break"]]
                                                 .mean().round(4).rename(index={0: "legit", 1: "fraud"}).to_dict("index"))
    m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=63, min_child_samples=100,
                       scale_pos_weight=5, verbose=-1, seed=0),
                  lgb.Dataset(tr[BASE], tr.is_fraud), 300)
    s = m.predict(te[BASE])
    prec, rec_ = precision_at_review_rate(te.is_fraud.to_numpy(), s, 0.01)
    out["attributes_model_oot"] = {"auc": float(roc_auc_score(te.is_fraud, s)),
                                   "pr_auc": float(average_precision_score(te.is_fraud, s)),
                                   "precision_at_1pct_review": prec, "recall_at_1pct_review": rec_}
    imp = pd.Series(m.feature_importance("gain"), index=BASE)
    out["attributes_model_gain_share"] = (imp / imp.sum()).round(3).sort_values(ascending=False).to_dict()
    print("[paysim]", out)
    return out


if __name__ == "__main__":
    run()
