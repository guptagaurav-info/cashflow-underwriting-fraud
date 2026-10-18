"""Default (PD) and fraud models.

Design choices:
* Fraud is a GATE, not a PD feature. Confirmed-fraud applications are excluded from PD training:
  a doctored statement looks healthy, and we don't want the credit model to learn that pattern as "safe".
* PD challenger = bureau/firmographics + cash-flow features; baseline = bureau/firmographics only.
  The lift is the business case for cash-flow underwriting.
* Out-of-time split by application date; isotonic calibration fit with cross-validated predictions
  inside the training window only.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

BUREAU = ["bureau_score", "years_in_business", "log_requested_amount", "industry"]
CASHFLOW = [
    "revenue_in_avg_6m", "revenue_in_avg_3m", "revenue_cv_6m", "revenue_min_to_avg_6m", "revenue_trend_3v3",
    "revenue_in_day_share_6m", "revenue_in_max_share_6m", "transfer_in_share_6m", "loanproceeds_in_sum_3m",
    "mca_out_to_revenue_3m", "mca_out_distinct_funders_3m", "debt_service_to_revenue_6m", "opex_to_revenue_6m",
    "payroll_to_revenue_6m", "owner_draw_to_revenue_6m", "nsf_count_6m", "nsf_count_3m", "balance_min_6m",
    "balance_avg_to_revenue_6m", "days_negative_6m", "days_negative_3m", "liquidity_runway_days",
    "dscr_existing_6m", "dscr_proforma", "ask_to_monthly_revenue",
]
DOC = ["running_balance_break_share", "header_recon_gap_max", "producer_is_editor", "created_modified_gap_days",
       "round_thousand_credit_share", "mobile_deposit_share", "benford_mad", "repeated_amount_share"]
NETWORK = ["er_lenders_14d", "er_lenders_90d", "er_shared_account_entities", "er_shared_phone_entities",
           "loanproceeds_in_sum_3m", "mca_out_distinct_funders_3m", "ask_to_monthly_revenue"]
FRAUD = DOC + [c for c in NETWORK if c not in DOC]

# Monotone constraints a credit committee would sign off on (+1 = risk increases with the feature)
PD_MONOTONE = {"bureau_score": -1, "nsf_count_6m": +1, "nsf_count_3m": +1, "days_negative_6m": +1,
               "days_negative_3m": +1, "dscr_proforma": -1, "liquidity_runway_days": -1,
               "debt_service_to_revenue_6m": +1, "mca_out_to_revenue_3m": +1}

PD_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=60,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=7,
                 deterministic=True, force_row_wise=True, num_threads=1)


def frame(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    X = df[feats].copy()
    if "industry" in X:
        X["industry"] = X["industry"].astype("category")
    return X


def fit_gbm(X, y, params, monotone=None, rounds=400):
    p = dict(params)
    if monotone:
        p["monotone_constraints"] = [monotone.get(c, 0) for c in X.columns]
    return lgb.train(p, lgb.Dataset(X, y), rounds)


class CalibratedGBM:
    """GBM + isotonic calibration from 5-fold out-of-fold predictions (no peeking at the test window)."""

    def __init__(self, feats, params=PD_PARAMS, monotone=None, rounds=400):
        self.feats, self.params, self.monotone, self.rounds = feats, params, monotone, rounds

    def fit(self, df, y):
        X, y = frame(df, self.feats), np.asarray(y)
        oof = np.zeros(len(y))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            m = fit_gbm(X.iloc[tr], y[tr], self.params, self.monotone, self.rounds)
            oof[va] = m.predict(X.iloc[va])
        self.cal_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(oof, y)
        self.model_ = fit_gbm(X, y, self.params, self.monotone, self.rounds)
        return self

    def predict_raw(self, df):
        return self.model_.predict(frame(df, self.feats))

    def predict_proba(self, df):
        return self.cal_.predict(self.predict_raw(df))

    def contributions(self, df):
        c = self.model_.predict(frame(df, self.feats), pred_contrib=True)
        return pd.DataFrame(c[:, :-1], columns=self.feats, index=df.index)


def pd_metrics(y, p) -> dict:
    return {"n": int(len(y)), "default_rate": float(np.mean(y)), "mean_pd": float(np.mean(p)),
            "auc": float(roc_auc_score(y, p)), "gini": float(2 * roc_auc_score(y, p) - 1),
            "brier": float(brier_score_loss(y, p))}


# ----------------------------------------------------------------- fraud
RULES = {  # explainable red flags; each maps to an analyst-facing reason
    "running balance does not reconcile": lambda d: d.running_balance_break_share > 0.05,
    "statement header does not tie out": lambda d: d.header_recon_gap_max > 0.01,
    "PDF produced/modified by an editing tool": lambda d: d.producer_is_editor == 1,
    "high share of round-thousand credits": lambda d: d.round_thousand_credit_share > 0.15,
    # Benford is weak on ~300-transaction accounts (honest median MAD ~0.04); threshold set near the honest p97
    "transaction amounts deviate from Benford's law": lambda d: d.benford_mad > 0.08,
    "amounts repeat across months (template statement)": lambda d: d.repeated_amount_share > 0.20,
    "applied to 2+ other lenders in last 14 days": lambda d: d.er_lenders_14d >= 2,
    "bank account/phone shared with another business": lambda d: (d.er_shared_account_entities + d.er_shared_phone_entities) >= 1,
    "multiple new funders in last 90 days": lambda d: d.mca_out_distinct_funders_3m >= 3,
}


def rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({k: f(df).astype(int) for k, f in RULES.items()}, index=df.index)


def fraud_models(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[dict, dict]:
    """Compare three detectors at the same review budget: rules, unsupervised anomaly, supervised GBM."""
    y_tr, y_te = tr.is_fraud.to_numpy(), te.is_fraud.to_numpy()
    scores = {}
    scores["rules_count"] = rule_flags(te).sum(1).to_numpy() + 1e-3 * te.running_balance_break_share.fillna(0).to_numpy()
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=0).fit(tr.loc[y_tr == 0, FRAUD].fillna(0))
    scores["isolation_forest"] = -iso.score_samples(te[FRAUD].fillna(0))
    gbm = CalibratedGBM(FRAUD, dict(PD_PARAMS, num_leaves=7, min_child_samples=20, scale_pos_weight=5), rounds=300).fit(tr, y_tr)
    scores["supervised_gbm"] = gbm.predict_proba(te)
    res = {}
    for k, s in scores.items():
        res[k] = {"pr_auc": float(average_precision_score(y_te, s)), "auc": float(roc_auc_score(y_te, s)),
                  **{f"recall_at_{int(r * 100)}pct_review": recall_at(y_te, s, r) for r in (0.03, 0.05, 0.10)}}
    return res, {"gbm": gbm, "iso": iso, "scores": scores}


def recall_at(y, s, rate):
    k = max(1, int(round(rate * len(y))))
    top = np.argsort(-np.asarray(s))[:k]
    return float(np.asarray(y)[top].sum() / max(np.sum(y), 1))
