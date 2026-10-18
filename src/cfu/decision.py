"""Decisioning: fraud gate -> PD bands -> debt-capacity offer sizing, plus analyst-facing memos."""
from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd

LABELS = {
    "bureau_score": "Business bureau score", "years_in_business": "Years in business", "log_requested_amount": "Requested amount (log)",
    "industry": "Industry", "revenue_in_avg_6m": "Avg monthly revenue (6m)", "revenue_in_avg_3m": "Avg monthly revenue (3m)",
    "revenue_cv_6m": "Revenue volatility (CV, 6m)", "revenue_min_to_avg_6m": "Weakest month vs average",
    "revenue_trend_3v3": "Revenue trend (last 3m vs prior 3m)", "revenue_in_day_share_6m": "Share of days with deposits",
    "revenue_in_max_share_6m": "Largest single deposit share", "transfer_in_share_6m": "Non-revenue transfers in (share)",
    "loanproceeds_in_sum_3m": "Loan proceeds received (3m)", "mca_out_to_revenue_3m": "MCA payments / revenue (3m)",
    "mca_out_distinct_funders_3m": "Distinct MCA funders (3m)", "debt_service_to_revenue_6m": "Debt service / revenue (6m)",
    "opex_to_revenue_6m": "Opex / revenue", "payroll_to_revenue_6m": "Payroll / revenue", "owner_draw_to_revenue_6m": "Owner draws / revenue",
    "nsf_count_6m": "NSF / overdraft events (6m)", "nsf_count_3m": "NSF / overdraft events (3m)", "balance_min_6m": "Minimum balance (6m)",
    "balance_avg_to_revenue_6m": "Avg balance / monthly revenue", "days_negative_6m": "Days with negative balance (6m)",
    "days_negative_3m": "Days with negative balance (3m)", "liquidity_runway_days": "Liquidity runway (days)",
    "dscr_existing_6m": "DSCR on existing debt", "dscr_proforma": "Pro-forma DSCR incl. requested loan",
    "ask_to_monthly_revenue": "Requested amount / monthly revenue",
}


def fmt(feat: str, v) -> str:
    if isinstance(v, str):
        return v.replace("_", " ")
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if feat in ("revenue_in_avg_6m", "revenue_in_avg_3m", "loanproceeds_in_sum_3m", "balance_min_6m"):
        return f"${v:,.0f}"
    if feat.endswith(("_share_6m", "_to_revenue_6m", "_to_revenue_3m")) or feat in ("revenue_min_to_avg_6m",):
        return f"{v:.0%}"
    if isinstance(v, (int, np.integer)) or (isinstance(v, float) and float(v).is_integer()):
        return f"{int(v):,}"
    return f"{v:.2f}"


def debt_capacity_offer(row: pd.Series, cfg: dict) -> float:
    """Max new payment = free operating cash flow after existing debt / target DSCR; size the advance from it."""
    d = cfg["decision"]
    free_cf = (row.operating_cf_avg_6m or 0) - (row.debt_service_avg_6m or 0)
    max_payment = max(0.0, free_cf) / d["target_dscr"]
    max_offer = max_payment * d["offer_term_months"] / d["factor_rate"]
    return float(np.floor(min(row.requested_amount, max_offer) / 500) * 500)


def decide(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = cfg["decision"]
    out = df[["app_id"]].copy()
    out["offer_amount"] = df.apply(lambda r: debt_capacity_offer(r, cfg), axis=1)
    dec = np.where(df.fraud_score >= d["fraud_refer_min"], "REFER_FRAUD",
          np.where(df.pd >= d["pd_decline_min"], "DECLINE",
          np.where(out.offer_amount < 5000, "DECLINE_CAPACITY",
          np.where(df.pd <= d["pd_approve_max"], "APPROVE", "REFER_CREDIT"))))
    out["decision"] = dec
    out.loc[~out.decision.isin(["APPROVE", "REFER_CREDIT"]), "offer_amount"] = 0.0
    return out


def portfolio_outcome(df: pd.DataFrame, approve_mask: np.ndarray, amount: np.ndarray, cfg: dict) -> dict:
    """Illustrative economics on funded deals: good deals earn (factor - 1 - cost) on the advance,
    defaults lose 65% of the advance (partial collections)."""
    f = cfg["decision"]["factor_rate"]
    a = np.asarray(amount)[approve_mask]
    y = df.is_default.to_numpy()[approve_mask]
    pnl = np.where(y == 1, -0.65 * a, (f - 1 - 0.08) * a)
    return {"n_funded": int(approve_mask.sum()), "funded_usd": float(a.sum()), "bad_rate": float(y.mean()) if len(y) else None,
            "fraud_funded": int(df.is_fraud.to_numpy()[approve_mask].sum()), "pnl_usd": float(pnl.sum()),
            "return_on_funded": float(pnl.sum() / max(a.sum(), 1))}


MEMO_CSS = """body{font-family:Inter,Segoe UI,Arial,sans-serif;max-width:860px;margin:24px auto;color:#1d2433;padding:0 16px}
h1{font-size:20px;margin:0}h2{font-size:15px;margin:22px 0 8px;border-bottom:1px solid #e3e6ee;padding-bottom:4px}
.badge{display:inline-block;padding:4px 10px;border-radius:12px;font-weight:600;font-size:13px}
.APPROVE{background:#e3f6e8;color:#1e7a3a}.DECLINE,.DECLINE_CAPACITY{background:#fde7e7;color:#a12a2a}
.REFER_FRAUD{background:#fff1d6;color:#8a5a00}.REFER_CREDIT{background:#e8eefc;color:#2c4fa3}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:5px 8px;border-bottom:1px solid #eef0f5;text-align:left}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.kpi{background:#f6f7fb;border-radius:8px;padding:10px}
.kpi b{display:block;font-size:18px}.muted{color:#6b7385;font-size:12px}"""


def memo(row: pd.Series, reasons: list[str], flags: list[str], cfg: dict) -> tuple[dict, str]:
    keys = ["revenue_in_avg_6m", "revenue_cv_6m", "dscr_proforma", "nsf_count_6m", "days_negative_6m",
            "mca_out_distinct_funders_3m", "debt_service_to_revenue_6m", "liquidity_runway_days"]
    data = {"app_id": int(row.app_id), "industry": row.industry, "requested_amount": float(row.requested_amount),
            "decision": row.decision, "offer_amount": float(row.offer_amount), "pd_12m": round(float(row.pd), 4),
            "fraud_score": round(float(row.fraud_score), 4), "pd_reasons": reasons, "fraud_flags": flags,
            "key_metrics": {LABELS[k]: fmt(k, row[k]) for k in keys}}
    kp = "".join(f"<div class='kpi'><span class='muted'>{html.escape(k)}</span><b>{html.escape(v)}</b></div>"
                 for k, v in list(data["key_metrics"].items())[:8])
    li = lambda xs: "".join(f"<li>{html.escape(x)}</li>" for x in xs) or "<li>None</li>"
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Credit memo {data['app_id']}</title><style>{MEMO_CSS}</style></head><body>
<h1>Underwriting memo &middot; application {data['app_id']}</h1>
<p class="muted">{html.escape(row.industry.replace('_', ' ').title())} &middot; requested ${data['requested_amount']:,.0f} &middot; synthetic applicant</p>
<p><span class="badge {data['decision']}">{data['decision'].replace('_', ' ')}</span>
&nbsp; Offer: <b>${data['offer_amount']:,.0f}</b> &nbsp; 12-month PD: <b>{data['pd_12m']:.1%}</b> &nbsp; Fraud score: <b>{data['fraud_score']:.2f}</b></p>
<h2>Cash-flow profile (6 months of statements)</h2><div class="kpis">{kp}</div>
<h2>Top drivers of credit risk</h2><ol>{li(reasons)}</ol>
<h2>Document &amp; network fraud checks</h2><ul>{li(flags)}</ul>
<p class="muted">Generated by cashflow-underwriting-fraud. Offer sized at a {cfg['decision']['target_dscr']}x DSCR on free operating cash flow
({cfg['decision']['offer_term_months']}-month term, {cfg['decision']['factor_rate']} factor).</p></body></html>"""
    return data, page


def write_memo(path_json, path_html, data, page):
    path_json.write_text(json.dumps(data, indent=2))
    path_html.write_text(page)
