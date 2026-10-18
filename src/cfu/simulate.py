"""Synthetic SMB lending book: businesses, network applications, bank statements, transactions, outcomes.

Why simulate: no public dataset links SMB bank transactions to loan performance. This generator
encodes the *mechanisms* an underwriter cares about, so the downstream pipeline is exercised on
realistic structure rather than on hand-written labels:

* Default is NOT drawn from a logit of the features. Each business's cash position is simulated
  forward 12 months with its true revenue process, margin, existing debt (MCA daily debits, term
  loans) and the new loan payment. Default = the business runs out of cash. Features only see the
  6 months of statements submitted at application, so predictability is earned, not hard-coded.
* Fraud manipulates the DOCUMENT the lender sees, while the outcome follows the TRUE business:
    - inflated_naive:  revenue deposits scaled up, running/closing balances not fixed (reconciliation break)
    - inflated_sophisticated: fabricated round-dollar "mobile deposits", balances recomputed, edited PDF
    - fabricated: template statement (month 1 copied forward, non-Benford amounts), shell business,
      bank account / phone shared across a ring of "different" businesses
    - stacking: genuine statements, but several applications to other lenders within days under name
      variants and fresh MCA fundings -> the stacked payments sink the business
* Legitimate businesses rate-shop too (a second lender within 30 days), so velocity is not a free label.
"""
from __future__ import annotations

import string
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import load_config, p

INDUSTRIES = {
    #                     card share, season amp, vol, op margin, weight, supplier names
    "restaurant":        (0.85, 0.15, 0.18, 0.17, 0.18, ["SYSCO CORP", "US FOODS", "RESTAURANT DEPOT", "PERFORMANCE FOOD GRP"]),
    "retail":            (0.75, 0.25, 0.20, 0.16, 0.14, ["ULINE", "FAIRE WHOLESALE", "AMAZON BUSINESS", "COSTCO WHSE"]),
    "construction":      (0.10, 0.30, 0.30, 0.20, 0.15, ["HOME DEPOT PRO", "LOWES PRO", "84 LUMBER", "FERGUSON ENT"]),
    "trucking":          (0.05, 0.10, 0.25, 0.15, 0.10, ["PILOT TRAVEL CTR", "LOVES TRAVEL STOP", "TA PETRO", "FLEETPRIDE"]),
    "professional_svcs": (0.25, 0.05, 0.15, 0.26, 0.13, ["STAPLES", "WEWORK", "ADOBE", "MICROSOFT 365"]),
    "healthcare":        (0.30, 0.05, 0.12, 0.23, 0.10, ["MCKESSON", "HENRY SCHEIN", "PATTERSON DENTAL", "MEDLINE"]),
    "auto_repair":       (0.60, 0.10, 0.18, 0.19, 0.10, ["NAPA AUTO PARTS", "AUTOZONE", "O REILLY AUTO", "WORLDPAC"]),
    "ecommerce":         (0.95, 0.30, 0.28, 0.15, 0.10, ["SHOPIFY", "ALIBABA", "UPS", "FEDEX"]),
}
PROCESSORS = ["SQUARE INC", "STRIPE", "TOAST INC", "CLOVER FISERV", "SHOPIFY PAYMENTS", "HEARTLAND PAYMENT"]
FUNDERS = ["FORA FINANCIAL", "YELLOWSTONE CAP", "PEARL CAPITAL", "LIBERTAS FUNDING", "KAPITUS", "ONDECK CAPITAL",
           "RAPID FINANCE", "CREDIBLY", "BRISK CAPITAL", "EVEREST BUS FUNDING", "MULLIGAN FUNDING", "IOU CENTRAL"]
LANDLORDS = ["BROOKFIELD PROPERTIES", "REGENCY CENTERS", "KIMCO REALTY", "MAIN ST PROPERTIES", "CBRE MGMT", "JLL PROPERTY"]
PAYROLL = ["ADP PAYROLL", "GUSTO", "PAYCHEX", "INTUIT PAYROLL", "RIPPLING"]
UTILITIES = ["CON ED OF NY", "PG&E", "DUKE ENERGY", "COMCAST BUSINESS", "VERIZON WIRELESS", "SPECTRUM BUSINESS", "AT&T"]
BANKS = ["CHASE", "BANK OF AMERICA", "WELLS FARGO", "TD BANK", "PNC", "CITIZENS", "MERCURY", "TRUIST"]
CUSTOMERS = ["SMITH & CO", "NORTHSIDE HOLDINGS", "CITY OF NEWARK", "ACME MEDICAL", "BLUE RIVER LLC", "TRI-STATE MGMT",
             "GREENLEAF HOA", "SUMMIT PARTNERS", "HARBOR LOGISTICS", "UNITEDHEALTHCARE", "AETNA", "CIGNA"]
STATES = ["NY", "NJ", "CA", "TX", "FL", "IL", "GA", "PA", "OH", "NC", "AZ", "WA"]
NAME_A = ["ATLAS", "BLUE", "SUMMIT", "EMPIRE", "GOLDEN", "PIONEER", "LIBERTY", "EAGLE", "PRIME", "METRO", "NORTH STAR",
          "RIVERSIDE", "UNITED", "PREMIER", "SUNRISE", "KEYSTONE", "HARBOR", "ROYAL", "CROWN", "FIRST CHOICE",
          "PATEL", "GARCIA", "NGUYEN", "KIM", "COHEN", "SMITH", "JOHNSON", "RODRIGUEZ", "CHEN", "MURPHY", "ROSSI",
          "OKAFOR", "SHAH", "LOPEZ", "MILLER", "DAVIS", "SINGH", "WONG", "BAKER", "HILL"]
NAME_B = {
    "restaurant": ["GRILL", "KITCHEN", "PIZZERIA", "TAQUERIA", "DINER", "BISTRO", "CAFE"],
    "retail": ["BOUTIQUE", "MARKET", "SUPPLY", "OUTFITTERS", "GOODS", "HARDWARE"],
    "construction": ["CONSTRUCTION", "BUILDERS", "PLUMBING", "ELECTRIC", "ROOFING", "CONTRACTING"],
    "trucking": ["TRUCKING", "LOGISTICS", "FREIGHT", "TRANSPORT", "HAULING"],
    "professional_svcs": ["CONSULTING", "ADVISORS", "LAW GROUP", "ACCOUNTING", "MARKETING", "DESIGN STUDIO"],
    "healthcare": ["DENTAL", "PHYSICAL THERAPY", "MEDICAL GROUP", "CHIROPRACTIC", "PHARMACY", "VET CLINIC"],
    "auto_repair": ["AUTO REPAIR", "COLLISION", "TIRE & AUTO", "MOTORS", "AUTO BODY"],
    "ecommerce": ["ONLINE", "DIRECT", "BRANDS", "SHOP", "TRADING CO"],
}
SUFFIX = ["LLC", "INC", "CORP", "LLC", "LLC", "CO"]
CLEAN_PRODUCERS = ["Chase Statement Engine 4.2", "BofA eStatements", "Wells Fargo DocGen", "TD Statements 3.1",
                   "PNC Statement Svc", "Citizens eDocs", "Mercury Statements", "Truist eStatements"]
EDITORS = ["Adobe Acrobat Pro 2024", "iLovePDF", "Sejda PDF Editor", "PDFescape", "Smallpdf"]


_ALPH = np.array(list(string.ascii_uppercase + string.digits))
_POOL: dict[int, np.ndarray] = {}


def _rid(rng, n=8) -> str:
    """Random reference code (pooled for speed)."""
    if n not in _POOL:
        _POOL[n] = np.array(["".join(r) for r in np.random.default_rng(n).choice(_ALPH, (20000, n))])
    return _POOL[n][rng.integers(0, 20000)]


@dataclass
class Txn:
    date: pd.Timestamp
    amount: float          # signed: + credit, - debit
    category: str          # ground-truth cash-flow category
    description: str


def describe(rng, cat: str, ctx: dict) -> str:
    """Bank-statement style descriptors (sign-prefixed, as in the public v2 categorisation dataset)."""
    r = rng.random()
    if cat == "revenue_card":
        proc = ctx["processor"]
        return rng.choice([f"[credit] {proc} DEPOSIT {_rid(rng, 10)} PPD ID: {rng.integers(1e9, 9e9)}",
                           f"[credit] {proc} TRANSFER {_rid(rng, 6)}", f"[credit] {proc}  PAYOUT  {_rid(rng, 8)}",
                           f"[credit] ORIG CO NAME:{proc} ENTRY DESCR:SETTLEMENT"])
    if cat == "revenue_ach":
        cust = rng.choice(CUSTOMERS)
        return rng.choice([f"[credit] {cust} ACH CREDIT INV {rng.integers(1000, 99999)}",
                           f"[credit] ORIG CO NAME:{cust} ENTRY DESCR:PAYMENT",
                           f"[credit] REMOTE ONLINE DEPOSIT # {rng.integers(1, 9)}",
                           f"[credit] DEPOSIT", f"[credit] ZELLE FROM {cust} ON {rng.integers(1, 12)}/{rng.integers(1, 28)}",
                           f"[credit] WIRE TRANSFER IN {cust}"])
    if cat == "revenue_fabricated":
        return rng.choice(["[credit] MOBILE DEPOSIT", f"[credit] MOBILE DEPOSIT REF {rng.integers(100000, 999999)}",
                           "[credit] DEPOSIT", f"[credit] COUNTER DEPOSIT BR {rng.integers(100, 999)}"])
    if cat == "transfer_in":
        return rng.choice([f"[credit] ONLINE TRANSFER FROM CHK ...{rng.integers(1000, 9999)}",
                           f"[credit] TRANSFER FROM SAVINGS XXXXXX{rng.integers(1000, 9999)}",
                           f"[credit] ZELLE FROM {ctx['owner']}", f"[credit] DEPOSIT", "[credit] BOOK TRANSFER CREDIT"])
    if cat == "loan_proceeds":
        f = ctx.get("funder") or rng.choice(FUNDERS)
        return rng.choice([f"[credit] {f} FUNDING", f"[credit] WIRE TRANSFER IN {f}", f"[credit] ORIG CO NAME:{f} ENTRY DESCR:ADVANCE"])
    if cat == "payroll":
        pr = ctx["payroll"]
        return rng.choice([f"[debit] {pr} NET PAY {_rid(rng, 6)}", f"[debit] {pr} TAX {_rid(rng, 6)} CCD ID: {rng.integers(1e9, 9e9)}",
                           f"[debit] ORIG CO NAME:{pr} ENTRY DESCR:PAYROLL"])
    if cat == "rent":
        return rng.choice([f"[debit] {ctx['landlord']} RENT", f"[debit] ONLINE PMT {ctx['landlord']}",
                           f"[debit] CHECK {rng.integers(1000, 9999)}" if r < 0.3 else f"[debit] {ctx['landlord']} LEASE PMT"])
    if cat == "cogs_opex":
        s = rng.choice(ctx["suppliers"])
        return rng.choice([f"[debit] {s} #{rng.integers(100, 9999)}", f"[debit] POS DEBIT {s}", f"[debit] {s}*{_rid(rng, 9)}",
                           f"[debit] ORIG CO NAME:{s} ENTRY DESCR:ACH PMT", f"[debit] CHECK {rng.integers(1000, 9999)}"])
    if cat == "utilities":
        return f"[debit] {rng.choice(UTILITIES)} {rng.choice(['ONLINE PMT', 'AUTOPAY', 'WEB PAY', ''])}".strip()
    if cat == "debt_mca":
        f = ctx["funder"]
        return rng.choice([f"[debit] {f} ACH DEBIT {_rid(rng, 6)}", f"[debit] ORIG CO NAME:{f} ENTRY DESCR:DAILY PMT",
                           f"[debit] {f} {rng.integers(100000, 999999)}"])
    if cat == "debt_loan":
        return rng.choice([f"[debit] {ctx['bank']} LOAN PMT", f"[debit] SBA LOAN PYMT {rng.integers(1000, 9999)}",
                           f"[debit] {ctx['bank']} COMMERCIAL LOAN {rng.integers(100, 999)}"])
    if cat == "owner_draw":
        return rng.choice([f"[debit] ONLINE TRANSFER TO CHK ...{rng.integers(1000, 9999)}", f"[debit] ZELLE TO {ctx['owner']}",
                           "[debit] BOOK TRANSFER DEBIT", f"[debit] ATM WITHDRAWAL {rng.integers(1000, 9999)}"])
    if cat == "tax":
        return rng.choice(["[debit] IRS USATAXPYMT", "[debit] EFTPS FEDERAL TAX", f"[debit] {ctx['state']} DEPT OF REVENUE SALES TAX"])
    if cat == "nsf_fee":
        return rng.choice(["[debit] NSF RETURNED ITEM FEE", "[debit] OVERDRAFT ITEM FEE", "[debit] RETURNED ITEM CHARGEBACK FEE",
                           "[debit] INSUFFICIENT FUNDS FEE"])
    raise KeyError(cat)


CATEGORY_MAP = {"revenue_card": "revenue", "revenue_ach": "revenue", "revenue_fabricated": "revenue"}


def _business_name(rng, industry: str) -> str:
    return f"{rng.choice(NAME_A)} {rng.choice(NAME_B[industry])} {rng.choice(SUFFIX)}"


def name_variant(rng, name: str) -> str:
    """How the same business shows up on a different lender's application."""
    v = name
    ops = rng.choice(["case", "suffix", "drop_suffix", "typo", "dba", "punct"], size=rng.integers(1, 3), replace=False)
    for op in ops:
        if op == "case":
            v = v.title()
        elif op == "suffix":
            v = v.replace(" LLC", " L.L.C.").replace(" INC", " INCORPORATED").replace(" CORP", " CORPORATION")
        elif op == "drop_suffix":
            v = " ".join(w for w in v.split() if w.upper().strip(".") not in {"LLC", "INC", "CORP", "CO", "L.L.C"})
        elif op == "typo" and len(v) > 6:
            i = rng.integers(1, len(v) - 1)
            v = v[:i] + v[i + 1] + v[i] + v[i + 2:]
        elif op == "dba":
            v = f"{v} DBA {v.split()[0]} {rng.choice(['GROUP', 'SERVICES', 'ENTERPRISES'])}"
        elif op == "punct":
            v = v.replace("&", "AND").replace(" ", "  ", 1)
    return v


def simulate(cfg: dict | None = None, n: int | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    sc = cfg["simulation"]
    rng = np.random.default_rng(cfg["seed"])
    n = n or sc["n_businesses"]
    S, F = sc["statement_months"], sc["forward_months"]
    ind_names = list(INDUSTRIES)
    ind_w = np.array([INDUSTRIES[k][4] for k in ind_names]); ind_w /= ind_w.sum()
    start = pd.Timestamp(sc["app_start"])

    biz, apps, stmts, txns, labels, truth = [], [], [], [], [], []
    fraud_types = rng.choice(["none", "inflated_naive", "inflated_sophisticated", "fabricated", "stacking"], size=n,
                             p=[1 - sc["fraud_rate"], sc["fraud_rate"] * .2, sc["fraud_rate"] * .2,
                                sc["fraud_rate"] * .25, sc["fraud_rate"] * .35])
    # fabricated rings share bank accounts / phones across "different" businesses
    ring_ids = {}
    fab_idx = np.where(fraud_types == "fabricated")[0]
    for k, chunk in enumerate(np.array_split(rng.permutation(fab_idx), max(1, len(fab_idx) // 4))):
        acct, phone = f"{rng.integers(1e8, 9e8)}", f"{rng.integers(200, 999)}-{rng.integers(200, 999)}-{rng.integers(1000, 9999)}"
        for i in chunk:
            ring_ids[i] = (k, acct, phone)

    app_counter = 0
    for b in range(n):
        ft = fraud_types[b]
        ind = rng.choice(ind_names, p=ind_w)
        card_share, season_amp, vol, margin0, _, suppliers = INDUSTRIES[ind]
        h = rng.normal()                                   # latent business health
        if ft in ("inflated_naive", "inflated_sophisticated", "stacking"):
            h -= 1.0                                       # fraud selects on weak businesses
        base_rev = float(np.clip(sc["target_median_monthly_revenue"] * np.exp(rng.normal(0, 0.8)), 8000, 1.5e6))
        margin = margin0 + 0.05 * h + rng.normal(0, 0.03)
        growth = 0.008 * h + rng.normal(0, 0.01)
        sigma = vol * (1 + 0.25 * max(-h, 0))
        phase = rng.uniform(0, 2 * np.pi)
        n_mca = rng.poisson(0.45 * np.exp(-0.6 * h))
        mca = [{"funder": rng.choice(FUNDERS), "daily": base_rev * rng.uniform(0.0025, 0.005),
                "remaining_days": int(rng.integers(40, 260)), "funded_month": int(rng.integers(-12, S))} for _ in range(n_mca)]
        has_loan = rng.random() < 0.25
        loan_pmt = base_rev * rng.uniform(0.03, 0.07) if has_loan else 0.0
        state = rng.choice(STATES)
        ctx = dict(processor=rng.choice(PROCESSORS), owner=f"{rng.choice(NAME_A)} {rng.choice(list(string.ascii_uppercase))}",
                   payroll=rng.choice(PAYROLL), landlord=rng.choice(LANDLORDS), suppliers=suppliers, bank=rng.choice(BANKS),
                   state=state)
        name = _business_name(rng, ind)
        zip5 = f"{rng.integers(10000, 99999)}"
        phone = f"{rng.integers(200, 999)}-{rng.integers(200, 999)}-{rng.integers(1000, 9999)}"
        acct = f"{rng.integers(1e8, 9e8)}"
        if b in ring_ids:
            _, acct, phone = ring_ids[b]
        years = float(np.clip(np.exp(rng.normal(1.6 + 0.2 * h, 0.7)), 0.3, 40))
        if ft == "fabricated":
            years = float(rng.uniform(0.5, 3))
        bureau = float(np.clip(150 + 28 * (0.55 * h + 0.84 * rng.normal()), 0, 300))   # SBSS-like 0-300
        app_date = start + pd.Timedelta(days=int(rng.integers(0, sc["app_months"] * 30.4)))
        stmt_start = (app_date - pd.DateOffset(months=S)).replace(day=1)

        # ------------------------------------------------ true monthly revenue path (history + forward)
        T = S + F
        eps = np.zeros(T)
        for t in range(T):
            eps[t] = 0.5 * (eps[t - 1] if t else 0) + rng.normal(0, sigma)
        months = pd.date_range(stmt_start, periods=T, freq="MS")
        season = 1 + season_amp * np.sin(2 * np.pi * months.month.to_numpy() / 12 + phase)
        rev_path = base_rev * (1 + growth) ** np.arange(T) * season * np.exp(eps)
        # forward shocks (customer loss, health event) - more likely for weak businesses
        shock_month = None
        if rng.random() < 0.10 + 0.12 * np.exp(-0.5 * h):          # mostly unforecastable from statements
            shock_month = int(rng.integers(S, T))
            rev_path[shock_month:] *= rng.uniform(0.35, 0.75)

        # ------------------------------------------------ transactions for the statement window (TRUE)
        bal = base_rev * rng.uniform(0.15, 0.6) * np.exp(0.4 * h)
        true_tx: list[Txn] = []
        rent = base_rev * rng.uniform(0.05, 0.10)
        month_close_true = []
        opening_true = bal
        for m in range(S):
            m0 = months[m]
            days = pd.bdate_range(m0, m0 + pd.offsets.MonthEnd(0))
            R = rev_path[m]
            month_tx: list[Txn] = []
            n_card = len(days) if card_share > 0.2 else 0
            if n_card:
                for d, a in zip(days, R * card_share * rng.dirichlet(np.ones(n_card) * 6)):
                    month_tx.append(Txn(d, round(a, 2), "revenue_card", describe(rng, "revenue_card", ctx)))
            n_ach = max(1, rng.poisson(3 + 7 * (1 - card_share)))
            for a in R * (1 - card_share) * rng.dirichlet(np.ones(n_ach) * 2):
                if ind in ("construction", "professional_svcs", "healthcare", "auto_repair") and a > 1000 and rng.random() < 0.25:
                    # honest customers often pay round-dollar checks, deposited by phone -> fraud-rule false positives
                    month_tx.append(Txn(rng.choice(days), float(round(a, -2)), "revenue_ach",
                                        describe(rng, "revenue_fabricated", ctx)))
                else:
                    month_tx.append(Txn(rng.choice(days), round(a, 2), "revenue_ach", describe(rng, "revenue_ach", ctx)))
            for pos in mca:
                if pos["funded_month"] == m:
                    month_tx.append(Txn(rng.choice(days[:5]), round(pos["daily"] * 160, 2), "loan_proceeds",
                                        describe(rng, "loan_proceeds", {"funder": pos["funder"]})))
            opex = R * (1 - margin)
            for d in (days[min(9, len(days) - 1)], days[-1]):
                month_tx.append(Txn(d, -round(opex * 0.40 / 2 * rng.uniform(.95, 1.05), 2), "payroll", describe(rng, "payroll", ctx)))
            month_tx.append(Txn(days[0], -round(rent, -1), "rent", describe(rng, "rent", ctx)))
            n_sup = max(2, rng.poisson(8))
            for a in opex * 0.45 * rng.dirichlet(np.ones(n_sup)):
                month_tx.append(Txn(rng.choice(days), -round(a, 2), "cogs_opex", describe(rng, "cogs_opex", ctx)))
            for _ in range(3):
                month_tx.append(Txn(rng.choice(days), -round(opex * 0.02 / 3 * rng.uniform(.6, 1.4), 2), "utilities", describe(rng, "utilities", ctx)))
            if m0.month in (1, 4, 6, 9):
                month_tx.append(Txn(rng.choice(days), -round(R * rng.uniform(0.03, 0.06), 2), "tax", describe(rng, "tax", ctx)))
            for pos in mca:
                if pos["funded_month"] <= m and pos["remaining_days"] > 0:
                    for d in days[: min(len(days), pos["remaining_days"])]:
                        month_tx.append(Txn(d, -round(pos["daily"], 2), "debt_mca", describe(rng, "debt_mca", {"funder": pos["funder"]})))
                    pos["remaining_days"] -= len(days)
            if has_loan:
                month_tx.append(Txn(days[14 % len(days)], -round(loan_pmt, 2), "debt_loan", describe(rng, "debt_loan", ctx)))
            draw_amt = max(0.0, R * rng.uniform(0.02, 0.07) * (1 + 0.5 * h))
            if draw_amt > 0:
                month_tx.append(Txn(rng.choice(days), -round(draw_amt, 2), "owner_draw", describe(rng, "owner_draw", ctx)))
            # owner injects cash when the account is stressed
            if bal < 0.1 * R and rng.random() < 0.6:
                month_tx.append(Txn(days[0], round(R * rng.uniform(0.05, 0.25), 2), "transfer_in", describe(rng, "transfer_in", ctx)))
            elif rng.random() < 0.12:
                month_tx.append(Txn(rng.choice(days), round(R * rng.uniform(0.02, 0.1), 2), "transfer_in", describe(rng, "transfer_in", ctx)))
            # post in date order (credits before debits within a day); NSF fees when balance < 0
            month_tx.sort(key=lambda t: (t.date, -t.amount))
            for t in month_tx:
                prev_bal = bal
                bal += t.amount
                true_tx.append(t)
                if t.amount < 0 and bal < 0 and (prev_bal >= 0 or rng.random() < 0.12):
                    fee = Txn(t.date, -35.0, "nsf_fee", describe(rng, "nsf_fee", ctx))
                    bal += fee.amount
                    true_tx.append(fee)
            month_close_true.append(bal)

        # ------------------------------------------------ loan request + forward performance (TRUE business)
        presented_rev = float(np.mean(rev_path[:S]))
        mult = rng.uniform(1.6, 2.4) if ft.startswith("inflated") else 1.0
        if ft == "fabricated":
            presented_rev = base_rev * rng.uniform(0.8, 1.5)
        ask = float(np.clip(presented_rev * mult * rng.uniform(0.5, 1.3), 5000, 500000))
        new_pmt = ask * cfg["decision"]["factor_rate"] / cfg["decision"]["offer_term_months"]
        stacked_pmt = 0.0
        n_stack = 0
        if ft == "stacking":
            n_stack = int(rng.integers(2, 5))
            stacked_pmt = sum(presented_rev * rng.uniform(0.3, 0.8) * 1.3 / 9 for _ in range(n_stack))
        cash = bal
        buffer = max(0.0, presented_rev * rng.lognormal(-1.6, 1.0) * np.exp(0.3 * h))   # owner's outside liquidity (unobserved)
        margin_fwd = margin + rng.normal(0, 0.05)                    # cost shocks after funding (unobserved)
        default_month, neg_streak = None, 0
        for f in range(F):
            t = S + f
            R = rev_path[t]
            mca_out = sum(pos["daily"] * 21 for pos in mca if pos["remaining_days"] > 0)
            for pos in mca:
                pos["remaining_days"] -= 21
            net = R * margin_fwd - R * 0.04 * (1 + 0.5 * h) - mca_out - loan_pmt - new_pmt - (stacked_pmt if f < 9 else 0)
            cash += net
            if cash < 0 and buffer > 0:
                inject = min(buffer, -cash)
                cash += inject; buffer -= inject
            neg_streak = neg_streak + 1 if cash < -0.5 * new_pmt else 0
            if neg_streak >= 2 or cash < -2.0 * new_pmt:
                default_month = f + 1
                break
        if default_month is None and rng.random() < 0.03:           # idiosyncratic events (owner health, disputes, fire)
            default_month = int(rng.integers(1, F + 1))
        if ft == "fabricated" and rng.random() < 0.85:
            default_month = int(rng.integers(1, 4))           # shell business: early payment default
        is_default = int(default_month is not None)

        # ------------------------------------------------ the DOCUMENT the lender sees (fraud manipulations)
        doc_tx = [Txn(t.date, t.amount, t.category, t.description) for t in true_tx]
        producer = rng.choice(CLEAN_PRODUCERS)
        if rng.random() < 0.03:
            producer = rng.choice(EDITORS)                    # some honest borrowers re-save PDFs
        recompute_balances = True
        if ft == "inflated_naive":
            for t in doc_tx:
                if t.category.startswith("revenue"):
                    t.amount = round(t.amount * mult, 2)
            recompute_balances = False
            producer = rng.choice(EDITORS) if rng.random() < 0.7 else producer
        elif ft == "inflated_sophisticated":
            extra = presented_rev * (mult - 1)
            for m in range(S):
                m0 = months[m]
                days = pd.bdate_range(m0, m0 + pd.offsets.MonthEnd(0))
                k = int(rng.integers(2, 6))
                for a in extra * rng.dirichlet(np.ones(k)):
                    doc_tx.append(Txn(rng.choice(days), float(max(1000, round(a, -3) if rng.random() < .5 else round(a + rng.uniform(-90, 90), 2))), "revenue_fabricated",
                                      describe(rng, "revenue_fabricated", ctx)))
            producer = rng.choice(EDITORS) if rng.random() < 0.6 else producer
        elif ft == "fabricated":
            m0 = months[0]
            first = [t for t in doc_tx if t.date < months[1]]
            template = []
            for t in first:                                   # template generator: non-Benford, uniform amounts
                amt = float(np.round(rng.uniform(100, 9999), 2)) * (1 if t.amount > 0 else -1) if t.category not in ("rent", "debt_mca") else t.amount
                template.append(Txn(t.date, amt, t.category, t.description))
            doc_tx = []
            for m in range(S):
                for t in template:
                    d = t.date + pd.DateOffset(months=m) + pd.Timedelta(days=int(rng.integers(-1, 2)))
                    doc_tx.append(Txn(d, t.amount, t.category, t.description))
            producer = rng.choice(EDITORS + ["ReportLab PDF Library"]) if rng.random() < 0.8 else producer
        elif ft == "stacking":
            for j in range(n_stack):                          # fresh fundings in the last 60 days
                d = app_date - pd.Timedelta(days=int(rng.integers(5, 60)))
                f = rng.choice(FUNDERS)
                doc_tx.append(Txn(d, round(presented_rev * rng.uniform(0.3, 0.8), 2), "loan_proceeds",
                                  describe(rng, "loan_proceeds", {"funder": f})))
        stmt_end = stmt_start + pd.DateOffset(months=S)
        doc_tx = [t for t in doc_tx if stmt_start <= t.date < stmt_end]
        doc_tx.sort(key=lambda t: (t.date, -t.amount))

        # running balance column + monthly statement headers as presented
        run = opening_true
        rows = []
        for i, t in enumerate(doc_tx):
            if recompute_balances:
                run += t.amount
                shown = run
            else:
                shown = None                                  # filled from the TRUE running balance below
            rows.append([app_counter, b, i, t.date, t.amount, t.description, CATEGORY_MAP.get(t.category, t.category), shown])
        if not recompute_balances:                            # naive edit: balances still reflect true flows
            tb = opening_true
            true_sorted = sorted(true_tx, key=lambda t: (t.date, -t.amount))
            bal_by_date = {}
            for t in true_sorted:
                tb += t.amount
                bal_by_date[t.date] = tb
            last = opening_true
            for r_ in rows:
                last = bal_by_date.get(r_[3], last)
                r_[7] = last
        txns.extend(rows)
        tx_df = pd.DataFrame(rows, columns=["app_id", "business_id", "seq", "date", "amount", "description", "category", "balance"])
        for m in range(S):
            m0, m1 = months[m], months[m] + pd.offsets.MonthEnd(0)
            mt = tx_df[(tx_df.date >= m0) & (tx_df.date <= m1)]
            prev = stmts[-1][7] if (m > 0 and stmts) else opening_true
            stmts.append([app_counter, m, m0, m1, prev,
                          float(mt.loc[mt.amount > 0, "amount"].sum()), float(-mt.loc[mt.amount < 0, "amount"].sum()),
                          float(mt.balance.iloc[-1]) if len(mt) else prev, producer,
                          int(rng.integers(0, 2)) if producer in CLEAN_PRODUCERS else int(rng.integers(0, 40))])

        # ------------------------------------------------ network applications (all lenders)
        apps.append([app_counter, b, "LENDER_US", app_date, name, phone, acct, zip5, state, ask, True, years, bureau, ind])
        extra_apps = n_stack + int(rng.integers(1, 3)) if ft == "stacking" else (int(rng.integers(1, 3)) if rng.random() < 0.2 else 0)
        for j in range(extra_apps):
            window = 10 if ft == "stacking" else 30
            d = app_date + pd.Timedelta(days=int(rng.integers(-window, window)))
            lender = f"LENDER_{rng.integers(1, 40):02d}"
            apps.append([None, b, lender, d, name_variant(rng, name), phone if rng.random() < .8 else
                         f"{rng.integers(200, 999)}-{rng.integers(200, 999)}-{rng.integers(1000, 9999)}",
                         acct, zip5, state, ask * rng.uniform(.6, 1.2), False, years, bureau, ind])
        labels.append([app_counter, b, is_default, default_month, int(ft != "none"), ft])
        truth.append([b, h, margin, base_rev, presented_rev, n_mca, has_loan, shock_month is not None])
        app_counter += 1

    out = {
        "transactions": pd.DataFrame(txns, columns=["app_id", "business_id", "seq", "date", "amount", "description", "category", "balance"]),
        "statements": pd.DataFrame(stmts, columns=["app_id", "month_idx", "period_start", "period_end", "opening_balance",
                                                   "total_credits", "total_debits", "closing_balance", "pdf_producer",
                                                   "created_modified_gap_days"]),
        "applications": pd.DataFrame(apps, columns=["app_id", "business_id", "lender_id", "app_date", "applicant_name", "phone",
                                                    "bank_account", "zip", "state", "requested_amount", "is_primary",
                                                    "years_in_business", "bureau_score", "industry"]),
        "labels": pd.DataFrame(labels, columns=["app_id", "business_id", "is_default", "default_month", "is_fraud", "fraud_type"]),
        "truth": pd.DataFrame(truth, columns=["business_id", "latent_health", "margin", "base_revenue", "presented_revenue",
                                              "n_mca", "has_term_loan", "forward_shock"]),
    }
    a = out["applications"]
    a["network_app_id"] = np.arange(len(a))
    a["app_id"] = a["app_id"].astype("Int64")
    # The statement document: the debits/credits totals in the header are recomputed from the
    # (possibly edited) transactions, but closing balances come from the running-balance column.
    return out


def run(cfg=None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    out = simulate(cfg)
    d = p("data/processed"); d.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        v.to_parquet(d / f"sim_{k}.parquet", index=False)
    lab = out["labels"]
    print(f"[simulate] {len(lab):,} applications | {len(out['transactions']):,} transactions | "
          f"{len(out['applications']):,} network applications")
    print(f"[simulate] default rate {lab.is_default.mean():.3f} | fraud rate {lab.is_fraud.mean():.3f}")
    print(lab.groupby("fraud_type").is_default.agg(["size", "mean"]).round(3).to_string())
    return out


if __name__ == "__main__":
    run()
