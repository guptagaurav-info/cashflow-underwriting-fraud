"""Transaction classification with confidence routing.

Stage 1 (fast path): TF-IDF (char + word n-grams) -> multinomial logistic regression.
Stage 2 (agent path): rows below the auto-accept confidence go to `TransactionAgent`, which
uses tools a per-row classifier doesn't have:
    * lexicon tool    - high-precision rules (NSF/overdraft fees, known MCA funders, payroll processors, tax)
    * retrieval tool  - nearest labelled neighbours (cosine on TF-IDF), weighted vote
    * context tool    - account-level context: is this amount recurring every month? is it round?
                        was the balance low right before this credit?
    * LLM tool        - optional: if ANTHROPIC_API_KEY is set and `anthropic` is installed, the agent
                        asks Claude to adjudicate between the model's top candidates, given the tool outputs.
Anything the agent still can't resolve is flagged for human review, the analogue of a
human-in-the-loop verification queue.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

LEXICON = [  # (regex, category) - deliberately few, high-precision rules
    (r"\b(NSF|OVERDRAFT|INSUFFICIENT FUNDS|RETURNED ITEM)\b.*FEE|FEE.*\b(NSF|OVERDRAFT)\b", "nsf_fee"),
    (r"^\[debit\].*\b(FORA FINANCIAL|YELLOWSTONE|PEARL CAPITAL|LIBERTAS|KAPITUS|ONDECK|RAPID FINANCE|CREDIBLY|BRISK CAPITAL|EVEREST BUS|MULLIGAN|IOU CENTRAL)\b", "debt_mca"),
    (r"^\[credit\].*\b(FORA FINANCIAL|YELLOWSTONE|PEARL CAPITAL|LIBERTAS|KAPITUS|ONDECK|RAPID FINANCE|CREDIBLY|BRISK CAPITAL|EVEREST BUS|MULLIGAN|IOU CENTRAL)\b", "loan_proceeds"),
    (r"^\[debit\].*\b(ADP|GUSTO|PAYCHEX|INTUIT PAYROLL|RIPPLING)\b", "payroll"),
    (r"^\[debit\].*\b(IRS|EFTPS|DEPT OF REVENUE)\b", "tax"),
    (r"^\[debit\].*\b(LOAN PMT|LOAN PYMT|COMMERCIAL LOAN)\b", "debt_loan"),
    (r"^\[credit\].*\b(SQUARE INC|STRIPE|TOAST INC|CLOVER FISERV|SHOPIFY PAYMENTS|HEARTLAND PAYMENT)\b", "revenue"),
]


def normalise(desc: pd.Series) -> pd.Series:
    """Strip volatile tokens (ids, dates, digits) so the model learns merchants, not reference numbers."""
    s = desc.str.upper()
    s = s.str.replace(r"\d{4,}", " # ", regex=True).str.replace(r"\d", "0", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def merchant_key(desc: pd.Series) -> pd.Series:
    """Crude merchant identity used for group (merchant-holdout) splits."""
    s = normalise(desc).str.replace(r"^\[(DEBIT|CREDIT)\]\s*", "", regex=True)
    s = s.str.replace(r"^(PP\*|TST\*|SQ \*|PYPL\*|CLV\*|POS DEBIT |EXPRESS CHECKOUT PAYMENT: |PREAPPROVED PAYMENT[A-Z ]*: )", "", regex=True)
    return s.str.extract(r"^([A-Z&'\.]+(?: [A-Z&'\.]+)?)", expand=False).fillna("UNK")


class TxnClassifier:
    def __init__(self, C: float = 8.0):
        self.C = C

    def fit(self, desc: pd.Series, y: pd.Series):
        x = normalise(desc)
        self.char_ = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200_000)
        self.word_ = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"[A-Z\[\]\*&']+")
        X = hstack([self.char_.fit_transform(x), self.word_.fit_transform(x)]).tocsr()
        self.lr_ = LogisticRegression(C=self.C, max_iter=300).fit(X, y)
        self.classes_ = self.lr_.classes_
        self.X_train_, self.y_train_ = X, np.asarray(y)
        return self

    def transform(self, desc: pd.Series):
        x = normalise(desc)
        return hstack([self.char_.transform(x), self.word_.transform(x)]).tocsr()

    def predict_proba(self, desc: pd.Series) -> np.ndarray:
        return self.lr_.predict_proba(self.transform(desc))


class TransactionAgent:
    """Resolves low-confidence rows using tools; returns label + which tool decided + review flag."""

    def __init__(self, clf: TxnClassifier, k: int = 15, use_llm: bool | None = None, max_ref: int = 30_000):
        self.clf, self.k = clf, k
        rng = np.random.default_rng(0)
        idx = rng.choice(clf.X_train_.shape[0], min(max_ref, clf.X_train_.shape[0]), replace=False)
        self.ref_X, self.ref_y = clf.X_train_[idx], clf.y_train_[idx]
        self.use_llm = (os.environ.get("ANTHROPIC_API_KEY") is not None) if use_llm is None else use_llm
        self._lex = [(re.compile(rx, re.I), c) for rx, c in LEXICON]

    # ---- tools
    def lexicon(self, desc: str) -> str | None:
        for rx, c in self._lex:
            if rx.search(desc) and c in self.clf.classes_:
                return c
        return None

    def retrieve(self, X_rows, batch: int = 400) -> list[tuple[str, float]]:
        out = []
        for s in range(0, X_rows.shape[0], batch):
            out.extend(self._retrieve(X_rows[s:s + batch]))
        return out

    def _retrieve(self, X_rows) -> list[tuple[str, float]]:
        sims = (X_rows @ self.ref_X.T).toarray()
        out = []
        for row in sims:
            top = np.argsort(-row)[: self.k]
            w = pd.Series(row[top]).groupby(self.ref_y[top]).sum()
            w = w / max(w.sum(), 1e-9)
            out.append((w.idxmax(), float(w.max())))
        return out

    @staticmethod
    def context(df: pd.DataFrame) -> pd.DataFrame:
        """Account-level context signals (requires app_id, date, amount, balance columns)."""
        d = df.copy()
        d["month"] = pd.to_datetime(d["date"]).dt.to_period("M")
        key = d["amount"].round(-1)
        rec = d.groupby(["app_id", key]).month.transform("nunique")
        d["recurring_monthly"] = (rec >= 3).to_numpy()
        d["round_amount"] = (d["amount"].abs() % 100 == 0).to_numpy()
        d["prior_balance"] = (d["balance"] - d["amount"]).to_numpy()
        typical = d[d.amount > 0].groupby("app_id").amount.median()
        d["low_prior_balance"] = (d["prior_balance"] < d["app_id"].map(typical).fillna(0) * 3).to_numpy()
        return d[["recurring_monthly", "round_amount", "low_prior_balance"]]

    def llm_adjudicate(self, desc: str, candidates: list[tuple[str, float]], evidence: dict) -> str | None:  # pragma: no cover
        try:
            import anthropic
        except ImportError:
            return None
        prompt = ("You classify small-business bank transactions for cash-flow underwriting.\n"
                  f"Transaction: {desc}\nModel candidates (label, prob): {candidates}\nEvidence: {json.dumps(evidence)}\n"
                  f"Allowed labels: {list(self.clf.classes_)}\n"
                  'Reply with JSON only: {"label": <one allowed label>, "confident": true|false}')
        msg = anthropic.Anthropic().messages.create(model=os.environ.get("CFU_LLM_MODEL", "claude-sonnet-4-5"),
                                                    max_tokens=100, messages=[{"role": "user", "content": prompt}])
        try:
            r = json.loads(msg.content[0].text)
            return r["label"] if r.get("confident") and r["label"] in self.clf.classes_ else None
        except Exception:
            return None

    # ---- policy (vectorised; the order of tools is the agent's decision procedure)
    def resolve(self, df: pd.DataFrame, proba: np.ndarray, uncertain: np.ndarray | None = None) -> pd.DataFrame:
        """`uncertain` marks rows routed for low model confidence (vs. routed for being generic or material).
        Retrieval/LLM/human review are spent only on uncertain rows; confident rows that pass the
        lexicon and context checks unchanged keep the model label."""
        desc = df["description"].reset_index(drop=True)
        classes = self.clf.classes_
        n = len(desc)
        top2 = np.argsort(-proba, axis=1)[:, :2]
        c1, c2 = classes[top2[:, 0]], classes[top2[:, 1]]
        label = np.full(n, None, dtype=object)
        tool = np.full(n, None, dtype=object)

        # 1) lexicon: first matching high-precision rule wins
        for rx, c in reversed(self._lex):
            if c in classes:
                m = desc.str.contains(rx.pattern, flags=re.I, regex=True).to_numpy()
                label[m], tool[m] = c, "lexicon"
        # 2) account context
        if {"app_id", "balance"} <= set(df.columns):
            ctx = self.context(df).reset_index(drop=True)
            is_credit = desc.str.lower().str.startswith("[credit]").to_numpy()
            # (A "round credit at a low balance => owner transfer" rule was tested and REMOVED: honest
            #  round-dollar customer checks made it misfire and it worsened true-revenue error.)
            is_check = desc.str.match(r"^\[debit\] CHECK \d+").to_numpy()
            m = (tool == None) & ~is_credit & ctx.recurring_monthly.to_numpy() & ((c1 == "rent") | (c2 == "rent") | is_check)  # noqa: E711
            label[m], tool[m] = "rent", "context"
        # 3) retrieval over labelled neighbours (only for still-unresolved, uncertain rows)
        unc = np.ones(n, bool) if uncertain is None else np.asarray(uncertain, bool)
        keep = (tool == None) & ~unc  # noqa: E711
        label[keep], tool[keep] = c1[keep], "model_checked"
        todo = np.where(tool == None)[0]  # noqa: E711
        if len(todo):
            nn = self.retrieve(self.clf.transform(desc.iloc[todo]))
            nn_lab = np.array([x[0] for x in nn], dtype=object)
            nn_w = np.array([x[1] for x in nn])
            conf_ok = (nn_lab == c1[todo]) & (nn_w >= 0.5)
            over = (nn_lab == c2[todo]) & (nn_w >= 0.8)
            label[todo[conf_ok]], tool[todo[conf_ok]] = c1[todo[conf_ok]], "retrieval_confirm"
            label[todo[over]], tool[todo[over]] = c2[todo[over]], "retrieval_override"
        # 4) optional LLM adjudication, then 5) human review
        todo = np.where(tool == None)[0]  # noqa: E711
        if self.use_llm:  # pragma: no cover
            for i in todo:
                lab = self.llm_adjudicate(desc.iloc[i], [(classes[j], round(float(proba[i, j]), 3)) for j in top2[i]], {})
                if lab:
                    label[i], tool[i] = lab, "llm"
        todo = tool == None  # noqa: E711
        label[todo], tool[todo] = c1[todo], "model_fallback"
        return pd.DataFrame({"label": label, "tool": tool, "needs_review": todo})


# Low-information descriptors: a confident model score on these is not trustworthy evidence.
GENERIC = re.compile(r"^\[(credit|debit)\] (DEPOSIT$|CHECK \d+|ZELLE |ONLINE TRANSFER|WIRE TRANSFER|BOOK TRANSFER|"
                     r"REMOTE ONLINE DEPOSIT|MOBILE DEPOSIT|COUNTER DEPOSIT|ATM )", re.I)


def route_mask(df: pd.DataFrame, conf: np.ndarray, threshold: float) -> np.ndarray:
    """Route a row to the agent if (a) the model is unsure, (b) the descriptor is generic, or
    (c) it is a MATERIAL credit (> 20% of the account's average monthly credits) - lumps like loan
    proceeds, owner injections or fabricated deposits move 'true revenue' the most."""
    m = conf < threshold
    m |= df["description"].str.match(GENERIC).to_numpy()
    if {"app_id", "amount", "date"} <= set(df.columns):
        months = df.groupby("app_id").date.transform(lambda d: max(1, pd.to_datetime(d).dt.to_period("M").nunique()))
        credits = df["amount"].clip(lower=0).groupby(df["app_id"]).transform("sum") / months
        m |= (df["amount"] > 0.2 * credits).to_numpy()
    return m


def classify_with_routing(clf: TxnClassifier, agent: TransactionAgent, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    proba = clf.predict_proba(df["description"])
    conf = proba.max(1)
    out = pd.DataFrame({"model_label": clf.classes_[proba.argmax(1)], "confidence": conf})
    out["label"], out["tool"], out["needs_review"] = out["model_label"], "model", False
    low = np.where(route_mask(df, conf, threshold))[0]
    if len(low):
        res = agent.resolve(df.iloc[low].reset_index(drop=True), proba[low], uncertain=conf[low] < threshold)
        out.loc[low, ["label", "tool", "needs_review"]] = res[["label", "tool", "needs_review"]].to_numpy()
    out["needs_review"] = out["needs_review"].astype(bool)
    return out


def routing_report(y_true, routed: pd.DataFrame) -> dict:
    y = np.asarray(y_true)
    auto = routed.tool.eq("model").to_numpy()
    rep = {"n": int(len(y)),
           "model_only_accuracy": float(accuracy_score(y, routed.model_label)),
           "model_only_macro_f1": float(f1_score(y, routed.model_label, average="macro")),
           "routed_accuracy": float(accuracy_score(y, routed.label)),
           "routed_macro_f1": float(f1_score(y, routed.label, average="macro")),
           "auto_accept_coverage": float(auto.mean()),
           "auto_accept_accuracy": float((routed.model_label.to_numpy()[auto] == y[auto]).mean()) if auto.any() else None,
           "routed_to_agent": int((~auto).sum()),
           "agent_rows_model_accuracy": float((routed.model_label.to_numpy()[~auto] == y[~auto]).mean()) if (~auto).any() else None,
           "agent_rows_agent_accuracy": float((routed.label.to_numpy()[~auto] == y[~auto]).mean()) if (~auto).any() else None,
           "human_review_rate": float(routed.needs_review.mean()),
           "tool_mix": routed.tool.value_counts(normalize=True).round(4).to_dict()}
    return rep
