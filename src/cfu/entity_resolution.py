"""Entity resolution across the lender network: which applications are the same business?

Blocking -> pairwise scoring -> union-find clustering. Each match rule is explainable:
    same bank account   + name similarity >= 80
    same phone          + name similarity >= 80
    same ZIP            + name similarity >= 90
Names are normalised (case, punctuation, legal suffixes, DBA clauses) before fuzzy matching.

Two features come out of the clusters:
    * application velocity: distinct lenders the SAME entity applied to around this application (stacking)
    * shared identifiers:   distinct entities that use the same bank account / phone (synthetic-identity rings)
"""
from __future__ import annotations

import re
from itertools import combinations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

SUFFIXES = r"\b(LLC|L L C|INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LTD)\b"


def normalise_name(s: str) -> str:
    s = s.upper()
    s = re.split(r"\bDBA\b", s)[0]
    s = s.replace("&", " AND ").replace(".", " ")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(SUFFIXES, " ", s)
    return re.sub(r"\s+", " ", s).strip()


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def resolve(apps: pd.DataFrame, max_block: int = 200) -> tuple[pd.Series, pd.DataFrame]:
    """Return entity_id per application row and the scored candidate pairs."""
    a = apps.reset_index(drop=True)
    names = a["applicant_name"].map(normalise_name).to_numpy()
    first = np.array([n.split(" ")[0] if n else "" for n in names])
    blocks = {
        "bank_account": a["bank_account"].to_numpy(),
        "phone": a["phone"].to_numpy(),
        "zip_first_token": (a["zip"].astype(str) + "|" + first),
    }
    thresholds = {"bank_account": 80, "phone": 80, "zip_first_token": 90}
    pairs = {}
    for key, vals in blocks.items():
        s = pd.Series(np.arange(len(a))).groupby(pd.Series(vals).to_numpy())
        for _, idx in s:
            idx = idx.to_numpy()
            if len(idx) < 2 or len(idx) > max_block:
                continue
            for i, j in combinations(idx, 2):
                sim = fuzz.token_set_ratio(names[i], names[j])
                k = (min(i, j), max(i, j))
                rec = pairs.setdefault(k, {"i": k[0], "j": k[1], "name_sim": sim, "rules": set()})
                if sim >= thresholds[key]:
                    rec["rules"].add(key)
    uf = UnionFind(len(a))
    for (i, j), r in pairs.items():
        if r["rules"]:
            uf.union(i, j)
    entity = pd.Series([uf.find(i) for i in range(len(a))], index=apps.index, name="entity_id")
    pdf = pd.DataFrame([{**r, "matched": bool(r["rules"]), "rules": ",".join(sorted(r["rules"]))} for r in pairs.values()])
    return entity, pdf


def evaluate(apps: pd.DataFrame, entity: pd.Series) -> dict:
    """Pairwise precision / recall against the true business id (available only in simulation)."""
    true = apps["business_id"].to_numpy()
    pred = entity.to_numpy()
    df = pd.DataFrame({"t": true, "p": pred})
    tp = sum(len(g) * (len(g) - 1) / 2 for _, g in df.groupby(["t", "p"]))
    pred_pairs = sum(len(g) * (len(g) - 1) / 2 for _, g in df.groupby("p"))
    true_pairs = sum(len(g) * (len(g) - 1) / 2 for _, g in df.groupby("t"))
    return {"pairwise_precision": tp / max(pred_pairs, 1), "pairwise_recall": tp / max(true_pairs, 1),
            "true_entities": int(df.t.nunique()), "resolved_entities": int(df.p.nunique())}


def network_features(apps: pd.DataFrame, entity: pd.Series, window_days: int = 14) -> pd.DataFrame:
    """Per primary application: velocity of the resolved entity and identifier sharing.

    Uses only applications dated ON OR BEFORE the primary application's date (point-in-time):
    the lender cannot see an application that has not happened yet."""
    a = apps.assign(entity_id=entity.to_numpy())
    a["app_date"] = pd.to_datetime(a["app_date"])
    prim = a[a.is_primary]
    rows = []
    by_ent = {k: g for k, g in a.groupby("entity_id")}
    for _, r in prim.iterrows():
        g = by_ent[r.entity_id]
        past = g[(g.app_date <= r.app_date) & (g.network_app_id != r.network_app_id)]
        recent = past[past.app_date >= r.app_date - pd.Timedelta(days=window_days)]
        past90 = past[past.app_date >= r.app_date - pd.Timedelta(days=90)]
        same_acct = a[(a.bank_account == r.bank_account) & (a.app_date <= r.app_date)]
        same_phone = a[(a.phone == r.phone) & (a.app_date <= r.app_date)]
        rows.append({"app_id": int(r.app_id), "entity_id": int(r.entity_id),
                     f"er_lenders_{window_days}d": int(recent.lender_id.nunique()),
                     "er_lenders_90d": int(past90.lender_id.nunique()),
                     "er_shared_account_entities": int(same_acct.entity_id.nunique() - 1),
                     "er_shared_phone_entities": int(same_phone.entity_id.nunique() - 1)})
    return pd.DataFrame(rows)
