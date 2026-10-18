import duckdb
import numpy as np
import pandas as pd
import pytest

from cfu import simulate
from cfu.config import load_config, p


@pytest.fixture(scope="module")
def book():
    cfg = load_config()
    cfg["simulation"]["fraud_rate"] = 0.3          # oversample fraud so every type appears
    return simulate.simulate(cfg, n=120)


def test_book_shapes(book):
    lab, tx = book["labels"], book["transactions"]
    assert len(lab) == 120 and lab.is_default.between(0, 1).all()
    assert set(tx.columns) >= {"app_id", "date", "amount", "description", "category", "balance"}
    assert tx.description.str.match(r"^\[(credit|debit)\] ").all()
    assert ((tx.amount > 0) == tx.description.str.startswith("[credit]")).all()


def _header_gap(book):
    s = book["statements"]
    gap = (s.opening_balance + s.total_credits - s.total_debits - s.closing_balance).abs() / s.total_credits
    return s.assign(gap=gap).groupby("app_id").gap.max()


def test_statements_reconcile_unless_naively_edited(book):
    g = _header_gap(book)
    ft = book["labels"].set_index("app_id").fraud_type
    assert (g[ft[ft == "none"].index] < 1e-6).all()                # honest statements tie out
    naive = ft[ft == "inflated_naive"].index
    if len(naive):
        assert (g[naive] > 0.01).all()                              # naive edits break the arithmetic


def test_fraud_selects_on_weaker_outcomes(book):
    lab = book["labels"]
    assert lab[lab.is_fraud == 1].is_default.mean() > lab[lab.is_fraud == 0].is_default.mean()


def test_document_features_sql(book, tmp_path):
    for k in ("transactions", "statements"):
        book[k].to_parquet(tmp_path / f"{k}.parquet")
    con = duckdb.connect()
    con.execute(f"SET VARIABLE txn_path = '{tmp_path / 'transactions.parquet'}'")
    con.execute(f"SET VARIABLE stmt_path = '{tmp_path / 'statements.parquet'}'")
    con.execute(p("sql/02_document_fraud_features.sql").read_text())
    d = con.sql("SELECT * FROM doc_features").df().merge(book["labels"], on="app_id")
    honest = d[d.fraud_type == "none"]
    assert (honest.running_balance_break_share < 0.01).all()
    assert d.benford_mad.between(0, 1).all()
