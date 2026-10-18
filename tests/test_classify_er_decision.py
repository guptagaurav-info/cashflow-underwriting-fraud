import numpy as np
import pandas as pd

from cfu import entity_resolution as er
from cfu.classify import GENERIC, TransactionAgent, TxnClassifier, normalise, route_mask
from cfu.config import load_config
from cfu.decision import debt_capacity_offer


def test_normalise_strips_reference_numbers():
    s = normalise(pd.Series(["[debit] FORA FINANCIAL ACH DEBIT 123456789", "[debit] FORA FINANCIAL ACH DEBIT 987654321"]))
    assert s.iloc[0] == s.iloc[1]


def test_generic_descriptors_are_routed():
    df = pd.DataFrame({"description": ["[credit] DEPOSIT", "[debit] CHECK 1043", "[credit] SQUARE INC DEPOSIT X1"]})
    m = route_mask(df, np.array([0.99, 0.99, 0.99]), 0.9)
    assert m.tolist() == [True, True, False]


def test_agent_lexicon_beats_confident_model_on_funder_credit():
    tr = pd.DataFrame({"description": ["[credit] WIRE TRANSFER IN ACME CORP"] * 30 + ["[debit] GUSTO NET PAY"] * 30
                       + ["[credit] ONDECK CAPITAL FUNDING"] * 3, "y": ["revenue"] * 30 + ["payroll"] * 30 + ["loan_proceeds"] * 3})
    clf = TxnClassifier().fit(tr.description, tr.y)
    agent = TransactionAgent(clf)
    assert agent.lexicon("[credit] WIRE TRANSFER IN ONDECK CAPITAL") == "loan_proceeds"


def test_entity_resolution_variants_and_rings():
    apps = pd.DataFrame({
        "network_app_id": range(5), "app_id": [0, None, None, 3, 4], "business_id": [0, 0, 0, 1, 2],
        "lender_id": ["US", "L1", "L2", "US", "US"],
        "app_date": pd.to_datetime(["2025-01-10", "2025-01-05", "2025-01-20", "2025-01-10", "2025-01-10"]),
        "applicant_name": ["ATLAS PLUMBING LLC", "Atlas Plumbing L.L.C.", "ATLAS PLUMBNIG", "SUMMIT DENTAL INC", "RIVERSIDE GRILL LLC"],
        "phone": ["111", "111", "999", "222", "222"], "bank_account": ["A1", "A1", "A1", "B1", "B1"],
        "zip": ["10001", "10001", "10001", "20002", "30003"], "is_primary": [True, False, False, True, True]})
    ent, _ = er.resolve(apps)
    assert ent[0] == ent[1] == ent[2]            # name variants of one business
    assert ent[3] != ent[4]                      # shared account, different names -> NOT merged...
    net = er.network_features(apps, ent).set_index("app_id")
    assert net.loc[0, "er_lenders_14d"] == 1     # only the PRIOR application (Jan 5) is visible on Jan 10
    assert net.loc[3, "er_shared_account_entities"] == 1   # ...but flagged as a shared-identifier ring


def test_debt_capacity_offer():
    cfg = load_config()
    row = pd.Series({"operating_cf_avg_6m": 20000.0, "debt_service_avg_6m": 5000.0, "requested_amount": 500000.0})
    # free CF 15,000 / 1.25 DSCR = 12,000 max payment -> x 12 months / 1.25 factor = 115,200 -> floor to 500
    assert debt_capacity_offer(row, cfg) == 115000.0
