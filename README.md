# Cash-Flow Underwriting & Fraud: From Bank Transactions to a Credit Decision

> Companion repo: [smb-credit-pd-lgd](https://github.com/guptagaurav-info/smb-credit-pd-lgd) (PD / LGD / expected-loss pricing on 772K real SBA loans). Author: Gaurav Gupta.

An end-to-end small-business lending pipeline, built the way a cash-flow underwriting platform works. Each application goes through these steps:

1. Raw bank-statement transactions are classified, with **confidence routing to an agent step**.
2. SQL builds cash-flow features in `{category}_{flow}_{stat}_{window}` form.
3. Document-integrity checks and **entity resolution** across a lender network run next.
4. A **fraud gate** screens the application.
5. A PD model scores it.
6. **Debt-capacity offer sizing** sets the amount, and an analyst-facing memo is written.

```bash
pip install -r requirements.txt
make all     # downloads public data (sha256-verified), simulates the book, runs everything (~6 min)
make test
```

## Data: what is real and what is simulated, and why

| Layer | Source | Used for |
|---|---|---|
| Bank-transaction descriptors (68K, 17 categories, MIT) | [`DoDataThings/us-bank-transaction-categories-v2`](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2), modelled on real US bank formats (Chase ACH, `TST*`, `PP*`…) | Public classifier benchmark |
| PaySim (6.36M mobile-money transactions with fraud labels) | [`theman10/paysim`](https://huggingface.co/datasets/theman10/paysim) | Fraud benchmark + a **data-validity check** (see below) |
| SMB lending book: 6,000 applicants, 1.72M transactions, 8,237 network applications | `src/cfu/simulate.py` | Full pipeline |

**Why simulate:** no public dataset links small-business bank transactions to loan outcomes. The simulator encodes **mechanisms**, not labels:

- **Default is not drawn from a formula of the features.** Each business's cash is simulated forward 12 months, using:
  - its true revenue process, margin and seasonality
  - existing MCA daily debits and term loans
  - the new loan payment
  - unobserved shocks: customer loss, cost shocks, owner liquidity

  Default means the business runs out of cash. Features see only the 6 months of statements submitted at application.
- **Fraud edits the document the lender sees, while the outcome follows the true business.** Five types:
  - `inflated_naive`: revenue scaled up, balances not fixed
  - `inflated_sophisticated`: fabricated "mobile deposits", balances recomputed, edited PDF
  - `fabricated`: template statement; shell business in a ring sharing a bank account and phone
  - `stacking`: genuine statements, but several near-simultaneous applications under name variants plus fresh MCA fundings
  - honest borrowers, who also produce noise: they rate-shop, deposit round-dollar checks and re-save PDFs
- 6,000 applications: 14.1% default (honest), 4.8% fraud.

## Results (out-of-time: train on applications ≤ Jun-2025, test on Jul–Dec-2025, n = 1,578)

### 1. Transaction classification with an agent step

**Public benchmark: the split matters more than the model.**

| Test split | Model accuracy | Auto-accepted (conf ≥ 0.90) | Accuracy of auto-accepted |
|---|---|---|---|
| Random 80/20 | 99.9% | 93% | 100% |
| **Unseen-merchant holdout** | **82.5%** | **53%** | **98.6%** |

A random split leaks merchant names across train and test. The merchant holdout is what production looks like. Confidence routing still gives a **98.6%-accurate fast lane for half the volume** and sends the rest to the agent or human queue (`reports/figures/classifier_coverage_accuracy.png`).

**On SMB statements (1.43M transactions), routing goes beyond "low confidence".** It also routes **generic descriptors** ("DEPOSIT", "CHECK 1043", "ZELLE FROM…") and **material credits** (over 20% of monthly credits), because those move *true revenue* the most. The `TransactionAgent` tools run in a fixed order:

1. high-precision lexicon
2. account context (e.g. a check for the same amount every month is rent)
3. nearest-neighbour retrieval
4. optional **LLM adjudication** (Claude, if `ANTHROPIC_API_KEY` is set)
5. human review

| | Model only | Model + agent |
|---|---|---|
| **Loan-proceeds recall** (funding credits misread as revenue) | 66% | **100%** |
| Accounts with > 5% error in true revenue | 1.9% | **0.4%** |
| Human-review rate | — | 0.05% of transactions |

The reported results use the deterministic tools. An account-context rule ("round credit at a low balance means owner transfer") was tested and **removed**: honest round-dollar customer checks made it misfire.

### 2. Cash-flow PD vs bureau-only (12-month default, non-fraud test applicants)

| Model | AUC | Gini | Brier |
|---|---|---|---|
| Bureau score + firmographics | 0.659 | 0.32 | 0.121 |
| **+ cash-flow features** | **0.844** | **0.69** | **0.089** |

- **Top drivers** (`reports/figures/pd_shap.png`): liquidity runway (days), requested amount ÷ monthly revenue, pro-forma DSCR including the new loan, payroll ÷ revenue, average balance ÷ revenue.
- **Monotone constraints** cover bureau score, NSFs, negative-balance days, DSCR, runway and debt-service burden.
- **Calibration**: isotonic, fitted on out-of-fold predictions inside the training window.

### 3. Fraud is a gate, not a feature

Confirmed-fraud applications are **excluded from PD training**, because doctored statements make weak businesses look healthy:

| Fraud type | Mean PD the credit model assigns | Observed default |
|---|---|---|
| inflated_sophisticated | **3.5%** | 52% |
| inflated_naive | 8.1% | 55% |
| stacking | 20.1% | 60% |
| none | 13.2% | 15% |

**Detectors** at the same review budget:

| Detector | PR-AUC | Recall @ 5% review |
|---|---|---|
| Explainable rules (9 red flags) | 0.56 | 57% |
| Isolation Forest (unsupervised) | 0.64 | 68% |
| **Supervised GBM** (document + network features) | **0.96** | **97%** |

- At the 0.30 threshold the false-positive rate is 0.3%.
- Catch rate by type: fabricated 100%, inflated-naive 100%, inflated-sophisticated 95%, **stacking 75%**.
- Stacking is the hardest. Point-in-time, the lender only sees applications that were made *before* this one; the rest show up in post-funding monitoring.

**Document checks** (`sql/02_document_fraud_features.sql`):

- running-balance arithmetic
- statement headers that don't tie out
- PDF producer / edit gap
- round-thousand credits
- Benford deviation
- template repetition across months

**Entity resolution** (`src/cfu/entity_resolution.py`):

- blocking on account / phone / ZIP+name, then rapidfuzz scoring and union-find clustering
- **pairwise precision 1.00, recall 0.986**
- produces 14-day and 90-day **multi-lender velocity** and **shared-identifier rings** (one bank account behind "different" businesses)

### 4. Decision policy (test window, illustrative economics)

| Policy | Funded | Bad rate | Fraud funded | P&L | Return on funded |
|---|---|---|---|---|---|
| Bureau-only PD | 790 | 7.5% | 25 | $4.5M | 10.8% |
| Cash-flow PD, **no fraud gate** | 1,129 | 7.1% | 47 | $6.1M | 10.3% |
| **Fraud gate → cash-flow PD → debt-capacity offer** | **1,083** | **5.3%** | **4** | **$7.3M** | **13.2%** |

- **Offers are sized from debt capacity**: free operating cash flow after existing debt, divided by a 1.25× target DSCR, sets the maximum payment, which sets the advance. The offer can come in below the request.
- **Decisions**: APPROVE 69%, DECLINE 12%, DECLINE (capacity) 8%, REFER credit 7%, REFER fraud 4%.
- **Sample memos** are in `reports/memos/` (HTML + JSON). Each shows key cash-flow metrics, the top risk drivers, and the fraud red flags that fired.

### 5. A real-data trap: PaySim's balance fields

Ledger arithmetic is a strong fraud signal on bank statements, so it was tested on PaySim first. **90.5% of *legitimate* PaySim transactions fail origin-balance reconciliation, versus 0.6% of fraudulent ones.** That is a simulator artefact, not a fraud mechanism. A model fed those fields learns the artefact and is unstable out of time, so the fields were **rejected**. The attribute-only PaySim model reaches an out-of-time AUC of 0.97 (PR-AUC 0.87).

## Repo map

```
src/cfu/
  ingest.py              download + sha256 verification
  simulate.py            mechanism-driven SMB lending book (documents vs truth)
  classify.py            TF-IDF + logistic classifier, confidence/materiality routing, TransactionAgent
  entity_resolution.py   blocking, fuzzy matching, union-find, point-in-time network features
  models.py              calibrated monotone GBMs (PD), rules / Isolation Forest / GBM (fraud)
  decision.py            fraud gate -> PD bands -> debt-capacity offer, memo rendering
  paysim.py              PaySim benchmark + validity check
  pipeline.py            orchestration, metrics, figures
sql/
  01_cashflow_features.sql
  02_document_fraud_features.sql
reports/                 metrics.json, figures/, memos/
```

## Limitations

- The SMB book is simulated. The numbers show the method and relative lift, not market-level performance.
- Fraud labels are perfectly known here; in production they arrive late and noisy (confirmed vs suspected).
- There is no reject inference: every simulated applicant's outcome is observed.
- Fraud sample sizes are small: 63 fraudulent applications in the test window.

---

*Built as a portfolio project. All applicant names, accounts and phone numbers are synthetic.*
