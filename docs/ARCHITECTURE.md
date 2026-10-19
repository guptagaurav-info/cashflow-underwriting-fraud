# How this project works

There are four diagrams:

1. **Pipeline:** every step, its source file, and the logic inside it.
2. **Transaction classification with agent routing:** how each bank transaction gets its category.
3. **Decision logic for one application:** fraud gate, then PD, then offer.
4. **Production design:** how this would run live. It is a proposed design, not implemented in this repo.

Colour key: **blue** = external data source · **purple** = simulated data · **lavender** = code step (file named in the box) · **yellow** = model · **red** = check or gate · **grey** = output

## 1. Pipeline (what `make all` runs)

```mermaid
flowchart TD
    S1["SOURCE: US bank-transaction descriptors, 68,000 rows, 17 categories<br/>huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2 (MIT)"]
    S2["SOURCE: PaySim mobile-money transactions, 6.36M rows with fraud labels<br/>huggingface.co/datasets/theman10/paysim"]
    S3["SIMULATED: SMB lending book, src/cfu/simulate.py<br/>6,000 businesses · 8 industries · 1.72M transactions · 8,237 network applications"]

    subgraph ING["Ingest · src/cfu/ingest.py"]
      I1{"sha256 matches config?"}
    end

    subgraph PS["Real-data fraud check · src/cfu/paysim.py"]
      P1["Validity test: 90.5% of LEGIT transactions fail balance reconciliation<br/>vs 0.6% of fraud, so it is a simulator artefact and the feature is rejected"]
      P2["Attribute-only LightGBM, out of time (hour ≤ 500 train)<br/>AUC 0.97 · PR-AUC 0.87"]
      P1 --> P2
    end

    subgraph CL["Public classifier benchmark · pipeline.stage_public_classifier"]
      C1["TF-IDF (char 2–5 + word 1–2) → logistic regression"]
      C2["Random split 99.9% vs unseen-merchant split 82.5%<br/>Auto-accept at conf ≥ 0.90: 53% of rows at 98.6% accuracy"]
      C1 --> C2
    end

    subgraph SIM["Simulator logic · simulate.py"]
      M1["Each business: industry, latent health, revenue process<br/>(trend, seasonality, shocks), margin, existing MCA / term debt"]
      M2["6 months of statements: card & ACH deposits, payroll, rent,<br/>suppliers, MCA daily debits, owner draws, NSF fees"]
      M3["Outcome = forward 12-month CASH simulation with the new payment<br/>default when cash runs out (not a formula of the features)"]
      M4["Fraud edits the DOCUMENT only: inflated deposits, fabricated statements,<br/>stacking bursts, shared-account rings; outcome follows the true business"]
      M1 --> M2 --> M3
      M2 --> M4
    end

    subgraph CLS["SMB classification · pipeline.stage_smb_classify + classify.py"]
      K1["Train on 1,000 'hand-labelled' training-period accounts"]
      K2["Classify 1.43M other transactions with routing (diagram 2)"]
      K1 --> K2
    end

    subgraph FE["Features in SQL (DuckDB)"]
      F1["sql/01_cashflow_features.sql<br/>{category}_{flow}_{stat}_{window}: revenue avg/CV/trend, transfer share,<br/>MCA/revenue, distinct funders, NSFs, negative days, min balance,<br/>runway days, DSCR existing & pro-forma, ask / revenue"]
      F2["sql/02_document_fraud_features.sql<br/>running-balance breaks, header tie-out, PDF producer & edit gap,<br/>round-thousand credits, Benford MAD, repeated amounts"]
    end

    subgraph ER["Entity resolution · entity_resolution.py"]
      E1["Block on account · phone · ZIP + first name token"]
      E2["Name normalisation + rapidfuzz token-set ratio<br/>(match ≥ 80 on account/phone, ≥ 90 on ZIP)"]
      E3["Union-find clusters → entity_id<br/>pairwise precision 1.00 · recall 0.986"]
      E4["Point-in-time network features: lenders applied to in 14 / 90 days,<br/>other businesses sharing account / phone"]
      E1 --> E2 --> E3 --> E4
    end

    SPLIT["Out-of-time split: train apps ≤ Jun-2025 (4,422) · test Jul–Dec-2025 (1,578)"]

    subgraph MOD["Models · models.py"]
      FR["Fraud: 9 explainable rules · Isolation Forest · supervised LightGBM<br/>(calibrated with out-of-fold isotonic)"]
      PD["PD: calibrated monotone LightGBM, trained on NON-FRAUD apps only<br/>bureau-only baseline vs bureau + 25 cash-flow features"]
    end

    DEC["Decision · decision.py (diagram 3)"]
    OUT["reports/metrics.json · figures/*.png · memos/*.html + *.json"]

    S1 --> I1
    S2 --> I1
    I1 --> PS
    I1 --> CL
    S3 --> SIM --> CLS --> FE
    S3 --> ER
    FE --> SPLIT
    ER --> SPLIT
    SPLIT --> MOD --> DEC --> OUT

    classDef src fill:#dbe8ff,stroke:#1f5fbf,color:#0b1f44
    classDef sim fill:#ece4ff,stroke:#6b46c1,color:#2a1259
    classDef model fill:#fff3c4,stroke:#b7791f,color:#3d2a00
    classDef gate fill:#fde2e0,stroke:#c53030,color:#4a0d0d
    classDef out fill:#eceff3,stroke:#5b6578,color:#1d2433
    class S1,S2 src
    class S3 sim
    class I1,P1 gate
    class P2,C1,FR,PD model
    class OUT out
```

## 2. Transaction classification with agent routing (classify.py)

```mermaid
flowchart TD
    T["Bank transaction<br/>e.g. '[credit] WIRE TRANSFER IN ONDECK CAPITAL'"] --> N["Normalise: upper-case, mask long numbers and digits"]
    N --> M["TF-IDF + logistic regression → category + confidence"]
    M --> R{"Route to agent?<br/>confidence < 0.90<br/>OR generic text (DEPOSIT, CHECK, ZELLE, TRANSFER)<br/>OR material credit (> 20% of monthly credits)"}
    R -- "no (84%)" --> A["Accept model label<br/>(100% accurate on this lane)"]
    R -- "yes (16%)" --> L{"Lexicon rule hits?<br/>funder names, payroll processors,<br/>NSF/overdraft fees, IRS/tax, loan payments"}
    L -- "yes" --> LA["Label from rule"]
    L -- "no" --> CX{"Account context?<br/>same-amount debit every month → rent"}
    CX -- "yes" --> CA["Label from context"]
    CX -- "no" --> U{"Model uncertain<br/>(conf < 0.90)?"}
    U -- "no" --> MC["Keep model label (checked)"]
    U -- "yes" --> RT{"Nearest labelled neighbours agree?<br/>confirm top-1 (vote ≥ 0.5) or<br/>override to top-2 (vote ≥ 0.8)"}
    RT -- "yes" --> RA["Label from retrieval"]
    RT -- "no" --> LLM{"LLM available?<br/>(ANTHROPIC_API_KEY)"}
    LLM -- "yes, confident" --> LL["Label from Claude adjudication"]
    LLM -- "no / unsure" --> H["Model label + FLAG for human review<br/>(0.05% of transactions)"]

    classDef model fill:#fff3c4,stroke:#b7791f,color:#3d2a00
    classDef gate fill:#fde2e0,stroke:#c53030,color:#4a0d0d
    classDef out fill:#eceff3,stroke:#5b6578,color:#1d2433
    class M,LLM model
    class R,L,CX,U,RT gate
    class H out
```

Result on 1.43M SMB transactions: loan-proceeds recall goes from 66% to 100%, and accounts with more than 5% error in true revenue fall from 1.9% to 0.4%. A "round credit at a low balance means owner transfer" context rule was tested and removed because it misfired on honest round-dollar checks.

## 3. Decision logic for one application (decision.py)

```mermaid
flowchart TD
    A["Application + 6 months of statements"] --> F["Fraud score (supervised LightGBM)<br/>document checks + network / entity-resolution features"]
    F --> G{"Fraud score ≥ 0.30?"}
    G -- "yes" --> RF["REFER_FRAUD<br/>(fired rules listed in the memo)"]
    G -- "no" --> P["PD from cash-flow model<br/>+ top-3 SHAP reason codes"]
    P --> D1{"PD ≥ 0.30?"}
    D1 -- "yes" --> DC["DECLINE"]
    D1 -- "no" --> O["Debt-capacity offer:<br/>free operating cash flow after existing debt ÷ 1.25 DSCR = max payment<br/>× 12-month term ÷ 1.25 factor = max advance<br/>offer = min(request, max advance), floored to $500"]
    O --> D2{"Offer < $5,000?"}
    D2 -- "yes" --> DCC["DECLINE_CAPACITY"]
    D2 -- "no" --> D3{"PD ≤ 0.12?"}
    D3 -- "yes" --> AP["APPROVE at offer amount"]
    D3 -- "no" --> RC["REFER_CREDIT (analyst review)"]
    RF --> MEMO["Memo: HTML + JSON<br/>key cash-flow metrics · risk drivers · fraud flags"]
    DC --> MEMO
    DCC --> MEMO
    AP --> MEMO
    RC --> MEMO

    classDef model fill:#fff3c4,stroke:#b7791f,color:#3d2a00
    classDef gate fill:#fde2e0,stroke:#c53030,color:#4a0d0d
    classDef out fill:#eceff3,stroke:#5b6578,color:#1d2433
    class F,P model
    class G,D1,D2,D3 gate
    class MEMO,RF,DC,DCC,AP,RC out
```

Why fraud is a gate and not a PD feature: doctored statements make weak businesses look healthy. The credit model gives "inflated_sophisticated" fraud a 3.5% PD, but 52% of those applicants default. Fraud cases are excluded from PD training. Compared with no gate, the gate cuts fraud funded from 47 to 4 and the bad rate from 7.1% to 5.3%.

## 4. Production design (proposed; not implemented in this repo)

```mermaid
flowchart TD
    subgraph IN["Ingestion"]
      U1["Statement upload / bank connection"] --> U2["Document capture + extraction<br/>(e.g. OCR / Capture service)"]
      U2 --> U3["Transactions table (warehouse)"]
    end

    subgraph SC["Scoring path (per application, seconds)"]
      U3 --> X1["Classification service<br/>fast model + merchant cache; agent / LLM only for routed rows"]
      X1 --> X2["Feature service: same SQL definitions as sql/*.sql,<br/>computed as of the application timestamp"]
      X2 --> X3["Document-integrity checks + entity resolution against network"]
      X3 --> X4["Fraud model → gate"]
      X4 --> X5["PD model → offer sizing → decision"]
      X5 --> X6["Decision + memo + reason codes to lender (API)"]
    end

    subgraph TR["Training path (offline, monthly)"]
      T1["Point-in-time snapshots of features + matured outcomes"] --> T2["Retrain challengers (same code as models.py)"]
      T2 --> T3{"Validation gate: OOT AUC / PR-AUC,<br/>calibration, PSI, fraud recall at review budget"}
      T3 -- "pass" --> T4["Registry: model + config + data hash"]
      T4 -- "shadow → champion/challenger" --> X4
      T4 --> X5
    end

    subgraph MO["Monitoring & feedback"]
      Q1["Routing mix: auto-accept % · agent % · review %"]
      Q2["Score / feature PSI, latency, cost per statement"]
      Q3["Outcomes: default, early-payment default, confirmed fraud"]
      Q4["Analyst reviews → labels for classifier and fraud model"]
    end

    X1 --> Q1
    X5 --> Q2
    X6 --> Q3
    Q3 --> T1
    Q4 --> T1

    classDef gate fill:#fde2e0,stroke:#c53030,color:#4a0d0d
    classDef model fill:#fff3c4,stroke:#b7791f,color:#3d2a00
    classDef out fill:#eceff3,stroke:#5b6578,color:#1d2433
    class T3 gate
    class X1,X4,X5,T2 model
    class X6 out
```

## How to run

| Step | Command | Time |
|---|---|---|
| Install | `pip install -r requirements.txt` | ~1 min |
| Everything | `make all` (or `PYTHONPATH=src python -m cfu.pipeline`) | ~10 min first run (simulates the book), ~6 min after |
| Tests | `make test` | seconds |
| Optional LLM step | set `ANTHROPIC_API_KEY` and `pip install anthropic` | — |
| Thresholds | edit `configs/config.yaml` (`decision:` block), rerun | — |
