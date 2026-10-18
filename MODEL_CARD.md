# Model Card: Cash-Flow PD + Fraud Gate (synthetic SMB book)

| | PD model | Fraud model |
|---|---|---|
| **Task** | 12-month default on an MCA/term advance | Probability that the application is fraudulent (document tampering, fabrication, stacking, identity rings) |
| **Algorithm** | LightGBM with monotone constraints + isotonic calibration (out-of-fold) | LightGBM + isotonic calibration; benchmarked against rules and an Isolation Forest |
| **Features** | Bureau score, years in business, industry, request size + 25 cash-flow features from classified transactions | 8 document-integrity + 7 network / entity-resolution features |
| **Training data** | Non-fraud applications, Jan-2024 to Jun-2025 (n = 4,199) | All applications, Jan-2024 to Jun-2025 (n = 4,422, 5.0% fraud) |
| **OOT performance** | AUC 0.844 (vs 0.659 bureau-only), Brier 0.089 | PR-AUC 0.96, recall 97% at 5% review |
| **Decision use** | PD ≤ 12% approve; 12–30% refer; ≥ 30% decline. Offer = debt-capacity limit | Score ≥ 0.30 → refer to the fraud team before credit |

**Known limitations.**

- Simulated outcomes and documents.
- Fraud typology is limited to the five simulated patterns; novel tactics need the unsupervised and rule layers plus analyst feedback.
- The stacking catch rate is capped by point-in-time network visibility.

**Monitoring plan.**

- Feature PSI and classifier routing mix (auto-accept %, review %), weekly.
- Calibration by PD decile as performance matures.
- Fraud-model precision on referred cases.
- Velocity-feature drift as network coverage grows.
