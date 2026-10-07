# Final model selection (common final test set)

Synthetic data only (100,000 transactions, seed 2026, 5.22% fraud). Not real upay production performance. All thresholds were selected on each model's own validation threshold block; the final test set was used once for reporting.

## Decision table

| Model | PR-AUC [95% CI] | Δ PR-AUC vs best [95% CI] | Recall | Precision | FPR | P95 ms (batch 1) | Tiered business cost | Components | Size MB | Recommendation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LightGBM_200k | 0.2417 [0.2265, 0.2577] | +0.0000 [+0.0000, +0.0000] | 0.291 | 0.370 | 0.0273 | 10.90 | 3,834,605 | 1 | 0.11 | SELECTED (primary) |
| Hybrid_WithoutSequence_500k | 0.2415 [0.2265, 0.2565] | -0.0002 [-0.0062, +0.0051] | 0.302 | 0.366 | 0.0288 | 21.45 | 3,770,154 | 5 | 2.04 | statistically tied, more complex |
| Hybrid_AddAnomaly_500k | 0.2413 [0.2268, 0.2561] | -0.0004 [-0.0058, +0.0045] | 0.285 | 0.374 | 0.0262 | 20.99 | 3,833,625 | 5 | 2.04 | statistically tied, more complex |
| Hybrid_RulesSupervised_500k | 0.2413 [0.2267, 0.2561] | -0.0004 [-0.0057, +0.0043] | 0.284 | 0.374 | 0.0263 | 11.69 | 3,837,083 | 4 | 2.04 | statistically tied, more complex |
| XGBoost_500k | 0.2351 [0.2212, 0.2488] | -0.0065 [-0.0116, -0.0021] | 0.280 | 0.373 | 0.0260 | 10.63 | 3,905,696 | 1 | 0.04 | significantly lower PR-AUC |
| LightGBM_500k | 0.2325 [0.2182, 0.2466] | -0.0092 [-0.0142, -0.0046] | 0.328 | 0.353 | 0.0331 | 10.46 | 3,746,042 | 1 | 0.11 | significantly lower PR-AUC |
| FullHybrid_500k | 0.2318 [0.2172, 0.2459] | -0.0099 [-0.0156, -0.0051] | 0.312 | 0.357 | 0.0310 | 20.84 | 3,780,918 | 5 | 2.04 | significantly lower PR-AUC |
| RulesOnly_500k | 0.0963 [0.0909, 0.1029] | -0.1453 [-0.1576, -0.1332] | 0.226 | 0.151 | 0.0702 | 1.82 | 4,352,004 | 1 | 0.00 | significantly lower PR-AUC |

Recall/precision/FPR are at each model's validation max-F1 threshold (STEP_UP). Tiered cost applies validation-selected monitor/step-up/hold thresholds with the assumed costs in config.yaml (synthetic placeholders). Day-cluster bootstrap, 1000 resamples, paired.

## Recall by fraud type at a matched 5% review budget

Comparing at the same review volume removes threshold-placement effects.

| Fraud type | FullHybrid_500k | Hybrid_AddAnomaly_500k | Hybrid_RulesSupervised_500k | Hybrid_WithoutSequence_500k | IsolationForest_500k | LightGBM_200k | LightGBM_500k | RulesOnly_500k | XGBoost_500k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ACCOUNT_TAKEOVER | 0.697 | 0.688 | 0.690 | 0.689 | 0.219 | 0.682 | 0.700 | 0.427 | 0.699 |
| AGENT_FRAUD | 0.016 | 0.023 | 0.023 | 0.023 | 0.058 | 0.039 | 0.032 | 0.029 | 0.019 |
| CASH_OUT_ABUSE | 0.262 | 0.267 | 0.267 | 0.264 | 0.154 | 0.275 | 0.270 | 0.077 | 0.259 |
| DEVICE_FRAUD | 0.296 | 0.301 | 0.305 | 0.287 | 0.060 | 0.315 | 0.316 | 0.296 | 0.333 |
| MULE_ACTIVITY | 0.525 | 0.494 | 0.500 | 0.516 | 0.036 | 0.500 | 0.498 | 0.020 | 0.486 |
| SIM_SWAP_PATTERN | 0.588 | 0.585 | 0.588 | 0.588 | 0.233 | 0.607 | 0.619 | 0.340 | 0.601 |
| SOCIAL_ENGINEERING | 0.454 | 0.449 | 0.447 | 0.444 | 0.111 | 0.435 | 0.442 | 0.070 | 0.442 |
| SUBTLE_FRAUD | 0.017 | 0.018 | 0.018 | 0.017 | 0.047 | 0.012 | 0.017 | 0.036 | 0.014 |
| VELOCITY_FRAUD | 0.021 | 0.021 | 0.020 | 0.020 | 0.039 | 0.011 | 0.016 | 0.078 | 0.020 |

## Answers

- **Which model won on the common test set?** LightGBM_200k: statistically tied for best PR-AUC and the simplest of the tied group.
- **Best PR-AUC:** LightGBM_200k (0.2417). Tied within bootstrap CI: LightGBM_200k, Hybrid_WithoutSequence_500k, Hybrid_AddAnomaly_500k, Hybrid_RulesSupervised_500k.
- **Best fraud recall (at validation thresholds):** LightGBM_500k (0.3280). It flags more traffic (FPR 0.0331); at a matched 5% review budget recall is within ~0.006 across the supervised/hybrid models.
- **Fewest false negatives:** LightGBM_500k (3510).
- **Lowest false-positive rate:** XGBoost_500k (0.0260).
- **Lowest inference latency (P95 batch 1):** RulesOnly_500k (1.82 ms); fastest ML model: LightGBM_500k (10.46 ms).
- **Lowest estimated business cost (validation tiers):** LightGBM_500k (3,746,042). Under the assumed costs (1000 per missed fraud vs ~7 per step-up), the validation cost-optimal policy steps up 33–100% of all transactions, and rules-only "wins" by flagging everything; cost alone cannot select a model without a review-capacity constraint.
- **Is the hybrid improvement worth the complexity?** No. The best hybrid (Hybrid_WithoutSequence_500k) differs from LightGBM_200k by -0.0002 PR-AUC (CI includes zero), adds 3–4 components and roughly doubles P95 latency. Its mule/social-engineering recall advantage at matched budget is ≤0.02 absolute.
- **Should LightGBM be the primary production/demo model?** Yes, LightGBM_200k.
- **Supporting layers that remain active:** rule engine (independent fallback and hard blocks), Isolation Forest behavioral anomaly score (supporting evidence; abstains on cold start), device risk, recipient risk and graph/mule risk scores (decision inputs and explanations).

## Limitations

- Three saved bundles use isotonic calibration (XGBoost_500k, FullHybrid_500k, LightGBM_500k), which collapses scores into few tied levels (LightGBM_500k: 240 distinct values, max 0.389, below its validation hold threshold, so it never emits HOLD). Their PR-AUC is evaluated as deployed, with ties; part of their gap may be calibration resolution rather than a weaker ranker. Matched-budget recall uses stable index order within ties.
- One training seed per model, one synthetic simulator; CIs capture test sampling only, not training variance.
- Latency is in-process on a prepared feature frame (round-robin interleaved timing); API latency is in final_integration_benchmark.md.

## Recommendation

Primary real-time fraud model: **LightGBM (trained on 200k), `models/200k/LightGBM.joblib`**, recommended STEP_UP threshold **0.2052** (validation max F1).

Supporting layers: Rule Engine, Isolation Forest, Recipient Risk, Device Risk, Graph Risk.

Reason: LightGBM_200k achieved PR-AUC statistically indistinguishable from the best hybrids on the common test set while being a single component, ~1.9x faster at P95 and simpler to deploy and explain (exact TreeSHAP).
