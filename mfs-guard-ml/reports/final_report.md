# MFS Guard synthetic fraud research report

## 1. Executive Summary


This project uses synthetic data only. Results do not represent real upay production performance.

- Recommended size: **100k**, the smallest scale within 0.01 validation PR-AUC of the best validation-selected system.
- Recommended prototype system: **Hybrid_WithoutGraph**. Held-out PR-AUC **0.2226**, recall **0.9235**, precision **0.0579**, F1 **0.1090**.
- Synthetic binary decision threshold: **0.017443**, selected on a disjoint validation block using assumed losses. Probability and action are separate outputs.
- The cost-optimal threshold reviews **75.1%** of test transactions. An alternative selected solely for validation F1 uses threshold **0.105000**, giving test precision **0.2356**, recall **0.4518**, F1 **0.3097**, and review rate **9.0%**. A validation-selected 5% review-budget comparison is also saved in `operating_point_comparison.csv`; its test review rate can differ.
- Prepared-frame batch-1 P95 inference latency: **32.808 ms**. This excludes online historical-state retrieval; production latency has not been established.
- Best supervised model by tune PR-AUC at 500k: **XGBoost**.
- Full hybrid minus selected standalone supervised test PR-AUC at 500k: **-0.0020**. Hybrid did not improve this metric.
- Isolation Forest incremental PR-AUC (full minus no-anomaly fusion): **+0.0043**, paired test-day bootstrap 95% CI **[+0.0015, +0.0070]**. Positive interval supports measurable benefit on this test.
- Graph-family retraining: selected supervised minus graph-removed PR-AUC **+0.0018**, CI **[-0.0009, +0.0047]**; mule recall delta **-0.0204**.
- Sequence-family retraining: selected supervised minus sequence-removed PR-AUC **+0.0055**, CI **[+0.0018, +0.0100]**; account-takeover recall delta **-0.0569**.
- Largest minus smallest scale test PR-AUC change for validation-selected systems: **+0.0166**. This is a descriptive population-scaling comparison, not a paired fixed-population learning curve or proof of statistical significance.
- Largest scale worthwhile under the predeclared 0.01 validation-AP trade-off rule: **no; a smaller scale met the rule**. Multiple-seed stability is not measured.

Largest false-positive channel count: APP (4287). Largest missed fraud category: SUBTLE_FRAUD (16). These are descriptive counts, not causal explanations.


Use `100k/Hybrid_WithoutGraph.joblib` with its saved preprocessing, calibration and threshold for a local synthetic demo. High-risk actions route to human review. Do not use this model to make live customer decisions without real-data validation.


## 2. Problem Definition

Estimate pre-transaction synthetic fraud risk, explain signals and support analyst review.

## 3. Dataset Generation Methodology

Five independently generated scales with seed 42; persistent wallets and user profiles, ~80 transactions per user across 120 days. Three-event latent episodes, noisy outcomes and overlapping legitimate stress; amounts gradually inflate. No deterministic fraud labels from observed thresholds.

## 4. Dataset Schema

All original mandatory columns plus historical context are retained in raw CSV and Parquet. Dataset cards and leakage inventories describe each column. Non-agent/merchant IDs use -1. Optional PageRank and some optional fine-grained telemetry are omitted; no fabricated substitutes.

## 5. Fraud Scenarios

Nine fraud categories: account takeover, mule activity, velocity, subtle, device, social engineering, agent, cash-out and SIM swap. Prevalence is validated at 4–6%. Difficult scenarios intentionally overlap legitimate transactions.

## 6. Leakage Prevention

Target labels/types, identifiers, rule aggregates, post-decision balances and simulator profile parameters are excluded. Prefix-invariance and target-mutation tests verify history causality. No future labels create reputation. Blacklists are fixed independently of outcomes.

## 7. Feature Engineering

Rolling windows, prior means/medians/std, user-relative amounts, location distances and known-recipient history are emitted before updating state. Incoming transfers update recipient liquidity. Preprocessors are fitted only on training rows.

## 8. Rule Engine

Nineteen configurable weighted rules in config.yaml. Rule-only calibrated baseline is separately evaluated. Aggregated rule score is excluded from standalone ML.

## 9. Supervised Models

Logistic Regression, Random Forest, HistGradientBoosting, XGBoost and LightGBM at every scale. Imbalance uses class weights or scale_pos_weight. Seeded two-trial randomized holdout search for RF/XGB/LGBM. Secondary subtype task uses LGBM/XGB/RF with fixed CPU budgets.

## 10. Behavioral Anomaly Detection

Isolation Forest fitted on legitimate training observations with at least five prior events. Cold starts explicitly abstain; global training prevalence is the fallback and fusion receives an activity flag. Anomaly raw rank is not a probability; held-out calibration maps it. Zero-observation baseline fields use simulator onboarding-profile priors rather than measured history. Replace unavailable priors with training-cohort defaults in real-data work; the fixed simulated population does not test continuous new-account arrival.

## 11. Graph Intelligence

Incremental degrees, recipient concentration, shared recipient/device state, reciprocal ratios and union-find components. No all-graph PageRank recomputation. Recipient and graph information are correlated, so removal of one named family does not remove every proxy.

## 12. Sequence Intelligence

Past credential/device/channel events within 24h combine with the present proposed transfer. Interpretable sequence indicators and a score replace costly recurrent networks.

## 13. Risk Fusion

Logistic regression fits on the fusion validation block using raw pretrained supervised probability, anomaly rank, rules and context scores. Base calibration fitted later is deliberately not fed backward into fusion training. Fusion outputs are calibrated downstream.

## 14. Evaluation Methodology

Earliest 70% train; next 15% split into five 3% blocks: tuning, fusion fitting, calibration fitting, calibration/system selection, cost threshold. Last 15% test. All test outcomes are untouched until evaluation. Returning users overlap splits by design; saved overlap and cold-start coverage quantify this.

## 15. Results by Dataset Size

- 100k: Hybrid_WithoutGraph; PR-AUC 0.2226, recall 0.9235, precision 0.0579, F1 0.1090.
- 200k: LightGBM; PR-AUC 0.2401, recall 0.7622, precision 0.0931, F1 0.1659.
- 300k: LightGBM; PR-AUC 0.2098, recall 0.8941, precision 0.0621, F1 0.1161.
- 400k: LightGBM; PR-AUC 0.2350, recall 0.8981, precision 0.0664, F1 0.1236.
- 500k: Hybrid_AddAnomaly; PR-AUC 0.2392, recall 0.8925, precision 0.0650, F1 0.1212.

## 16. Model Comparison

Full measured table: results/final/all_model_results.csv. Models and scale recommendations are selected on validation PR-AUC, not test accuracy. Failures are recorded per dataset in failures.json.

## 17. Dataset Size Comparison

See dataset_size_comparison.csv and scale_*.png. Population size grows with rows, holding approximate per-user depth fixed. Single-seed, different test populations cannot establish that any scale significantly outperforms another.

## 18. Ablation Study

At every scale: rules, anomaly, supervised, progressive fusion and leave-one-component-out fusion. Graph/sequence families are also removed and the selected supervised model is retrained with fixed hyperparameters. Same test rows and disjoint calibration/threshold blocks are used. Paired 95% day-cluster bootstrap intervals are conditional on fitted models; do not capture training-seed variance and are not adjusted for multiple comparisons. See ablation_all_sizes.csv and per-type recall files.

| Variant | PR-AUC | Recall | Precision | F1 | FPR | Cost / transaction | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.2337 | 0.8330 | 0.0764 | 0.1400 | 0.5145 | 17.91 | 12.13 |
| RulesOnly | 0.0924 | 1.0000 | 0.0486 | 0.0927 | 1.0000 | 19.03 | 2.06 |
| IsolationForest | 0.0707 | 0.9997 | 0.0487 | 0.0929 | 0.9976 | 19.00 | 9.35 |
| Hybrid_RulesSupervised | 0.2388 | 0.8829 | 0.0670 | 0.1246 | 0.6284 | 17.65 | 13.05 |
| Hybrid_AddAnomaly | 0.2392 | 0.8925 | 0.0650 | 0.1212 | 0.6559 | 17.71 | 61.71 |
| Hybrid_AddDeviceRecipient | 0.2324 | 0.8865 | 0.0663 | 0.1233 | 0.6386 | 17.67 | 23.17 |
| FullHybrid | 0.2318 | 0.8933 | 0.0655 | 0.1220 | 0.6515 | 17.58 | 20.24 |
| Hybrid_WithoutAnomaly | 0.2275 | 0.8536 | 0.0721 | 0.1330 | 0.5613 | 17.80 | 12.05 |
| Hybrid_WithoutGraph | 0.2283 | 0.8994 | 0.0644 | 0.1202 | 0.6678 | 17.60 | 21.25 |
| Hybrid_WithoutSequence | 0.2421 | 0.8722 | 0.0689 | 0.1277 | 0.6024 | 17.67 | 20.25 |
| WithoutGraphFeatures | 0.2319 | 0.8928 | 0.0647 | 0.1207 | 0.6593 | 17.76 | 11.24 |
| WithoutSequenceFeatures | 0.2282 | 0.9134 | 0.0620 | 0.1162 | 0.7060 | 17.65 | 10.82 |


## 19. Calibration

Identity, sigmoid and isotonic are fitted on a separate block and selected by Brier score on the select block, constrained to no more than 0.015 PR-AUC loss. Calibration curves use held-out test rows; calibration_metrics.csv records validation family selection.

## 20. Threshold Optimization

Grid and validation-score quantiles select the minimum simulated cost on the threshold block. Multi-action bands minimize assumed conditional expected loss using calibrated probability; they are assumption-derived, not empirical production thresholds.

## 21. Business Cost Analysis

Assumed missed-fraud loss 1000 units, false-positive investigation/friction 20 units. Binary cost and per-transaction cost are saved. Five actions use configured residual loss and legitimate costs. No real upay economics are claimed.

## 22. Latency and Scalability

Single-process CPU timings include train-only preprocessing, model inference and calibration on precomputed historical frames, batches 1/10/100/1000. Repeated mean/P50/P95/P99 and throughput saved. State-service lookups and HTTP/network latency are not included. Sampled process RSS is not per-model exclusive memory. Fusion training time excludes base model training, documented in artifacts.

## 23. Error Analysis

Largest false-positive channel count: APP (4287). Largest missed fraud category: SUBTLE_FRAUD (16). These are descriptive counts, not causal explanations.
 Full counts by fraud type, transaction type, channel, amount, device, recipient, history and hour are saved.

## 24. Adversarial Testing

Observed difficult fraud slices include near-threshold amounts, splitting, slow activity, known devices and normal locations. These are not an adaptive attacker simulation. A separate synthetic unseen-pattern experiment withholds SIM_SWAP_PATTERN from supervised training and every validation label fit, then reports recall; shared features do not imply zero-day generalization.

## 25. Synthetic Segment Analysis

Operational segments compare recall/FPR across customer segment, account age, amount band, channel, region and rooted-device state. These analyses cannot establish real demographic fairness.

## 26. Limitations

Synthetic latent scenarios, small scenario family, stationarity assumptions, one generation/training seed, correlated episodes and limited calibration support. No fixed-population sample-size learning curve; no multi-seed confidence intervals. Historical feature service and production API are not implemented. CPU budgets constrain tuning. SHAP explains uncalibrated tree margins, not causal effects or the full hybrid decision.

## 27. Recommended Final Architecture

Use measured validation-selected Hybrid_WithoutGraph at 100k, saved calibration and threshold, structured explanations and analyst review. Keep other layers only where ablation evidence supports them.

## 28. Future Real-Data Validation Plan

Collect consented de-identified event histories and delayed investigation outcomes; freeze data availability times, evaluate delayed labels and user-disjoint cold starts, compare on a fixed holdout, repeat seeds, estimate actual costs, shadow deploy, monitor drift and analyst overrides before live actions.

## 29. Conclusion

Artifacts reproduce a synthetic research benchmark. Measured performance and uncertainty determine the demo recommendation; no production effectiveness claim is made.