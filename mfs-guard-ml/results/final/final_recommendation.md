# Measured synthetic benchmark recommendation

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
