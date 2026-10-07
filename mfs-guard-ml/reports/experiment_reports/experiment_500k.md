# Experiment 500k

Synthetic data only; not upay production performance.

Rows: 500,000; fraud prevalence: 4.901%. Validation and leakage audit passed.

Validation-selected supervised model: **XGBoost**. Validation-selected system: **Hybrid_AddAnomaly**. Test PR-AUC: 0.2392; precision 0.0650; recall 0.8925; F1 0.1212. Threshold 0.019189; calibration sigmoid.

Total wall time: 521.5s. Prepared-frame batch-1 P95 latency: 61.707 ms.

Largest false-positive channel count: AGENT (17307). Largest missed fraud category: SUBTLE_FRAUD (120). These are descriptive counts, not causal explanations.

Metrics: `results/500k/metrics.csv`; ablations: `ablation_study.csv` and `ablation_by_fraud_type.csv`; calibration: `calibration_metrics.csv`; SHAP: `shap_summary.png`; exact failures: `failures.json`.

Train 70%; five validation blocks of 3% each for tune, fusion, calibration fit, calibration/system selection and threshold; test final 15%. No test tuning. Dates, overlap and class support are saved. Business costs are assumptions. Bootstrap resamples UTC days, conditional on one seed/model fit. Larger datasets include more users at the same history depth, so this is population scaling rather than a fixed-population learning curve.
