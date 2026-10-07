# Experiment 400k

Synthetic data only; not upay production performance.

Rows: 400,000; fraud prevalence: 4.933%. Validation and leakage audit passed.

Validation-selected supervised model: **LightGBM**. Validation-selected system: **LightGBM**. Test PR-AUC: 0.2350; precision 0.0664; recall 0.8981; F1 0.1236. Threshold 0.019219; calibration sigmoid.

Total wall time: 429.4s. Prepared-frame batch-1 P95 latency: 10.587 ms.

Largest false-positive channel count: AGENT (13642). Largest missed fraud category: SUBTLE_FRAUD (101). These are descriptive counts, not causal explanations.

Metrics: `results/400k/metrics.csv`; ablations: `ablation_study.csv` and `ablation_by_fraud_type.csv`; calibration: `calibration_metrics.csv`; SHAP: `shap_summary.png`; exact failures: `failures.json`.

Train 70%; five validation blocks of 3% each for tune, fusion, calibration fit, calibration/system selection and threshold; test final 15%. No test tuning. Dates, overlap and class support are saved. Business costs are assumptions. Bootstrap resamples UTC days, conditional on one seed/model fit. Larger datasets include more users at the same history depth, so this is population scaling rather than a fixed-population learning curve.
