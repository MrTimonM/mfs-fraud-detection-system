# Experiment 100k

Synthetic data only; not upay production performance.

Rows: 100,000; fraud prevalence: 4.969%. Validation and leakage audit passed.

Validation-selected supervised model: **LightGBM**. Validation-selected system: **Hybrid_WithoutGraph**. Test PR-AUC: 0.2226; precision 0.0579; recall 0.9235; F1 0.1090. Threshold 0.017443; calibration identity.

Total wall time: 97.7s. Prepared-frame batch-1 P95 latency: 32.808 ms.

Largest false-positive channel count: APP (4287). Largest missed fraud category: SUBTLE_FRAUD (16). These are descriptive counts, not causal explanations.

Metrics: `results/100k/metrics.csv`; ablations: `ablation_study.csv` and `ablation_by_fraud_type.csv`; calibration: `calibration_metrics.csv`; SHAP: `shap_summary.png`; exact failures: `failures.json`.

Train 70%; five validation blocks of 3% each for tune, fusion, calibration fit, calibration/system selection and threshold; test final 15%. No test tuning. Dates, overlap and class support are saved. Business costs are assumptions. Bootstrap resamples UTC days, conditional on one seed/model fit. Larger datasets include more users at the same history depth, so this is population scaling rather than a fixed-population learning curve.
