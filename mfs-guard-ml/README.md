# MFS Guard ML

A CPU-reproducible synthetic fraud benchmark at 100k, 200k, 300k, 400k and 500k transactions. It generates actual CSV/Parquet datasets, trains binary and subtype models, calibrates risks, compares rules and learned fusion, and performs component and feature-family ablations.

**This project uses synthetic data only. Results do not represent real upay production performance.**

## Run

Windows PowerShell, from this directory:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -v
.venv/Scripts/python scripts/run_all_experiments.py --generate --train --evaluate --compare
```

On Linux/macOS use `.venv/bin/python` instead. Windows CMD accepts `.venv\Scripts\python.exe`.

```powershell
.venv/Scripts/python scripts/generate_dataset.py --rows 100000 --seed 42
.venv/Scripts/python scripts/generate_all.py
.venv/Scripts/python scripts/train_experiment.py --size 100k
.venv/Scripts/python scripts/run_all_experiments.py --resume
.venv/Scripts/python scripts/compare_all.py
.venv/Scripts/python scripts/supplement_reports.py
.venv/Scripts/python scripts/verify_artifacts.py
```

`--resume` trusts completed experiment metadata. Remove that scale's completion metadata to rerun a changed experiment; remove its raw CSV/Parquet to regenerate. Existing datasets are not silently regenerated with a different seed. Generation metadata records the original seed. Keep code/config/data together for reproducibility.

## Design and evaluation

Persistent customers and wallets generate three-event episodes across 120 days. Noisy latent attacks overlap legitimate stressful events, producing approximately 5% fraud. About 80 transactions per customer support historical baselines. Nine fraud categories cover takeover, mule, velocity, subtle, device, social engineering, agent, cash-out and SIM swap. Every original mandatory column in the supplied prompt is preserved; safe optional historical features are added. Customer profile parameters guide simulation but are excluded from model features.

Transaction context feeds rules, supervised classification, a behavioral Isolation Forest, recipient/device history, graph features and sequence indicators. A learned logistic fusion produces a calibrated probability. A separately optimized threshold and configurable simulated action costs determine review actions. Cold starts explicitly abstain from behavioral scoring.

Chronological blocks: training 70%, validation 15%, test 15%. Validation is divided into five disjoint 3% blocks for tuning, fusion training, calibration fitting, calibration/system selection and cost threshold selection. Test labels never enter fitting. Preprocessors use training rows only; graph and rolling history use strictly preceding events. No label-derived risk features. Class weights handle imbalance. Randomized two-trial holdout searches bound RF/XGBoost/LightGBM tuning costs; logistic regression and histogram boosting use fixed budgets.

Models: Logistic Regression, Random Forest, HistGradientBoosting, XGBoost, LightGBM, Isolation Forest, rule baseline and seven fusion variants. Multiclass LightGBM/XGBoost/Random Forest form a secondary subtype experiment. Primary selection metric is average precision (reported as PR-AUC); precision, recall, F1, ROC-AUC, MCC, FPR/FNR, Brier score, review volume and simulated cost are also reported.

Every scale has progressive fusion ablations, leave-one-layer-out fusion ablations, and supervised retraining without graph or sequence features. Paired test-day cluster bootstrap intervals compare average precision. Subtype recall measures mule and takeover effects. Removing a fusion input does not remove that signal from the supervised base; the separate retraining comparisons and documented correlated proxies are essential to interpreting the results.

## Artifacts

- `data/raw/`: original CSV and Parquet datasets.
- `data/processed/`: eligible feature frames and split indexes.
- `data/metadata/`: profiles, configurations, seeds and dataset statistics.
- `models/<size>/`: serialized bundles, preprocessors, thresholds, metadata and subtype models.
- `results/<size>/`: metrics, test predictions, ablations, bootstrap intervals, validation/leakage audits, calibration, thresholds, errors, SHAP, drift, robustness, latency and plots.
- `results/final/`: cross-scale results, recommendation and mandatory artifact verification.
- `reports/`: dataset cards, experiment reports and final report.
- `logs/experiments.log`: progress and failed experiments.

Trained models, aggregate results, plots and reports are included in Git. Generated transaction datasets, customer profile tables, transaction-level prediction exports, local environments and download caches are excluded. Generate the datasets locally with the commands above when reproducing the experiments. Dependencies are pinned in `requirements.txt` to the tested environment.

`supplement_reports.py` adds individual ROC plots, subtype confusion plots, ablation interval plots, component latency timings, 100-repeat selected-system batch-1 latency and dependency-inclusive training costs. Run `compare_all.py` afterward to refresh cross-scale reports. `verify_artifacts.py` reloads every binary bundle and checks its probabilities against saved held-out predictions.

## Inference

```python
import joblib
import pandas as pd

bundle = joblib.load('models/100k/LightGBM.joblib')
frame = pd.read_parquet('data/raw/mfs_100k.parquet').tail(10)
probabilities = bundle.predict(frame)
review = probabilities >= bundle.threshold
```

Only load trusted joblib files. Inputs must contain pre-decision historical features; this project does not implement an online feature store. Saved bundles include preprocessing and calibration. Multi-action costs and rules are in `config.yaml`; units are arbitrary simulation units, not company figures. Analyst review remains available for high-risk outcomes.

## Interpretation and limitations

Read `reports/final_report.md` and `results/final/final_recommendation.md` for measured findings. The smallest scale within 0.01 validation average precision of the best validation-selected system is the default trade-off recommendation. The largest dataset is not automatically the winner.

Dataset size grows the customer population while preserving history depth. This is not a nested same-population learning curve. Bootstrap intervals capture test-day uncertainty conditional on one fitted model/seed, not full training variance. Risk-family proxies are correlated, rare-category calibration can be unstable, and simulated costs may favor high recall with high review rates. Optional PageRank, some fine-grained telemetry and neural sequence models are omitted for compute feasibility. No real demographic fairness or production efficacy claim is justified.

Zero-observation customer baselines start from the simulator's onboarding profile prior; they are not measured transaction history. The anomaly layer abstains until five previous observations exist. A real-data implementation must replace unavailable profile priors with training-cohort defaults and validate cold starts separately. This benchmark has a fixed customer population rather than continuous new-account arrivals, and recipient concentration rather than a complete adversarial mule-chain simulator.

Latency includes preprocessing and all model dependencies on a prepared historical frame; it excludes live state retrieval, network and API overhead. Sampled RAM is process RSS, not exclusive estimator memory. Fusion fit times exclude base training; base costs are reported separately. Real deployment requires a historical feature service, delayed-label validation, actual cost estimation, repeated seeds, fixed-population holdouts, monitored shadow evaluation and human review.
