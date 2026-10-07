"""Additional plots and measured inference component timings from saved artifacts."""
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay
from .config import ROOT, config, dump
from .plots import save
from .risk_fusion import meta_features
from .rule_engine import apply_rules
from .latency_benchmark import benchmark


def supplements():
    cfg = config()
    for size in cfg['sizes']:
        tag = f'{size//1000}k'
        out = ROOT/f'results/{tag}'
        if not (out/'experiment_metadata.json').exists():
            continue
        metadata = json.loads((out/'experiment_metadata.json').read_text())
        results = pd.read_csv(out/'metrics.csv')
        costs = results.set_index('model').training_time_seconds.to_dict()
        stack_costs = []
        for model in results.model:
            model_meta = json.loads((ROOT/f'models/{tag}/{model}_metadata.json').read_text())
            total = costs[model]
            if model_meta['kind'] == 'fusion':
                total += costs[metadata['best_supervised']]
                if 2 in model_meta['meta_columns']:
                    total += costs['IsolationForest']
            stack_costs.append(total)
        results['stack_training_time_seconds'] = stack_costs
        results.to_csv(out/'metrics.csv', index=False)
        predictions = pd.read_parquet(out/'test_predictions.parquet')
        for name, group in predictions.groupby('model_name'):
            fpr, tpr, _ = roc_curve(group.fraud_label, group.predicted_probability)
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], '--', color='gray')
            plt.title(name)
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            save(out, f'roc_{name}.png')
        for name in ['LightGBM', 'XGBoost', 'RandomForest']:
            path = out/f'multiclass_confusion_{name}.json'
            if path.exists():
                data = json.loads(path.read_text())
                _, ax = plt.subplots(figsize=(12, 10))
                ConfusionMatrixDisplay(np.asarray(data['matrix']), display_labels=data['classes']).plot(
                    ax=ax, xticks_rotation=90, colorbar=False, values_format='d')
                plt.title(f'{name}: held-out subtype confusion matrix')
                save(out, f'multiclass_confusion_{name}.png')
        ablation = pd.read_csv(out/'ablation_study.csv')
        plt.figure(figsize=(11, 7))
        positions = np.arange(len(ablation))
        delta = ablation.pr_auc_delta.to_numpy()
        low, high = ablation.delta_ci_low.to_numpy(), ablation.delta_ci_high.to_numpy()
        # Plot interval endpoints directly; bootstrap interval need not contain estimate.
        plt.hlines(positions, low, high, color='steelblue')
        plt.scatter(delta, positions, color='navy', s=18)
        plt.yticks(positions, ablation.model)
        plt.axvline(0, linestyle='--', color='gray')
        plt.xlabel('Reference minus variant PR-AUC; paired test-day bootstrap 95% CI')
        plt.title(f'{tag}: reference is FullHybrid; retrained feature removals use selected supervised')
        save(out, 'ablation_deltas.png')
        raw_frame = pd.read_parquet(ROOT/f'data/raw/mfs_{tag}.parquet')
        explanation_path = out/'local_explanations.json'
        if explanation_path.exists():
            explanations = json.loads(explanation_path.read_text())
            contexts = raw_frame[raw_frame.transaction_id.isin([r['transaction_id'] for r in explanations])].set_index('transaction_id')
            context_columns = ['amount', 'balance_before', 'transaction_type', 'channel', 'amount_vs_user_mean',
                'tx_count_5m', 'first_time_recipient', 'device_is_new', 'failed_pin_attempts', 'otp_resend_count',
                'device_risk_score', 'recipient_risk_score', 'sequence_risk_score', 'mule_network_score', 'behavioral_history_count']
            for record in explanations:
                record['feature_value_space'] = 'standardized numeric features or one-hot categories; see raw_context for original units'
                record['raw_context'] = contexts.loc[record['transaction_id'], context_columns].to_dict()
            dump(explanation_path, explanations)
        frame = raw_frame.tail(1)
        del raw_frame
        selected = joblib.load(ROOT/f'models/{tag}/{metadata["selected_system"]}.joblib')
        # A more stable batch-1 tail estimate for the validation-selected system.
        selected_latency = benchmark(selected, frame, repeats=100).iloc[[0]]
        selected_latency['model'] = metadata['selected_system']
        selected_latency.to_csv(out/'selected_system_latency_100_repeats.csv', index=False)
        full = joblib.load(ROOT/f'models/{tag}/FullHybrid.joblib')
        supervised = full.supervised
        prepared = supervised.preprocessor.transform(frame[supervised.columns]).astype('float32')
        raw_supervised = supervised.predict(frame)
        raw_anomaly = full.anomaly.predict(frame)
        meta = meta_features(frame, raw_supervised, raw_anomaly)
        raw_fusion = full.model.predict_proba(meta[:, full.meta_columns])[:, 1]
        stages = {
            'prepared_frame_encoding': lambda: supervised.preprocessor.transform(frame[supervised.columns]).astype('float32'),
            'rules': lambda: apply_rules(frame.copy(), cfg),
            'supervised_estimator': lambda: supervised.model.predict_proba(prepared),
            'anomaly_with_its_preprocessing': lambda: full.anomaly.predict(frame),
            'fusion_estimator_with_meta_scaling': lambda: full.model.predict_proba(meta[:, full.meta_columns]),
            'final_calibration': lambda: full.calibration.predict(raw_fusion),
            'total_prepared_frame_hybrid': lambda: full.predict(frame),
        }
        records = []
        for stage, operation in stages.items():
            operation()
            timings = []
            for _ in range(50):
                start = time.perf_counter()
                operation()
                timings.append((time.perf_counter()-start)*1000)
            records.append({'stage': stage, 'mean_ms': np.mean(timings), 'p50_ms': np.percentile(timings, 50),
                'p95_ms': np.percentile(timings, 95), 'p99_ms': np.percentile(timings, 99), 'repeats': 50,
                'scope': 'single-process batch-1, historical frame available; does not measure state retrieval'})
        pd.DataFrame(records).to_csv(out/'latency_components.csv', index=False)
        print(f'Supplemental figures and inference component timings: {tag}', flush=True)
