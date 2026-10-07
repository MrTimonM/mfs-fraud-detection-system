"""Complete chronological synthetic benchmark, persistence and controlled ablation."""
import gc
import importlib.metadata
import json
import logging
import platform
import time
import traceback
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits
from .config import config, paths, ROOT, dump
from .data_generator import generate
from .data_validation import validate
from .leakage_audit import audit, feature_columns, GRAPH, SEQUENCE
from .split_data import split
from .preprocessing import preprocessor
from .train_supervised import estimators, fit_model
from .train_anomaly import BehavioralAnomaly
from .risk_fusion import RiskBundle, meta_features, fit_fusion, FUSION_COLUMNS
from .calibration import calibrate
from .evaluate import metrics, paired_bootstrap
from .threshold_optimizer import optimize, actions
from .resource_benchmark import ResourceMonitor
from .latency_benchmark import benchmark
from .error_analysis import analyze
from .drift_analysis import drift
from .adversarial_tests import adversarial
from .explain import explain
from .plots import plots, save


def logging_setup():
    warnings.filterwarnings('ignore', message='X does not have valid feature names.*')
    (ROOT/'logs').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(ROOT/'logs/experiments.log', encoding='utf-8')], force=True)


def dataset_card(d, size, validation):
    info = {'rows': len(d), 'users': d.user_id.nunique(), 'receivers': d.receiver_id.nunique(),
        'devices': d.device_id.nunique(), 'agents': d.loc[d.agent_id >= 0, 'agent_id'].nunique(),
        'merchants': d.loc[d.merchant_id >= 0, 'merchant_id'].nunique(),
        'date_range': [str(d.timestamp.min()), str(d.timestamp.max())], 'columns': len(d.columns),
        'fraud_prevalence': d.fraud_label.mean(), 'fraud_types': d.fraud_type.value_counts().to_dict(),
        'transaction_types': d.transaction_type.value_counts().to_dict(), 'channels': d.channel.value_counts().to_dict()}
    dump(ROOT/f'data/metadata/dataset_{size//1000}k.json', info)
    body = f'# Synthetic MFS dataset {size//1000}k\n\nThis project uses synthetic data only. Results do not represent real upay production performance.\n\n'
    body += '\n'.join(f'- {k}: {v}' for k, v in info.items())
    body += '\n\nSeed 42 by default; actual seed and full configuration are in generation metadata. Persistent customer profiles, wallet accounting and three-event latent episodes simulate 120 days. Risky episodes overlap legitimate stress; outcomes are noisy. Histories are emitted before state updates; no label-based reputation. Around 80 events per user intentionally trades user breadth for historical depth. Missing agent/merchant IDs use -1. Authentication and balance-inquiry counters are simulated pre-event telemetry. PageRank and deep sequence networks are omitted for CPU feasibility; graph degrees, components and temporal patterns are exact over observed ledger events. Only nine scenario families are simulated; profile stations, episode durations and loss assumptions are not real-world estimates. No independent real-data validation.\n\n## Feature inventory\n\n'
    body += '\n'.join(f'- `{c}`: '+('outcome, identifier or context; see leakage audit' if c in ['fraud_label','fraud_type','timestamp'] else 'observable context or past-only derived signal') for c in d)
    (ROOT/f'reports/dataset_cards/dataset_{size//1000}k.md').write_text(body, encoding='utf-8')


def run(size, seed=42):
    cfg = config()
    cfg['seed'] = seed
    with threadpool_limits(limits=cfg['threads']):
        return _run(size, cfg)


def _run(size, cfg):
    p = paths(size)
    out, modeldir, tag = p['results'], p['models'], p['tag']
    started = time.perf_counter()
    logging.info('START %s', tag)
    source = ROOT/f'data/raw/mfs_{tag}.parquet'
    load_started = time.perf_counter()
    bootstrap = ROOT/f'data/processed/bootstrap_{tag}.pkl'
    generation_metadata = ROOT/f'data/metadata/generation_{tag}.json'
    if (source.exists() or bootstrap.exists()) and generation_metadata.exists():
        recorded_seed = json.loads(generation_metadata.read_text())['seed']
        if recorded_seed != cfg['seed']:
            raise ValueError(f'Existing {tag} dataset seed {recorded_seed} does not match requested {cfg["seed"]}')
    if source.exists():
        d = pd.read_parquet(source)
    elif bootstrap.exists():
        d = pd.read_pickle(bootstrap)  # Only the locally generated bootstrap artifact.
        d.to_parquet(source, index=False)
    else:
        d = generate(size, cfg['seed'])
    dataset_seconds = time.perf_counter()-load_started
    validation = validate(d, size)
    dump(out/'dataset_validation.json', validation)
    dump(out/'leakage_audit.json', audit(d))
    dataset_card(d, size, validation)
    indexes, boundaries = split(d)
    dump(out/'split_boundaries.json', boundaries)
    columns = feature_columns(d)
    frames = {k: d.iloc[ix] for k, ix in indexes.items()}
    y = {k: f.fraud_label.to_numpy() for k, f in frames.items()}
    prep = preprocessor(d, columns).fit(frames['train'][columns])
    x = {k: prep.transform(f[columns]).astype('float32') for k, f in frames.items()}
    names = prep.get_feature_names_out()
    joblib.dump(prep, modeldir/'preprocessor.joblib', compress=3)
    dump(modeldir/'feature_list.json', list(columns))
    d[columns].to_parquet(ROOT/f'data/processed/features_{tag}.parquet', index=False)
    dump(ROOT/f'data/processed/splits_{tag}.json', {k: [int(v[0]), int(v[-1])+1] for k, v in indexes.items()})
    rows, calibration_rows, thresholds_rows, latency_rows, pred_rows = [], [], [], [], []
    bundles, scores, thresholds, reports, failure = {}, {}, {}, {}, []

    def finalize(name, bundle, raw, resource, selection_ap=None):
        bundle.calibration, cal = calibrate(raw['calibrate'], y['calibrate'], raw['select'], y['select'])
        for r in cal:
            calibration_rows.append({'model': name, **r})
        calibrated = {k: bundle.calibration.predict(value) for k, value in raw.items()}
        threshold, curve = optimize(y['threshold'], calibrated['threshold'], cfg)
        bundle.threshold = threshold
        thresholds[name], bundles[name], scores[name] = threshold, bundle, calibrated['test']
        curve['model'] = name
        thresholds_rows.append(curve)
        target = modeldir/f'{name}.joblib'
        joblib.dump(bundle, target, compress=3)
        latency = benchmark(bundle, frames['test'], repeats=12)
        latency['model'] = name
        latency_rows.append(latency)
        result = {'dataset_size': size, 'model': name, 'status': 'SUCCESS',
            **metrics(y['test'], calibrated['test'], threshold, cfg),
            'selection_pr_auc': average_precision_score(y['select'], calibrated['select']),
            'tune_pr_auc': selection_ap if selection_ap is not None else np.nan,
            'training_time_seconds': resource.get('training_time_seconds', 0),
            'peak_memory_mb': resource.get('peak_memory_mb', np.nan),
            'model_file_size_mb': target.stat().st_size/1024**2,
            'p95_inference_latency_ms': float(latency.iloc[0].p95_ms),
            'inference_time_seconds': float(latency.loc[latency.batch_size == min(1000, len(frames['test'])), 'mean_ms'].iloc[0]/1000),
            'calibration': bundle.calibration.method}
        rows.append(result)
        predictions = frames['test'][['transaction_id', 'fraud_label', 'fraud_type']].copy()
        predictions['model_name'], predictions['predicted_probability'] = name, calibrated['test']
        predictions['predicted_label'], predictions['selected_threshold'] = calibrated['test'] >= threshold, threshold
        predictions['final_action'] = actions(calibrated['test'], cfg)
        pred_rows.append(predictions)
        dump(modeldir/f'{name}_metadata.json', {**resource, 'threshold': threshold, 'calibration': bundle.calibration.method,
             'kind': bundle.kind, 'feature_columns': bundle.columns, 'meta_columns': bundle.meta_columns})
        logging.info('%s %s TEST AP %.4f recall %.3f precision %.3f', tag, name, result['pr_auc'], result['recall'], result['precision'])
        return calibrated

    def failed(name, exc):
        logging.exception('FAILED %s %s', tag, name)
        failure.append({'model': name, 'status': 'FAILED', 'error': str(exc), 'traceback': traceback.format_exc()})
        dump(out/'failures.json', failure)

    raw_supervised, tuned = {}, {}
    ratio = (y['train'] == 0).sum()/max(y['train'].sum(), 1)
    for name, estimator in estimators(cfg, ratio).items():
        try:
            model, resource = fit_model(name, estimator, x['train'], y['train'], x['tune'], y['tune'], cfg)
            tuned[name] = resource
            bundle = RiskBundle('supervised', cfg, model=model, preprocessor=prep, columns=columns)
            raw = {k: model.predict_proba(v)[:, 1] for k, v in x.items() if k != 'train'}
            raw_supervised[name] = raw
            finalize(name, bundle, raw, resource, resource['best_validation_pr_auc'])
        except Exception as exc:
            failed(name, exc)
    if not tuned:
        raise RuntimeError('No supervised model succeeded')
    best = max(tuned, key=lambda name: tuned[name]['best_validation_pr_auc'])
    best_bundle = bundles[best]
    logging.info('Selected supervised model using tune block: %s', best)
    finalize('RulesOnly', RiskBundle('rules', cfg),
        {k: f.rule_risk_score.to_numpy()/100 for k, f in frames.items() if k != 'train'}, {})
    with ResourceMonitor() as monitor:
        anomaly = BehavioralAnomaly().fit(frames['train'], cfg)
    ap = {k: anomaly.predict(f) for k, f in frames.items() if k != 'train'}
    finalize('IsolationForest', RiskBundle('anomaly', cfg, anomaly=anomaly), ap,
             {'training_time_seconds': monitor.seconds, 'peak_memory_mb': monitor.peak_mb})
    dump(out/'anomaly_coverage.json', {k: {'active_fraction': float(f.behavioral_model_active.mean()),
         'abstentions': int((f.behavioral_model_active == 0).sum()),
         'fallback': 'Training prevalence; behavioral_model_active exposes abstention to fusion'} for k, f in frames.items()})
    # Fusion must use an upstream probability map not fit on later validation labels.
    # Identity probabilities are used for every fusion base; calibration is only downstream.
    fusion_supervised = RiskBundle('supervised', cfg, model=best_bundle.model, preprocessor=prep, columns=columns)
    meta = {k: meta_features(frames[k], raw_supervised[best][k], ap[k]) for k in ap}
    variants = {'Hybrid_RulesSupervised': [0, 1], 'Hybrid_AddAnomaly': [0, 1, 2, 7],
        'Hybrid_AddDeviceRecipient': [0, 1, 2, 3, 4, 7], 'FullHybrid': list(range(8)),
        'Hybrid_WithoutAnomaly': [0, 1, 3, 4, 5, 6, 7],
        'Hybrid_WithoutGraph': [0, 1, 2, 3, 4, 6, 7], 'Hybrid_WithoutSequence': [0, 1, 2, 3, 4, 5, 7]}
    for name, cols in variants.items():
        with ResourceMonitor() as monitor:
            model = fit_fusion(meta['fusion'][:, cols], y['fusion'])
        bundle = RiskBundle('fusion', cfg, model=model, supervised=fusion_supervised, anomaly=anomaly, meta_columns=cols)
        raw = {k: model.predict_proba(v[:, cols])[:, 1] for k, v in meta.items()}
        finalize(name, bundle, raw, {'training_time_seconds': monitor.seconds, 'peak_memory_mb': monitor.peak_mb,
                 'dependencies': [best, 'IsolationForest'], 'base_training_time_excluded': True})
    # Actual feature-family retraining (separate from removing fusion score inputs).
    for name, removed in [('WithoutGraphFeatures', GRAPH), ('WithoutSequenceFeatures', SEQUENCE)]:
        try:
            keep = np.array([c not in removed for c in names])
            reduced_columns = [c for c in columns if c not in removed]
            reduced_prep = preprocessor(d, reduced_columns).fit(frames['train'][reduced_columns])
            # Transformer column order differs when numeric columns are removed; use its own matrix.
            reduced = {k: reduced_prep.transform(f[reduced_columns]).astype('float32') for k, f in frames.items()}
            with ResourceMonitor() as monitor:
                model = clone(best_bundle.model).fit(reduced['train'], y['train'])
            bundle = RiskBundle('supervised', cfg, model=model, preprocessor=reduced_prep, columns=reduced_columns)
            raw = {k: model.predict_proba(v)[:, 1] for k, v in reduced.items() if k != 'train'}
            finalize(name, bundle, raw, {'training_time_seconds': monitor.seconds, 'peak_memory_mb': monitor.peak_mb,
                     'removed_features': removed, 'same_hyperparameters_as': best})
            del reduced
        except Exception as exc:
            failed(name, exc)
    # Secondary multiclass task, fixed bounded budgets, no test-driven tuning.
    encoder = LabelEncoder().fit(frames['train'].fraud_type)
    mc_y = encoder.transform(frames['train'].fraud_type)
    mc_rows = []
    for name in ['LightGBM', 'XGBoost', 'RandomForest']:
        try:
            with ResourceMonitor() as monitor:
                model = estimators(cfg, ratio, multiclass=True)[name].fit(x['train'], mc_y)
            pred = encoder.inverse_transform(model.predict(x['test']).astype(int))
            report = classification_report(frames['test'].fraud_type, pred, labels=encoder.classes_, output_dict=True, zero_division=0)
            dump(out/f'multiclass_{name}.json', report)
            dump(out/f'multiclass_confusion_{name}.json', {'classes': encoder.classes_,
                 'matrix': confusion_matrix(frames['test'].fraud_type, pred, labels=encoder.classes_)})
            joblib.dump({'model': model, 'preprocessor': prep, 'encoder': encoder, 'columns': columns}, modeldir/f'multiclass_{name}.joblib', compress=3)
            mc_rows.append({'model': name, 'macro_f1': report['macro avg']['f1-score'],
                'weighted_f1': report['weighted avg']['f1-score'], 'training_time_seconds': monitor.seconds})
            logging.info('%s multiclass %s macro F1 %.4f', tag, name, mc_rows[-1]['macro_f1'])
            del model
        except Exception as exc:
            failed('multiclass_'+name, exc)
    pd.DataFrame(mc_rows).to_csv(out/'multiclass_metrics.csv', index=False)
    # Unseen-pattern robustness: remove one category from ALL train/validation fits.
    try:
        unseen_type = 'SIM_SWAP_PATTERN'
        masks = {k: (f.fraud_type != unseen_type).to_numpy() for k, f in frames.items()}
        unknown_prep = preprocessor(d, columns).fit(frames['train'].loc[masks['train'], columns])
        unknown_train = unknown_prep.transform(frames['train'].loc[masks['train'], columns]).astype('float32')
        unknown_model = clone(best_bundle.model).fit(unknown_train, y['train'][masks['train']])
        unknown_raw = {k: unknown_model.predict_proba(unknown_prep.transform(f[columns]).astype('float32'))[:, 1]
                       for k, f in frames.items() if k != 'train'}
        unknown_meta = {k: meta_features(frames[k], unknown_raw[k], ap[k]) for k in unknown_raw}
        unknown_fusion = fit_fusion(unknown_meta['fusion'][masks['fusion']], y['fusion'][masks['fusion']])
        unknown_outputs = {'supervised': unknown_raw,
            'hybrid': {k: unknown_fusion.predict_proba(v)[:, 1] for k, v in unknown_meta.items()},
            'anomaly': ap,
            'rules': {k: f.rule_risk_score.to_numpy()/100 for k, f in frames.items() if k != 'train'}}
        unseen_rows = []
        for name, raw in unknown_outputs.items():
            mapping, _ = calibrate(raw['calibrate'][masks['calibrate']], y['calibrate'][masks['calibrate']],
                raw['select'][masks['select']], y['select'][masks['select']])
            threshold, _ = optimize(y['threshold'][masks['threshold']], mapping.predict(raw['threshold'][masks['threshold']]), cfg)
            prob = mapping.predict(raw['test'])
            unseen = ~masks['test']
            unseen_rows.append({'model': name, 'withheld_type': unseen_type, 'withheld_test_count': int(unseen.sum()),
                'unseen_recall': float(np.mean(prob[unseen] >= threshold)),
                **metrics(y['test'], prob, threshold, cfg)})
        pd.DataFrame(unseen_rows).to_csv(out/'unseen_pattern_results.csv', index=False)
        joblib.dump({'supervised': unknown_model, 'fusion': unknown_fusion, 'preprocessor': unknown_prep,
                     'withheld_type': unseen_type}, modeldir/'unseen_pattern_models.joblib', compress=3)
        del unknown_train, unknown_model, unknown_fusion, unknown_prep
    except Exception as exc:
        failed('unseen_pattern', exc)
    results = pd.DataFrame(rows)
    results.to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(calibration_rows).to_csv(out/'calibration_metrics.csv', index=False)
    curves = pd.concat(thresholds_rows, ignore_index=True)
    curves.to_csv(out/'threshold_metrics.csv', index=False)
    curves.to_csv(out/'business_threshold_analysis.csv', index=False)
    latency = pd.concat(latency_rows, ignore_index=True)
    latency.to_csv(out/'latency_benchmark.csv', index=False)
    dump(out/'latency_benchmark.json', latency.to_dict('records'))
    preds = pd.concat(pred_rows, ignore_index=True)
    preds.to_parquet(out/'test_predictions.parquet', index=False)
    preds.to_csv(out/'test_predictions.csv', index=False)
    full_preds = frames['test'][['transaction_id', 'fraud_label', 'fraud_type', 'rule_risk_score', 'behavioral_model_active']].copy()
    full_preds['anomaly_score'], full_preds['supervised_probability'] = ap['test'], raw_supervised[best]['test']
    full_preds['final_risk_probability'] = scores['FullHybrid']
    full_preds['final_action'] = actions(scores['FullHybrid'], cfg)
    full_preds['behavioral_model_status'] = np.where(full_preds.behavioral_model_active == 1, 'ACTIVE', 'INSUFFICIENT_HISTORY')
    full_preds.to_csv(out/'hybrid_predictions.csv', index=False)
    # Paired cluster confidence intervals over identical held-out transactions.
    ablation_names = ['RulesOnly', 'IsolationForest', best, *variants, 'WithoutGraphFeatures', 'WithoutSequenceFeatures']
    ablation = results[results.model.isin(ablation_names)].copy()
    days = frames['test'].timestamp.dt.floor('D').astype('int64').to_numpy()
    intervals = []
    for name in ablation.model:
        comparator = best if name.startswith('Without') else 'FullHybrid'
        interval = paired_bootstrap(y['test'], scores[comparator], scores[name], days, cfg['bootstrap_repeats'])
        intervals.append({'model': name, 'reference_model': comparator, **interval})
    ablation = ablation.merge(pd.DataFrame(intervals), on='model')
    ablation.to_csv(out/'ablation_study.csv', index=False)
    type_rows = []
    for name in ablation.model:
        for subtype in ['MULE_ACTIVITY', 'ACCOUNT_TAKEOVER', 'SUBTLE_FRAUD', 'SIM_SWAP_PATTERN']:
            mask = frames['test'].fraud_type.to_numpy() == subtype
            type_rows.append({'model': name, 'fraud_type': subtype, 'count': int(mask.sum()),
                 'recall': float(np.mean(scores[name][mask] >= thresholds[name]))})
    pd.DataFrame(type_rows).to_csv(out/'ablation_by_fraud_type.csv', index=False)
    eligible = results[results.model.isin([*tuned, 'RulesOnly', 'IsolationForest', *variants])]
    winner = eligible.sort_values('selection_pr_auc', ascending=False).iloc[0]
    error = analyze(frames['test'], scores[winner.model], thresholds[winner.model], out)
    drift(frames['train'], frames['test'], out)
    adversarial(frames['test'], {name: scores[name] for name in [best, 'RulesOnly', 'IsolationForest', 'FullHybrid']}, thresholds, out)
    importance = []
    for name in ['LightGBM', 'RandomForest', 'XGBoost']:
        if name in bundles:
            importance.extend({'model': name, 'feature': f, 'importance': v}
                              for f, v in zip(names, bundles[name].model.feature_importances_))
    importance = pd.DataFrame(importance)
    importance.to_csv(out/'feature_importance.csv', index=False)
    try:
        # Explain a tree even when logistic regression wins overall.
        tree_name = best if best in ['LightGBM', 'XGBoost', 'RandomForest'] else 'LightGBM'
        chosen = np.random.default_rng(cfg['seed']).choice(len(y['test']), min(256, len(y['test'])), replace=False)
        local = np.argsort(scores[tree_name])[-12:][::-1]
        explain(bundles[tree_name].model, x['test'][chosen], names, out,
                frames['test'].transaction_id.to_numpy()[chosen], x['test'][local], frames['test'].transaction_id.to_numpy()[local])
        dump(out/'explainability_scope.json', {'model': tree_name, 'selection': 'seeded random 256 held-out transactions for global SHAP; highest-risk 12 for local explanations',
                 'limitation': 'Finite sampled model attributions, not causal or calibrated hybrid explanations.'})
    except Exception as exc:
        failed('SHAP', exc)
    plots(d, y['test'], scores, results, out, thresholds)
    import matplotlib.pyplot as plt
    importance[importance.model == 'LightGBM'].nlargest(20, 'importance').set_index('feature').importance.plot.barh(figsize=(8, 7))
    save(out, 'feature_importance.png')
    curves[curves.model == winner.model].plot(x='threshold', y=['precision', 'recall', 'f1'], figsize=(8, 5))
    save(out, 'threshold_curve.png')
    dump(out/'failures.json', failure)
    elapsed = time.perf_counter()-started
    metadata = {'dataset_size': size, 'seed': cfg['seed'], 'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','lightgbm','xgboost','shap']},
        'best_supervised': best, 'selected_system': winner.model, 'selection_metric': 'PR-AUC on select validation block',
        'total_seconds': elapsed, 'dataset_load_or_generate_seconds': dataset_seconds, 'config': cfg,
        'successful_models': len(results), 'failures': len(failure),
        'latency_scope': 'Prepared historical feature frame through preprocessing and all model dependencies; excludes feature-state storage/lookup.',
        'tuning_method': 'Seeded ParameterSampler randomized holdout search, 2 trials on train/tune chronology',
        'ablation_limitations': 'Fusion score removals retain graph/sequence information in base supervised model; retrained feature-family variants address direct inputs but correlated proxy signals remain.'}
    dump(out/'experiment_metadata.json', metadata)
    dump(modeldir/'metadata.json', metadata)
    report = f'# Experiment {tag}\n\nSynthetic data only; not upay production performance.\n\n'
    report += f'Rows: {len(d):,}; fraud prevalence: {d.fraud_label.mean():.3%}. Validation and leakage audit passed.\n\n'
    report += f'Validation-selected supervised model: **{best}**. Validation-selected system: **{winner.model}**. Test PR-AUC: {winner.pr_auc:.4f}; precision {winner.precision:.4f}; recall {winner.recall:.4f}; F1 {winner.f1:.4f}. Threshold {winner.threshold:.6f}; calibration {winner.calibration}.\n\n'
    report += f'Total wall time: {elapsed:.1f}s. Prepared-frame batch-1 P95 latency: {winner.p95_inference_latency_ms:.3f} ms.\n\n{error}\n'
    report += 'Metrics: `results/'+tag+'/metrics.csv`; ablations: `ablation_study.csv` and `ablation_by_fraud_type.csv`; calibration: `calibration_metrics.csv`; SHAP: `shap_summary.png`; exact failures: `failures.json`.\n\n'
    report += 'Train 70%; five validation blocks of 3% each for tune, fusion, calibration fit, calibration/system selection and threshold; test final 15%. No test tuning. Dates, overlap and class support are saved. Business costs are assumptions. Bootstrap resamples UTC days, conditional on one seed/model fit. Larger datasets include more users at the same history depth, so this is population scaling rather than a fixed-population learning curve.\n'
    (ROOT/f'reports/experiment_reports/experiment_{tag}.md').write_text(report, encoding='utf-8')
    logging.info('COMPLETE %s %.1fs (%d failures)', tag, elapsed, len(failure))
    del d, frames, x, bundles, scores, preds, pred_rows
    gc.collect()
    return metadata
