"""STEP 4a: export the selected bundle's parts to backend/models/final (no refitting)."""
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
from src.config import ROOT, config, dump

OUT = ROOT/'backend/models/final'


def main():
    selection = json.loads((ROOT/'results/final_common_test/final_selection.json').read_text())
    tag, name, label = selection['scale'], selection['artifact_model'], selection['selected_model']
    bundle = joblib.load(ROOT/f'models/{tag}/{name}.joblib')
    comparison = json.loads((ROOT/'results/final_common_test/model_comparison.json').read_text())
    row = next(r for r in comparison['models'] if r['model'] == label)
    points = comparison['operating_points'][label]
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.model, OUT/'model.joblib', compress=3)
    joblib.dump(bundle.preprocessor, OUT/'preprocessor.joblib', compress=3)
    joblib.dump(bundle.calibration, OUT/'calibrator.joblib', compress=3)
    # Supporting behavioral anomaly layer (rank score, abstains on cold start).
    joblib.dump(joblib.load(ROOT/'models/500k/IsolationForest.joblib').anomaly, OUT/'anomaly.joblib', compress=3)
    dump(OUT/'feature_list.json', list(bundle.columns))
    training = json.loads((ROOT/f'models/{tag}/metadata.json').read_text())
    dump(OUT/'thresholds.json', {
        'source': 'validation threshold block of the training run; never tuned on the common final test set',
        'ml_probability': {'monitor': points['monitor'], 'step_up': points['step_up'], 'hold': points['hold']},
        'cost_optimal_reference': points['cost_optimal'],
        'rule_score': {'monitor': 20, 'step_up': 40, 'hold': 70},
        'reject_requires': {'ml_at_least': 'hold', 'rule_score_at_least': 70},
        'risk_level': {'MEDIUM': points['monitor'], 'HIGH': points['step_up'], 'CRITICAL': points['hold']},
        'rules': config()['rules']})
    dump(OUT/'model_metadata.json', {
        'model_name': name, 'artifact': f'models/{tag}/{name}.joblib', 'version': f'{label}-v1',
        'training_dataset_size': training['dataset_size'], 'training_seed': training['seed'],
        'training_date': datetime.fromtimestamp((ROOT/f'models/{tag}/{name}.joblib').stat().st_mtime, timezone.utc).isoformat(),
        'exported_at_utc': datetime.now(timezone.utc).isoformat(), 'feature_count': len(bundle.columns),
        'calibration': bundle.calibration.method, 'PR-AUC': row['pr_auc'], 'ROC-AUC': row['roc_auc'],
        'metrics_source': 'common final test set (seed 2026, 100k, synthetic)',
        'recommended_threshold': points['step_up'], 'synthetic_only': True})
    shutil.copy(ROOT/'config.yaml', OUT/'config_snapshot.yaml')
    print(f'Exported {label} to {OUT}')


if __name__ == '__main__':
    main()
