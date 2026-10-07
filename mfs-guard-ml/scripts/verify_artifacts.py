"""Verify every completed scale and reproduce saved predictions after reload."""
import json
import hashlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from src.config import ROOT, config, dump


def verify():
    records = []
    for size in config()['sizes']:
        tag = f'{size//1000}k'
        folder = ROOT/'results'/tag
        metrics = pd.read_csv(folder/'metrics.csv')
        metadata = json.loads((folder/'experiment_metadata.json').read_text())
        frame = pd.read_parquet(ROOT/f'data/raw/mfs_{tag}.parquet').tail(10)
        predictions = pd.read_parquet(folder/'test_predictions.parquet')
        for model in metrics.loc[metrics.status == 'SUCCESS', 'model']:
            bundle = joblib.load(ROOT/f'models/{tag}/{model}.joblib')
            expected = predictions[(predictions.model_name == model) & predictions.transaction_id.isin(frame.transaction_id)]
            expected = expected.set_index('transaction_id').loc[frame.transaction_id, 'predicted_probability'].to_numpy()
            actual = bundle.predict(frame)
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
            records.append({'dataset': tag, 'model': model, 'prediction_reload': 'PASSED',
                            'max_probability_difference': float(np.max(np.abs(actual-expected)))})
        assert metadata['failures'] == 0, f'Inspect {tag}/failures.json'
        print(f'{tag}: verified {len(metrics)} trained model bundles', flush=True)
    dump(ROOT/'results/final/artifact_reload_verification.json', records)
    checks = json.loads((ROOT/'results/final/mandatory_output_check.json').read_text())
    assert all(checks.values()), 'Mandatory output missing'
    manifest = {}
    files = list((ROOT/'data/raw').glob('*')) + list((ROOT/'models').glob('*/*.joblib'))
    files += list((ROOT/'src').glob('*.py')) + list((ROOT/'data/metadata').glob('*.json'))
    files += [ROOT/'config.yaml', ROOT/'requirements.txt']
    for path in files:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            while chunk := stream.read(8*1024*1024):
                digest.update(chunk)
        manifest[path.relative_to(ROOT).as_posix()] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    dump(ROOT/'results/final/artifact_manifest.json', manifest)
    print(f'All {len(records)} model reload checks and {len(checks)} mandatory artifact checks passed.')


if __name__ == '__main__':
    verify()
