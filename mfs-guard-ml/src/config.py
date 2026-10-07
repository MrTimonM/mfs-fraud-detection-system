"""Configuration and portable artifact paths."""
import json
from pathlib import Path
import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def config():
    return yaml.safe_load((ROOT / 'config.yaml').read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    path.write_text(json.dumps(value, indent=2, default=default), encoding='utf-8')


def paths(size):
    tag = f'{size // 1000}k'
    out = {name: ROOT / name / tag for name in ('models', 'results')}
    out['tag'] = tag
    for name in ('models', 'results'):
        out[name].mkdir(parents=True, exist_ok=True)
    for name in ('data/raw', 'data/processed', 'data/metadata', 'reports/dataset_cards',
                 'reports/experiment_reports', 'logs', 'results/final'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    return out
