"""Run every required scale sequentially, with optional completed-run resume."""
import argparse
import os
import sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import config, ROOT
from src.experiment_runner import run, logging_setup
from src.final_comparison import compare

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    for flag in ['generate', 'train', 'evaluate', 'compare']:
        parser.add_argument('--'+flag, action='store_true', help='Full workflow always includes this stage')
    args = parser.parse_args()
    logging_setup()
    for i, size in enumerate(config()['sizes'], 1):
        print(f'[{i}/5] Running {size//1000}k experiment', flush=True)
        if args.resume and (ROOT/f'results/{size//1000}k/experiment_metadata.json').exists():
            continue
        run(size)
    from src.report_supplements import supplements
    supplements()
    checks = compare()
    if not all(checks.values()):
        raise RuntimeError('Mandatory output check failed')
