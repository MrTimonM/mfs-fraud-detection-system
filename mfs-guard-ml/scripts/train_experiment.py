import argparse
import os
import sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '4')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment_runner import run, logging_setup

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', required=True)
    args = parser.parse_args()
    size = int(args.size[:-1])*1000 if args.size.endswith('k') else int(args.size)
    logging_setup()
    run(size)
