import os
import sys
import warnings
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '4')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from threadpoolctl import threadpool_limits
from src.common_test import run

if __name__ == '__main__':
    warnings.filterwarnings('ignore', message='X does not have valid feature names.*')
    with threadpool_limits(limits=4):
        table = run()
    print(table[['model', 'pr_auc', 'recall', 'precision', 'false_positive_rate', 'p95_latency_ms',
                 'tiered_total_business_cost']].to_string(index=False))
