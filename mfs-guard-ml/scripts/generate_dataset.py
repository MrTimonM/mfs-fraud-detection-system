import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_generator import generate
from src.data_validation import validate
from src.config import paths, dump

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    d = generate(args.rows, args.seed)
    dump(paths(args.rows)['results']/'dataset_validation.json', validate(d, args.rows))
    print(f'Generated and validated {len(d):,} transactions.')
