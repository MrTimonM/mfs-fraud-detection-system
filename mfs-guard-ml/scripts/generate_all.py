import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import config, paths, dump
from src.data_generator import generate
from src.data_validation import validate

if __name__ == '__main__':
    for size in config()['sizes']:
        d = generate(size)
        dump(paths(size)['results']/'dataset_validation.json', validate(d, size))
        print(f'Generated {size:,}', flush=True)
