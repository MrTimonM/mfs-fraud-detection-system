import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.operating_points import operating_points

if __name__ == '__main__':
    result = operating_points()
    print(f'Saved {len(result)} validation-selected operating-point comparisons.')
