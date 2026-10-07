import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.final_comparison import compare

if __name__ == '__main__':
    compare()
