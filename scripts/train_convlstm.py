from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_directories
from src.modeling import train_convlstm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-days", type=int, default=7)
    args = parser.parse_args()
    ensure_directories()
    print(train_convlstm(horizon_days=args.horizon_days))
