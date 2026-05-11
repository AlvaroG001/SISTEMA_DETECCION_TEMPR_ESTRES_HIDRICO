from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_directories
from src.modeling import train_random_forest


if __name__ == "__main__":
    ensure_directories()
    print(train_random_forest())
