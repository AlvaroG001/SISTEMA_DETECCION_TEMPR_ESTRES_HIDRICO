from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ensure_directories
from src.evaluation import save_not_run_metrics
from src.modeling import dependency_available


def main():
    ensure_directories()
    if not dependency_available("prophet"):
        print(
            save_not_run_metrics(
                "prophet",
                "not_run_dependency_missing",
                "Install Prophet with: pip install prophet. The script can then be extended to fit weekly parcel series.",
            )
        )
        return
    print(
        save_not_run_metrics(
            "prophet",
            "not_implemented_optional",
            "Prophet is installed, but this MVP keeps it as an optional weekly-series extension.",
        )
    )


if __name__ == "__main__":
    main()
