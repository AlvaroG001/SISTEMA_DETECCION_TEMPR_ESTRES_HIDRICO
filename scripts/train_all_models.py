from pathlib import Path
import argparse
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODEL_SCRIPTS, ensure_directories
from src.evaluation import update_model_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare data and run all model scripts in order.")
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Do not run scripts/prepare_dataset.py before training.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[7],
        help="Prediction horizons in days to prepare/train. Example: --horizons 5 7",
    )
    args = parser.parse_args()
    ensure_directories()
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    failures = []
    for horizon in args.horizons:
        if not args.skip_prepare:
            print(f"\n=== Running prepare_dataset.py --horizon-days {horizon} ===")
            result = subprocess.run(
                [sys.executable, "-u", str(scripts_dir / "prepare_dataset.py"), "--horizon-days", str(horizon)],
                cwd=root,
            )
            if result.returncode != 0:
                print(f"prepare_dataset.py failed for horizon {horizon}; stopping before model training.")
                return result.returncode
        for script_name in MODEL_SCRIPTS:
            script_path = scripts_dir / script_name
            print(f"\n=== Running {script_name} --horizon-days {horizon} ===")
            result = subprocess.run(
                [sys.executable, "-u", str(script_path), "--horizon-days", str(horizon)],
                cwd=root,
            )
            if result.returncode != 0:
                failures.append((f"{script_name} h{horizon}", result.returncode))
                print(f"{script_name} failed for horizon {horizon} with exit code {result.returncode}")
    comparison = update_model_comparison()
    print(f"\nModel comparison saved to {comparison}")
    if failures:
        print("Some scripts failed unexpectedly:")
        for script, code in failures:
            print(f"- {script}: exit code {code}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
