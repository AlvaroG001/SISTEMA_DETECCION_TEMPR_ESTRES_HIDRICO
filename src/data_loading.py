import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import PROCESSED_DATA_FILE, RAW_DATA_FILE, TARGET, TARGET_DATE


def add_project_root_to_path(script_file: str) -> None:
    root = Path(script_file).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def read_raw_dataset(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw dataset: {path}")
    df = pd.read_csv(path)
    if "fecha" not in df.columns:
        raise ValueError("The raw CSV must contain a 'fecha' column.")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])
    return df


def load_modeling_dataset(path: Path = PROCESSED_DATA_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing processed dataset: {path}. Run python scripts/prepare_dataset.py first."
        )
    df = pd.read_csv(path)
    for col in ["fecha", TARGET_DATE]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def get_feature_columns(df: pd.DataFrame, extra_exclude: Iterable[str] = ()) -> list[str]:
    exclude = {
        TARGET,
        TARGET_DATE,
        "target_delta_days",
        "stress_index",
        "fecha",
        "nombre_parcela",
        "split",
    }
    exclude.update(extra_exclude)
    numeric = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    return [col for col in numeric if col not in exclude and df[col].notna().any()]
