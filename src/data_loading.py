import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import DEFAULT_HORIZON_DAYS, PROCESSED_DATA_FILE, RAW_DATA_FILE, TARGET, TARGET_DATE, processed_data_file


def add_project_root_to_path(script_file: str) -> None:
    root = Path(script_file).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def read_raw_dataset(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset bruto: {path}")
    df = pd.read_csv(path)
    if "fecha" not in df.columns:
        raise ValueError("El CSV bruto debe contener una columna 'fecha'.")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])
    return df


def load_modeling_dataset(path: Path | None = None, horizon_days: int = DEFAULT_HORIZON_DAYS) -> pd.DataFrame:
    if path is None:
        path = processed_data_file(horizon_days)
        if not path.exists() and int(horizon_days) == DEFAULT_HORIZON_DAYS:
            path = PROCESSED_DATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset procesado: {path}. Ejecuta primero python scripts/prepare_dataset.py --horizon-days {horizon_days}."
        )
    df = pd.read_csv(path)
    for col in ["fecha", TARGET_DATE]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def get_feature_columns(
    df: pd.DataFrame,
    target_col: str = TARGET,
    extra_exclude: Iterable[str] = (),
) -> list[str]:
    exclude = {
        target_col,
        TARGET_DATE,
        "target_delta_days",
        "stress_index",
        "fecha",
        "nombre_parcela",
        "split",
    }
    exclude.update(extra_exclude)
    numeric = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    return [
        col
        for col in numeric
        if col not in exclude and not col.startswith("target_stress_") and df[col].notna().any()
    ]
