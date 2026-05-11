import numpy as np
import pandas as pd

from src.config import DEFAULT_HORIZON_DAYS, TARGET_DATE, TARGET_TOLERANCE_DAYS, target_column


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def add_spectral_indices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = ["B8A", "B4", "B12"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para calcular índices espectrales: {missing}")
    nir = pd.to_numeric(df["B8A"], errors="coerce")
    red = pd.to_numeric(df["B4"], errors="coerce")
    swir = pd.to_numeric(df["B12"], errors="coerce")
    df["ndvi"] = _safe_divide(nir - red, nir + red)
    df["ndmi"] = _safe_divide(nir - swir, nir + swir)
    df["msi"] = _safe_divide(swir, nir)
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    iso = df["fecha"].dt.isocalendar()
    df["year"] = df["fecha"].dt.year
    df["month"] = df["fecha"].dt.month
    df["dayofyear"] = df["fecha"].dt.dayofyear
    df["weekofyear"] = iso.week.astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 366)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 366)
    return df


def add_group_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["nombre_parcela", "fecha"]).copy()
    group = df.groupby("nombre_parcela", sort=False)
    for col, prefix in [
        ("stress_index", "stress"),
        ("ndvi", "ndvi"),
        ("ndmi", "ndmi"),
        ("msi", "msi"),
    ]:
        if col not in df.columns:
            continue
        df[f"{prefix}_lag_1"] = group[col].shift(1)
        df[f"{prefix}_lag_2"] = group[col].shift(2)
        df[f"{prefix}_diff_1"] = df[col] - df[f"{prefix}_lag_1"]
        df[f"{prefix}_roll_mean_3"] = group[col].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
    df["days_since_previous_observation"] = group["fecha"].diff().dt.days
    df["parcela_code"] = pd.factorize(df["nombre_parcela"].astype(str))[0]
    return df


def add_target(df: pd.DataFrame, horizon_days: int = DEFAULT_HORIZON_DAYS) -> pd.DataFrame:
    frames = []
    tolerance = pd.Timedelta(days=TARGET_TOLERANCE_DAYS)
    target_col = target_column(horizon_days)

    for _, group in df.groupby("nombre_parcela", sort=False):
        g = group.sort_values("fecha").copy()
        dates = g["fecha"].to_numpy(dtype="datetime64[ns]")
        values = g["stress_index"].to_numpy(dtype=float)
        target_values = np.full(len(g), np.nan)
        target_dates = np.full(len(g), np.datetime64("NaT"), dtype="datetime64[ns]")
        target_deltas = np.full(len(g), np.nan)

        for i, current in enumerate(dates):
            desired = current + np.timedelta64(int(horizon_days), "D")
            start = np.searchsorted(dates, current + np.timedelta64(1, "D"), side="left")
            if start >= len(dates):
                continue
            candidates = np.arange(start, len(dates))
            diffs = np.abs(dates[candidates] - desired)
            best_pos = candidates[int(np.argmin(diffs))]
            if pd.Timedelta(diffs[int(np.argmin(diffs))]) <= tolerance:
                target_values[i] = values[best_pos]
                target_dates[i] = dates[best_pos]
                target_deltas[i] = float((dates[best_pos] - current) / np.timedelta64(1, "D"))

        g[target_col] = target_values
        g[TARGET_DATE] = pd.to_datetime(target_dates)
        g["target_delta_days"] = target_deltas
        frames.append(g)

    return pd.concat(frames, ignore_index=True)


def build_modeling_dataset(raw_df: pd.DataFrame, horizon_days: int = DEFAULT_HORIZON_DAYS) -> pd.DataFrame:
    target_col = target_column(horizon_days)
    df = raw_df.copy()
    df["nombre_parcela"] = df["nombre_parcela"].fillna(df["parcela_id"].astype(str))
    numeric_cols = [col for col in df.columns if col not in ["nombre_parcela", "fecha"]]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["nombre_parcela", "fecha"])
    df = add_spectral_indices(df)
    df = add_temporal_features(df)
    df = add_group_features(df)
    df = add_target(df, horizon_days=horizon_days)
    df = df.dropna(subset=[target_col, TARGET_DATE])
    feature_cols = [
        col
        for col in df.select_dtypes(include=["number", "bool"]).columns
        if col not in [target_col, "target_delta_days"]
    ]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df.groupby("nombre_parcela")[feature_cols].transform(lambda x: x.ffill().bfill())
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    return df.sort_values(["fecha", "nombre_parcela"]).reset_index(drop=True)
