from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.config import DEFAULT_HORIZON_DAYS, MODELS_DIR, ensure_directories, output_stem, target_column
from src.data_loading import load_modeling_dataset
from src.evaluation import save_model_outputs, save_not_run_metrics
from src.modeling import dependency_available


def build_weekly_aggregate(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    daily = (
        df.groupby(["fecha", "split"], as_index=False)
        .agg(y=(target_col, "mean"))
        .sort_values("fecha")
    )
    frames = []
    for split, part in daily.groupby("split", sort=False):
        weekly = (
            part.set_index("fecha")[["y"]]
            .resample("W")
            .mean()
            .dropna()
            .reset_index()
            .rename(columns={"fecha": "ds"})
        )
        weekly["split"] = split
        frames.append(weekly)
    return pd.concat(frames, ignore_index=True).sort_values("ds")


def make_output_frame(part: pd.DataFrame, split: str, target_col: str, horizon_days: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parcela_id": "aggregate",
            "nombre_parcela": "aggregate",
            "fecha": part["ds"].values,
            "target_date": (part["ds"] + pd.Timedelta(days=horizon_days)).values,
            target_col: part["y"].values,
            "split": split,
        }
    )


def fit_prophet(train_part: pd.DataFrame):
    from prophet import Prophet

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=0.80,
    )
    model.fit(train_part[["ds", "y"]])
    return model


def predict_prophet(model, part: pd.DataFrame) -> np.ndarray:
    forecast = model.predict(part[["ds"]])
    return forecast["yhat"].clip(0, 1).to_numpy(dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    args = parser.parse_args()
    target_col = target_column(args.horizon_days)

    ensure_directories()
    if not dependency_available("prophet"):
        print(
            save_not_run_metrics(
                "prophet",
                "not_run_dependency_missing",
                "Install Prophet with: pip install prophet",
                target_col=target_col,
                horizon_days=args.horizon_days,
            )
        )
        return

    df = load_modeling_dataset(horizon_days=args.horizon_days)
    weekly = build_weekly_aggregate(df, target_col=target_col)
    train = weekly[weekly["split"] == "train"].copy()
    val = weekly[weekly["split"] == "val"].copy()
    test = weekly[weekly["split"] == "test"].copy()
    if min(len(train), len(val), len(test)) == 0:
        print(
            save_not_run_metrics(
                "prophet",
                "not_run_insufficient_data",
                "Weekly aggregate split has no rows for train, validation, or test.",
                target_col=target_col,
                horizon_days=args.horizon_days,
            )
        )
        return

    try:
        model = fit_prophet(train)
        val_pred = predict_prophet(model, val)
        refit = fit_prophet(pd.concat([train, val], ignore_index=True).sort_values("ds"))
        test_pred = predict_prophet(refit, test)
        joblib.dump(refit, MODELS_DIR / f"{output_stem('prophet', args.horizon_days)}.joblib")
    except Exception as exc:
        print(
            save_not_run_metrics(
                "prophet",
                "not_run_error",
                f"Prophet failed: {exc}",
                target_col=target_col,
                horizon_days=args.horizon_days,
            )
        )
        return

    train_df = make_output_frame(train, "train", target_col, args.horizon_days)
    val_df = make_output_frame(val, "val", target_col, args.horizon_days)
    test_df = make_output_frame(test, "test", target_col, args.horizon_days)
    print(
        save_model_outputs(
            "prophet",
            train_df,
            val_df,
            test_df,
            val_pred,
            test_pred,
            ["weekly_aggregate_target_history"],
            target_col=target_col,
            horizon_days=args.horizon_days,
            notes=f"Prophet fitted on weekly aggregate mean {target_col}, not parcel-level series.",
        )
    )


if __name__ == "__main__":
    main()
