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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    args = parser.parse_args()
    target_col = target_column(args.horizon_days)

    ensure_directories()
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception:
        print(
            save_not_run_metrics(
                "sarimax",
                "not_run_dependency_missing",
                "Install statsmodels with: pip install statsmodels",
                target_col=target_col,
                horizon_days=args.horizon_days,
            )
        )
        return

    df = load_modeling_dataset(horizon_days=args.horizon_days)
    weekly = (
        df.groupby(["fecha", "split"], as_index=False)[target_col]
        .mean()
        .sort_values("fecha")
        .set_index("fecha")
    )
    train = weekly[weekly["split"] == "train"][target_col]
    val = weekly[weekly["split"] == "val"][target_col]
    test = weekly[weekly["split"] == "test"][target_col]
    try:
        model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0), enforce_stationarity=False)
        result = model.fit(disp=False)
        val_pred_series = result.forecast(steps=len(val))
        refit_series = pd.concat([train, val])
        refit = SARIMAX(refit_series, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0), enforce_stationarity=False).fit(disp=False)
        test_pred_series = refit.forecast(steps=len(test))
        joblib.dump(refit, MODELS_DIR / f"{output_stem('sarimax', args.horizon_days)}.joblib")
    except Exception as exc:
        print(
            save_not_run_metrics(
                "sarimax",
                "not_run_error",
                f"SARIMAX failed: {exc}",
                target_col=target_col,
                horizon_days=args.horizon_days,
            )
        )
        return

    val_dates = val.index.to_frame(index=False).rename(columns={"fecha": "fecha"})
    test_dates = test.index.to_frame(index=False).rename(columns={"fecha": "fecha"})
    val_df = pd.DataFrame(
        {
            "parcela_id": "aggregate",
            "nombre_parcela": "aggregate",
            "fecha": val.index,
            "target_date": val.index,
            target_col: val.values,
            "split": "val",
        }
    )
    test_df = pd.DataFrame(
        {
            "parcela_id": "aggregate",
            "nombre_parcela": "aggregate",
            "fecha": test.index,
            "target_date": test.index,
            target_col: test.values,
            "split": "test",
        }
    )
    train_df = pd.DataFrame(
        {
            "parcela_id": "aggregate",
            "nombre_parcela": "aggregate",
            "fecha": train.index,
            "target_date": train.index,
            target_col: train.values,
            "split": "train",
        }
    )
    print(
        save_model_outputs(
            "sarimax",
            train_df,
            val_df,
            test_df,
            np.asarray(val_pred_series),
            np.asarray(test_pred_series),
            ["aggregate_target_history"],
            target_col=target_col,
            horizon_days=args.horizon_days,
            notes="SARIMAX baseline fitted on the aggregate mean stress series, not parcel-level series.",
        )
    )


if __name__ == "__main__":
    main()
