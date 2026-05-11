from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, TARGET, ensure_directories
from src.data_loading import load_modeling_dataset
from src.evaluation import save_model_outputs, save_not_run_metrics


def main():
    ensure_directories()
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception:
        print(
            save_not_run_metrics(
                "sarimax",
                "not_run_dependency_missing",
                "Install statsmodels with: pip install statsmodels",
            )
        )
        return

    df = load_modeling_dataset()
    weekly = (
        df.groupby(["fecha", "split"], as_index=False)[TARGET]
        .mean()
        .sort_values("fecha")
        .set_index("fecha")
    )
    train = weekly[weekly["split"] == "train"][TARGET]
    val = weekly[weekly["split"] == "val"][TARGET]
    test = weekly[weekly["split"] == "test"][TARGET]
    try:
        model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0), enforce_stationarity=False)
        result = model.fit(disp=False)
        val_pred_series = result.forecast(steps=len(val))
        refit_series = pd.concat([train, val])
        refit = SARIMAX(refit_series, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0), enforce_stationarity=False).fit(disp=False)
        test_pred_series = refit.forecast(steps=len(test))
        joblib.dump(refit, MODELS_DIR / "sarimax.joblib")
    except Exception as exc:
        print(save_not_run_metrics("sarimax", "not_run_error", f"SARIMAX failed: {exc}"))
        return

    val_dates = val.index.to_frame(index=False).rename(columns={"fecha": "fecha"})
    test_dates = test.index.to_frame(index=False).rename(columns={"fecha": "fecha"})
    val_df = pd.DataFrame(
        {
            "parcela_id": "aggregate",
            "nombre_parcela": "aggregate",
            "fecha": val.index,
            "target_date": val.index,
            TARGET: val.values,
            "split": "val",
        }
    )
    test_df = pd.DataFrame(
        {
            "parcela_id": "aggregate",
            "nombre_parcela": "aggregate",
            "fecha": test.index,
            "target_date": test.index,
            TARGET: test.values,
            "split": "test",
        }
    )
    train_df = pd.DataFrame(
        {
            "parcela_id": "aggregate",
            "nombre_parcela": "aggregate",
            "fecha": train.index,
            "target_date": train.index,
            TARGET: train.values,
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
            notes="SARIMAX baseline fitted on the aggregate mean stress series, not parcel-level series.",
        )
    )


if __name__ == "__main__":
    main()
