import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import DEFAULT_HORIZON_DAYS, METRICS_DIR, PLOTS_DIR, PREDICTIONS_DIR, TARGET, output_stem


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def build_prediction_frame(df: pd.DataFrame, y_pred, model_name: str, target_col: str = TARGET) -> pd.DataFrame:
    cols = ["parcela_id", "nombre_parcela", "fecha", "target_date", target_col, "split"]
    available = [col for col in cols if col in df.columns]
    pred = df[available].copy()
    pred = pred.rename(columns={target_col: "y_true"})
    pred["y_pred"] = np.asarray(y_pred, dtype=float)
    pred["model_name"] = model_name
    ordered = [
        "parcela_id",
        "nombre_parcela",
        "fecha",
        "target_date",
        "y_true",
        "y_pred",
        "model_name",
        "split",
    ]
    return pred[[col for col in ordered if col in pred.columns]]


def save_plot(predictions: pd.DataFrame, model_name: str, horizon_days: int = DEFAULT_HORIZON_DAYS) -> Path:
    stem = output_stem(model_name, horizon_days)
    path = PLOTS_DIR / f"{stem}_real_vs_pred.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    sample = predictions[predictions["split"] == "test"].copy()
    if sample.empty:
        sample = predictions.copy()
    if len(sample) > 2500:
        sample = sample.sample(2500, random_state=42)
    ax.scatter(sample["y_true"], sample["y_pred"], s=10, alpha=0.45)
    low = min(sample["y_true"].min(), sample["y_pred"].min())
    high = max(sample["y_true"].max(), sample["y_pred"].max())
    ax.plot([low, high], [low, high], color="black", linewidth=1)
    ax.set_title(f"{model_name} ({horizon_days} días): real vs predicho")
    ax.set_xlabel("Real")
    ax.set_ylabel("Predicho")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_metrics(metrics: dict[str, Any], model_name: str, horizon_days: int = DEFAULT_HORIZON_DAYS) -> Path:
    stem = output_stem(model_name, horizon_days)
    path = METRICS_DIR / f"{stem}_metrics.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    update_model_comparison()
    return path


def save_model_outputs(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_pred,
    test_pred,
    features: list[str],
    target_col: str = TARGET,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    status: str = "ok",
    notes: str = "",
) -> dict[str, Any]:
    stem = output_stem(model_name, horizon_days)
    pred_val = build_prediction_frame(val_df, val_pred, model_name, target_col=target_col)
    pred_test = build_prediction_frame(test_df, test_pred, model_name, target_col=target_col)
    predictions = pd.concat([pred_val, pred_test], ignore_index=True)
    predictions_path = PREDICTIONS_DIR / f"{stem}_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    plot_path = save_plot(predictions, model_name, horizon_days=horizon_days)
    test_metrics = regression_metrics(test_df[target_col], test_pred)
    metrics = {
        "model_name": model_name,
        "status": status,
        "notes": notes,
        **test_metrics,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "features_used": features,
        "target": target_col,
        "horizon_days": int(horizon_days),
        "predictions_file": str(predictions_path),
        "plot_file": str(plot_path),
    }
    save_metrics(metrics, model_name, horizon_days=horizon_days)
    return metrics


def save_not_run_metrics(
    model_name: str,
    status: str,
    notes: str,
    target_col: str = TARGET,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> dict[str, Any]:
    metrics = {
        "model_name": model_name,
        "status": status,
        "notes": notes,
        "mae": None,
        "rmse": None,
        "r2": None,
        "n_train": 0,
        "n_val": 0,
        "n_test": 0,
        "features_used": [],
        "target": target_col,
        "horizon_days": int(horizon_days),
    }
    save_metrics(metrics, model_name, horizon_days=horizon_days)
    return metrics


def update_model_comparison() -> Path:
    rows = []
    horizon_metrics = list(METRICS_DIR.glob("*_h*_metrics.json"))
    metric_paths = horizon_metrics if horizon_metrics else list(METRICS_DIR.glob("*_metrics.json"))
    for path in sorted(metric_paths):
        if path.name == "model_comparison_metrics.json":
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append(
            {
                "model_name": data.get("model_name"),
                "horizon_days": data.get("horizon_days"),
                "mae": data.get("mae"),
                "rmse": data.get("rmse"),
                "r2": data.get("r2"),
                "status": data.get("status", "ok"),
                "notes": data.get("notes", ""),
            }
        )
    comparison = pd.DataFrame(rows)
    out = METRICS_DIR / "model_comparison.csv"
    comparison.to_csv(out, index=False)
    return out
