from pathlib import Path
import json
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = ROOT / "outputs" / "predictions"
METRICS_DIR = ROOT / "outputs" / "metrics"
PLOTS_DIR = ROOT / "outputs" / "plots"
GEOJSON_PATH = ROOT / "data" / "processed" / "parcels.geojson"
DATASET_H7_PATH = ROOT / "data" / "processed" / "dataset_modeling_h7.csv"
DATASET_PATH = ROOT / "data" / "processed" / "dataset_modeling.csv"
CLIMATE_PATH = ROOT / "data" / "raw" / "climate_forecast.csv"

st.set_page_config(page_title="Estrés hídrico por parcela", layout="wide")
st.title("Predicción de estrés hídrico por parcela")


@st.cache_data
def load_dataset():
    path = DATASET_H7_PATH if DATASET_H7_PATH.exists() else DATASET_PATH
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ["fecha", "target_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


@st.cache_data
def load_geojson():
    if not GEOJSON_PATH.exists():
        return None
    return json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))


@st.cache_data
def prediction_catalog():
    rows = []
    pattern = re.compile(r"(.+)_h(\d+)_predictions\.csv$")
    for path in sorted(PREDICTIONS_DIR.glob("*_h*_predictions.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        rows.append({"model_name": match.group(1), "horizon_days": int(match.group(2)), "path": path})
    return pd.DataFrame(rows)


@st.cache_data
def load_predictions(path: Path):
    df = pd.read_csv(path)
    for col in ["fecha", "target_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


@st.cache_data
def load_metrics(model_name: str, horizon_days: int):
    path = METRICS_DIR / f"{model_name}_h{horizon_days}_metrics.json"
    if not path.exists():
        return {}
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if metrics.get("precision_pct") is None:
        metrics["precision_pct"] = precision_from_mae(metrics.get("mae"))
    return metrics


@st.cache_data
def load_comparison():
    path = METRICS_DIR / "model_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    comparison = pd.read_csv(path)
    if "precision_pct" not in comparison.columns:
        comparison["precision_pct"] = comparison["mae"].apply(precision_from_mae)
    else:
        comparison["precision_pct"] = comparison.apply(
            lambda row: precision_from_mae(row.get("mae"))
            if pd.isna(row.get("precision_pct"))
            else row.get("precision_pct"),
            axis=1,
        )
    return comparison


def precision_from_mae(mae):
    if mae is None or pd.isna(mae):
        return None
    return max(0.0, min(100.0, (1 - float(mae)) * 100))


def format_percent(value):
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.2f}%"


def climate_file_mtime() -> float:
    return CLIMATE_PATH.stat().st_mtime if CLIMATE_PATH.exists() else 0.0


@st.cache_data
def load_climate(_mtime: float):
    if not CLIMATE_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(CLIMATE_PATH)
    if "forecast_date" in df.columns:
        df["forecast_date"] = pd.to_datetime(df["forecast_date"], errors="coerce")
    return df


def normalize_climate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "forecast_date" in df.columns:
        df["forecast_date"] = pd.to_datetime(df["forecast_date"], errors="coerce")
    return df


def climate_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "forecast_date": "2026-05-12",
                "nombre_parcela": "A1",
                "temp_min_c": 18.2,
                "temp_max_c": 31.4,
                "temp_mean_c": 24.7,
                "precipitation_mm": 0.0,
                "humidity_pct": 58,
                "wind_m_s": 3.1,
                "et0_mm": 4.8,
            }
        ]
    )


def stress_color(value):
    if pd.isna(value):
        return "#9ca3af"
    if value < 0.33:
        return "#2ca25f"
    if value < 0.66:
        return "#f59e0b"
    return "#dc2626"


def polygon_center(coords):
    lon = [p[0] for p in coords]
    lat = [p[1] for p in coords]
    return sum(lon) / len(lon), sum(lat) / len(lat)


def draw_map(geojson, values: pd.DataFrame, value_col: str, label: str):
    fig = go.Figure()
    centers = []
    no_data_count = 0
    value_lookup = {}
    if not values.empty and "nombre_parcela" in values.columns and value_col in values.columns:
        value_lookup = values.set_index("nombre_parcela")[value_col].to_dict()

    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        name = str(props.get("nombre_parcela", "")).strip()
        generated = bool(props.get("generated_name", False))
        coords = feature.get("geometry", {}).get("coordinates", [[]])[0]
        if not coords:
            continue
        value = value_lookup.get(name)
        has_value = value is not None and not pd.isna(value)
        if not has_value:
            no_data_count += 1
        if generated:
            hover = f"{name}: Sin nombre en Excel o sin datos Sentinel/predicción"
        elif has_value:
            hover = f"{name}: {label} = {float(value):.3f}"
        else:
            hover = f"{name}: Sin datos Sentinel/predicción"
        lon = [p[0] for p in coords]
        lat = [p[1] for p in coords]
        fig.add_trace(
            go.Scattermapbox(
                lon=lon,
                lat=lat,
                mode="lines",
                fill="toself",
                fillcolor=stress_color(value if has_value else None),
                line={"color": "#111827", "width": 1},
                name=name,
                text=[hover] * len(lon),
                hoverinfo="text",
                showlegend=False,
            )
        )
        centers.append(polygon_center(coords))

    center_lon, center_lat = (-60.7, -21.84)
    if centers:
        center_lon = sum(c[0] for c in centers) / len(centers)
        center_lat = sum(c[1] for c in centers) / len(centers)
    fig.update_layout(
        mapbox={"style": "open-street-map", "center": {"lon": center_lon, "lat": center_lat}, "zoom": 11},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=560,
    )
    st.plotly_chart(fig, width="stretch")
    if no_data_count:
        st.warning(f"{no_data_count} parcelas del GeoJSON no cruzan con el CSV o no tienen predicción; se muestran en gris.")


def plot_parcel_history(parcel_df: pd.DataFrame, parcel_pred: pd.DataFrame | None = None):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=parcel_df["fecha"],
            y=parcel_df["stress_index"],
            mode="lines",
            name="Real histórico",
        )
    )
    if parcel_pred is not None and not parcel_pred.empty:
        fig.add_trace(
            go.Scatter(
                x=parcel_pred["target_date"],
                y=parcel_pred["y_pred"],
                mode="lines+markers",
                name="Predicción",
                line={"width": 3},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=parcel_pred["target_date"],
                y=parcel_pred["y_true"],
                mode="markers",
                name="Real evaluado",
            )
        )
    fig.update_layout(height=460, margin={"l": 0, "r": 0, "t": 20, "b": 0}, yaxis_title="valor")
    st.plotly_chart(fig, width="stretch")

def best_model_summary(comparison: pd.DataFrame) -> None:
    if comparison.empty or "horizon_days" not in comparison.columns:
        return
    ok = comparison[(comparison["status"] == "ok") & comparison["mae"].notna()].copy()
    if ok.empty:
        return

    best_overall = ok.sort_values(["mae", "rmse"], na_position="last").iloc[0]
    best_by_horizon = ok.sort_values(["horizon_days", "mae", "rmse"], na_position="last").groupby("horizon_days").head(1)
    pivot = ok.pivot_table(index="model_name", columns="horizon_days", values="mae", aggfunc="min")
    if {5, 7}.issubset(set(pivot.columns)):
        pivot["mejor_horizonte"] = pivot.apply(lambda row: "5 días" if row[5] <= row[7] else "7 días", axis=1)
        pivot["diferencia_mae_7_menos_5"] = pivot[7] - pivot[5]
        horizon_counts = pivot["mejor_horizonte"].value_counts()
        most_common = horizon_counts.idxmax()
    else:
        most_common = "no disponible"

    st.subheader("Mejores modelos")
    cols = st.columns(4)
    cols[0].metric("Mejor global", f"{best_overall['model_name']} ({int(best_overall['horizon_days'])} días)")
    cols[1].metric("MAE", f"{best_overall['mae']:.4f}")
    cols[2].metric("RMSE", f"{best_overall['rmse']:.4f}")
    cols[3].metric("Precisión", format_percent(best_overall.get("precision_pct")))

    st.markdown(
        "El mejor modelo se decide principalmente por **menor MAE** y, como desempate práctico, por **menor RMSE**. "
        "En estos resultados suelen ganar los modelos tabulares porque aprovechan directamente los lags, medias móviles "
        "e índices espectrales ya calculados."
    )
    if most_common != "no disponible":
        st.info(
            f"Comparando los mismos modelos entre horizontes, el horizonte que aparece más veces como mejor es **{most_common}**. "
            "Esto es esperable cuando el horizonte más corto conserva mejor la relación con las observaciones recientes."
        )
    best_display = best_by_horizon[["horizon_days", "model_name", "mae", "rmse", "r2", "precision_pct"]].copy()
    best_display["precision_pct"] = best_display["precision_pct"].map(format_percent)
    st.dataframe(best_display, width="stretch", hide_index=True)
    if {5, 7}.issubset(set(pivot.columns)):
        st.caption("Comparación por modelo: MAE menor indica mejor rendimiento.")
        pivot_display = pivot.reset_index().sort_values("diferencia_mae_7_menos_5", ascending=False)
        pivot_display["interpretación"] = pivot_display["mejor_horizonte"].apply(
            lambda value: f"Ha funcionado mejor a {value}."
        )
        st.dataframe(pivot_display, width="stretch", hide_index=True)


dataset = load_dataset()
geojson = load_geojson()
catalog = prediction_catalog()

tab_parcel, tab_climate, tab_performance = st.tabs(
    ["Información por parcela", "Predicción climática", "Rendimiento del modelo"]
)

with tab_parcel:
    if dataset.empty:
        st.info("No hay dataset procesado. Ejecuta python scripts/prepare_dataset.py --horizon-days 7.")
    else:
        latest_stress = dataset.sort_values("fecha").groupby("nombre_parcela", as_index=False).tail(1)
        control_col, horizon_col, model_col, metric_col_1, metric_col_2 = st.columns([1.6, 1, 1.4, 1, 1])
        with control_col:
            parcels = sorted(dataset["nombre_parcela"].dropna().astype(str).unique())
            selected_parcel = st.selectbox("Parcela", parcels, key="parcel_history")
        selected_history_horizon = None
        selected_history_model = None
        parcel_pred = pd.DataFrame()
        if not catalog.empty:
            history_horizons = sorted(catalog["horizon_days"].unique())
            with horizon_col:
                selected_history_horizon = st.selectbox(
                    "Horizonte",
                    history_horizons,
                    index=history_horizons.index(7) if 7 in history_horizons else 0,
                    format_func=lambda x: f"{x} días",
                    key="parcel_horizon",
                )
            history_models = sorted(catalog[catalog["horizon_days"] == selected_history_horizon]["model_name"].unique())
            default_model = "random_forest" if "random_forest" in history_models else history_models[0]
            with model_col:
                selected_history_model = st.selectbox(
                    "Modelo",
                    history_models,
                    index=history_models.index(default_model),
                    key="parcel_model",
                )
            pred_path = catalog[
                (catalog["horizon_days"] == selected_history_horizon)
                & (catalog["model_name"] == selected_history_model)
            ]["path"].iloc[0]
            pred_df = load_predictions(pred_path)
            parcel_pred = pred_df[pred_df["nombre_parcela"].astype(str) == selected_parcel].sort_values("target_date")
        parcel_df = dataset[dataset["nombre_parcela"].astype(str) == selected_parcel].sort_values("fecha")
        if parcel_df.empty:
            st.warning("La parcela seleccionada no tiene datos Sentinel.")
        else:
            last = parcel_df.iloc[-1]
            with metric_col_1:
                st.metric("Último stress_index", f"{last['stress_index']:.3f}")
            with metric_col_2:
                st.metric("Última fecha disponible", str(last["fecha"].date()))
            if selected_parcel == "D1":
                st.info(
                    "D1 aparece muy roja en Random Forest porque su histórico reciente ya muestra estrés alto; "
                    "no es un problema de nombre."
                )

        if geojson:
            draw_map(geojson, latest_stress, "stress_index", "último stress_index")
        else:
            st.warning("No se encontró data/processed/parcels.geojson.")

        if not parcel_df.empty:
            if selected_history_model is not None:
                st.subheader(
                    f"Histórico y predicción de la parcela ({selected_history_model}, {selected_history_horizon} días)"
                )
            else:
                st.subheader("Histórico de la parcela")
            plot_parcel_history(parcel_df, parcel_pred=parcel_pred)

with tab_climate:
    climate = load_climate(climate_file_mtime())
    uploaded_climate = st.file_uploader("Cargar CSV climático", type=["csv"])
    if uploaded_climate is not None:
        climate = normalize_climate(pd.read_csv(uploaded_climate))
    if climate.empty:
        st.info(
            "No se ve ninguna predicción climática porque no existe `data/raw/climate_forecast.csv` "
            "y todavía no se ha cargado ningún CSV desde esta pantalla."
        )
        template = climate_template()
        st.caption("Formato esperado. Puedes usarlo como plantilla:")
        st.dataframe(template, width="stretch", hide_index=True)
        st.download_button(
            "Descargar plantilla CSV",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="climate_forecast_template.csv",
            mime="text/csv",
        )
    elif "forecast_date" not in climate.columns or not ({"nombre_parcela", "parcela_id"} & set(climate.columns)):
        st.error("El CSV climático debe incluir `forecast_date` y `nombre_parcela` o `parcela_id`.")
        st.dataframe(climate.head(20), width="stretch")
    else:
        key_col = "nombre_parcela" if "nombre_parcela" in climate.columns else "parcela_id"
        climate = climate.dropna(subset=["forecast_date"])
        parcels = sorted(climate[key_col].dropna().astype(str).unique())
        selected_climate_parcel = st.selectbox("Parcela", parcels, key="climate_parcel")
        filtered = climate[climate[key_col].astype(str) == selected_climate_parcel].sort_values("forecast_date")
        min_date = filtered["forecast_date"].min().date()
        max_date = filtered["forecast_date"].max().date()
        date_range = st.date_input("Rango de fechas", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            filtered = filtered[(filtered["forecast_date"] >= start) & (filtered["forecast_date"] <= end)]
        numeric_cols = [
            col for col in filtered.select_dtypes(include="number").columns if col not in {"parcela_id"}
        ]
        fig = go.Figure()
        for col in [c for c in ["temp_min_c", "temp_max_c", "temp_mean_c", "precipitation_mm", "humidity_pct", "et0_mm"] if c in numeric_cols]:
            fig.add_trace(go.Scatter(x=filtered["forecast_date"], y=filtered[col], mode="lines+markers", name=col))
        if fig.data:
            fig.update_layout(height=380, margin={"l": 0, "r": 0, "t": 20, "b": 0})
            st.plotly_chart(fig, width="stretch")
        st.dataframe(filtered, width="stretch", hide_index=True)

with tab_performance:
    if catalog.empty:
        st.info("No hay predicciones con sufijo de horizonte. Ejecuta python scripts/train_all_models.py --horizons 5 7.")
    else:
        horizons = sorted(catalog["horizon_days"].unique())
        selected_horizon = st.selectbox("Horizonte", horizons, format_func=lambda x: f"{x} días")
        available = catalog[catalog["horizon_days"] == selected_horizon].sort_values("model_name")
        selected_model = st.selectbox("Modelo", available["model_name"].tolist())
        pred_path = available[available["model_name"] == selected_model]["path"].iloc[0]
        predictions = load_predictions(pred_path)
        metrics = load_metrics(selected_model, selected_horizon)

        metric_cols = st.columns(4)
        metric_cols[0].metric("MAE", f"{metrics.get('mae', float('nan')):.4f}" if metrics.get("mae") is not None else "NA")
        metric_cols[1].metric("RMSE", f"{metrics.get('rmse', float('nan')):.4f}" if metrics.get("rmse") is not None else "NA")
        metric_cols[2].metric("R2", f"{metrics.get('r2', float('nan')):.4f}" if metrics.get("r2") is not None else "NA")
        metric_cols[3].metric("Precisión", format_percent(metrics.get("precision_pct")))

        plot_path = PLOTS_DIR / f"{selected_model}_h{selected_horizon}_real_vs_pred.png"
        if plot_path.exists():
            st.image(str(plot_path), width="stretch")
        else:
            fig = go.Figure()
            test = predictions[predictions["split"] == "test"]
            fig.add_trace(go.Scatter(x=test["y_true"], y=test["y_pred"], mode="markers", name="test"))
            fig.update_layout(height=420, xaxis_title="Real", yaxis_title="Predicho")
            st.plotly_chart(fig, width="stretch")

        comparison = load_comparison()
        best_model_summary(comparison)
        if not comparison.empty and "horizon_days" in comparison.columns:
            st.subheader("Comparación global")
            comparison_display = comparison[comparison["horizon_days"] == selected_horizon].sort_values(
                "mae", na_position="last"
            ).copy()
            if "precision_pct" in comparison_display.columns:
                comparison_display["precision_pct"] = comparison_display["precision_pct"].map(format_percent)
            st.dataframe(
                comparison_display,
                width="stretch",
                hide_index=True,
            )

        st.subheader("Predicciones")
        st.dataframe(predictions.sort_values("fecha", ascending=False).head(200), width="stretch", hide_index=True)
