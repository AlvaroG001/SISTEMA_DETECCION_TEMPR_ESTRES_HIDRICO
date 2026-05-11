from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = ROOT / "outputs" / "predictions"
METRICS_DIR = ROOT / "outputs" / "metrics"
GEOJSON_PATH = ROOT / "data" / "processed" / "parcels.geojson"
DATASET_PATH = ROOT / "data" / "processed" / "dataset_modeling.csv"


st.set_page_config(page_title="Estres hidrico 7 dias", layout="wide")
st.title("Prediccion de estres hidrico por parcela a 7 dias")


@st.cache_data
def list_prediction_files():
    return sorted(PREDICTIONS_DIR.glob("*_predictions.csv"))


@st.cache_data
def load_predictions(path: Path):
    df = pd.read_csv(path)
    for col in ["fecha", "target_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


@st.cache_data
def load_dataset():
    if DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH)
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        return df
    return pd.DataFrame()


@st.cache_data
def load_metrics(model_name: str):
    path = METRICS_DIR / f"{model_name}_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_geojson():
    if not GEOJSON_PATH.exists():
        return None
    return json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))


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


def draw_map(geojson, latest_predictions):
    fig = go.Figure()
    centers = []
    pred_lookup = latest_predictions.set_index("nombre_parcela")["y_pred"].to_dict()
    for feature in geojson.get("features", []):
        name = feature.get("properties", {}).get("nombre_parcela", "")
        coords = feature.get("geometry", {}).get("coordinates", [[]])[0]
        if not coords:
            continue
        color = stress_color(pred_lookup.get(name))
        lon = [p[0] for p in coords]
        lat = [p[1] for p in coords]
        fig.add_trace(
            go.Scattermapbox(
                lon=lon,
                lat=lat,
                mode="lines",
                fill="toself",
                fillcolor=color,
                line={"color": "#111827", "width": 1},
                name=name,
                text=[f"{name}: {pred_lookup.get(name, float('nan')):.3f}" if name in pred_lookup else name] * len(lon),
                hoverinfo="text",
                showlegend=False,
            )
        )
        centers.append(polygon_center(coords))
    if centers:
        center_lon = sum(c[0] for c in centers) / len(centers)
        center_lat = sum(c[1] for c in centers) / len(centers)
    else:
        center_lon, center_lat = -60.7, -21.84
    fig.update_layout(
        mapbox={
            "style": "open-street-map",
            "center": {"lon": center_lon, "lat": center_lat},
            "zoom": 11,
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True)


prediction_files = list_prediction_files()
if not prediction_files:
    st.info("No hay predicciones guardadas. Ejecuta primero python scripts/train_all_models.py.")
    st.stop()

model_labels = [p.name.replace("_predictions.csv", "") for p in prediction_files]
selected_model = st.sidebar.selectbox("Modelo", model_labels)
pred_path = PREDICTIONS_DIR / f"{selected_model}_predictions.csv"
predictions = load_predictions(pred_path)
metrics = load_metrics(selected_model)
dataset = load_dataset()

latest = predictions.sort_values("fecha").groupby("nombre_parcela", as_index=False).tail(1)

left, right = st.columns([1.5, 1])
with left:
    geojson = load_geojson()
    if geojson:
        draw_map(geojson, latest)
    else:
        st.warning("No se encontro data/processed/parcels.geojson. Se muestra tabla de ultimas predicciones.")
        st.dataframe(latest, use_container_width=True)

with right:
    st.subheader("Metricas")
    metric_cols = st.columns(3)
    metric_cols[0].metric("MAE", f"{metrics.get('mae', float('nan')):.4f}" if metrics.get("mae") is not None else "NA")
    metric_cols[1].metric("RMSE", f"{metrics.get('rmse', float('nan')):.4f}" if metrics.get("rmse") is not None else "NA")
    metric_cols[2].metric("R2", f"{metrics.get('r2', float('nan')):.4f}" if metrics.get("r2") is not None else "NA")
    if metrics.get("status") and metrics.get("status") != "ok":
        st.warning(f"Estado: {metrics.get('status')}. {metrics.get('notes', '')}")

    parcelas = sorted(predictions["nombre_parcela"].dropna().unique())
    selected_parcel = st.selectbox("Parcela", parcelas)
    parcel_pred = predictions[predictions["nombre_parcela"] == selected_parcel].sort_values("fecha")
    parcel_real = dataset[dataset["nombre_parcela"] == selected_parcel].sort_values("fecha") if not dataset.empty else pd.DataFrame()
    fig = go.Figure()
    if not parcel_real.empty and "stress_index" in parcel_real:
        fig.add_trace(go.Scatter(x=parcel_real["fecha"], y=parcel_real["stress_index"], mode="lines", name="Real historico"))
    fig.add_trace(go.Scatter(x=parcel_pred["target_date"], y=parcel_pred["y_pred"], mode="lines+markers", name="Prediccion"))
    fig.add_trace(go.Scatter(x=parcel_pred["target_date"], y=parcel_pred["y_true"], mode="markers", name="Real evaluado"))
    fig.update_layout(height=330, margin={"l": 0, "r": 0, "t": 10, "b": 0}, yaxis_title="stress_index")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Ultimas predicciones")
st.dataframe(
    latest.sort_values("y_pred", ascending=False).head(50),
    use_container_width=True,
    hide_index=True,
)
