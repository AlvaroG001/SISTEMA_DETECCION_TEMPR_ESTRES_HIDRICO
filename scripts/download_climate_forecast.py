from pathlib import Path
import argparse
import json
import sys
import time
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PARCELS_GEOJSON_FILE, RAW_DIR, ensure_directories


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARIABLES = [
    "temperature_2m_min",
    "temperature_2m_max",
    "temperature_2m_mean",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "et0_fao_evapotranspiration",
]


def polygon_centroid(coords: list[list[float]]) -> tuple[float, float]:
    lon = sum(point[0] for point in coords) / len(coords)
    lat = sum(point[1] for point in coords) / len(coords)
    return lon, lat


def load_parcel_centroids(include_generated: bool) -> list[dict]:
    if not PARCELS_GEOJSON_FILE.exists():
        raise FileNotFoundError(
            "No se encontró data/processed/parcels.geojson. Ejecuta primero python scripts/prepare_dataset.py."
        )
    geojson = json.loads(PARCELS_GEOJSON_FILE.read_text(encoding="utf-8"))
    parcels = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        name = str(props.get("nombre_parcela", "")).strip()
        generated = bool(props.get("generated_name", False))
        if generated and not include_generated:
            continue
        coords = feature.get("geometry", {}).get("coordinates", [[]])[0]
        if not name or len(coords) < 3:
            continue
        lon, lat = polygon_centroid(coords)
        parcels.append(
            {
                "nombre_parcela": name,
                "latitude": lat,
                "longitude": lon,
                "generated_name": generated,
            }
        )
    return parcels


def fetch_daily_forecast(latitude: float, longitude: float, forecast_days: int) -> dict:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "forecast_days": forecast_days,
        "timezone": "auto",
        "daily": ",".join(DAILY_VARIABLES),
    }
    url = f"{OPEN_METEO_URL}?{urlencode(params)}"
    with urlopen(url, timeout=30) as response:
        data = json.load(response)
    if "daily" not in data:
        raise RuntimeError(f"Open-Meteo no devolvió bloque daily: {data}")
    return data


def rows_for_parcel(parcel: dict, forecast_days: int) -> list[dict]:
    data = fetch_daily_forecast(parcel["latitude"], parcel["longitude"], forecast_days)
    daily = data["daily"]
    rows = []
    for idx, date in enumerate(daily["time"]):
        rows.append(
            {
                "forecast_date": date,
                "nombre_parcela": parcel["nombre_parcela"],
                "latitude": parcel["latitude"],
                "longitude": parcel["longitude"],
                "temp_min_c": daily["temperature_2m_min"][idx],
                "temp_max_c": daily["temperature_2m_max"][idx],
                "temp_mean_c": daily["temperature_2m_mean"][idx],
                "precipitation_mm": daily["precipitation_sum"][idx],
                "humidity_pct": daily["relative_humidity_2m_mean"][idx],
                "wind_m_s": daily["wind_speed_10m_mean"][idx],
                "et0_mm": daily["et0_fao_evapotranspiration"][idx],
                "source": "Open-Meteo Forecast API",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga predicción climática diaria por parcela desde Open-Meteo.")
    parser.add_argument("--forecast-days", type=int, default=7, help="Número de días de predicción a descargar.")
    parser.add_argument(
        "--include-generated",
        action="store_true",
        help="Incluye parcelas del GeoJSON con nombre generado porque venían sin nombre en el Excel.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.15,
        help="Pausa entre llamadas para no saturar la API pública.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RAW_DIR / "climate_forecast.csv",
        help="Ruta del CSV de salida.",
    )
    args = parser.parse_args()

    ensure_directories()
    parcels = load_parcel_centroids(include_generated=args.include_generated)
    if not parcels:
        raise RuntimeError("No hay parcelas válidas para descargar predicción climática.")

    all_rows = []
    failures = []
    for idx, parcel in enumerate(parcels, start=1):
        try:
            print(f"[{idx}/{len(parcels)}] Descargando clima para {parcel['nombre_parcela']}...")
            all_rows.extend(rows_for_parcel(parcel, args.forecast_days))
        except Exception as exc:
            failures.append({"nombre_parcela": parcel["nombre_parcela"], "error": str(exc)})
        time.sleep(args.sleep_seconds)

    if not all_rows:
        raise RuntimeError(f"No se pudo descargar ninguna predicción climática. Errores: {failures}")

    df = pd.DataFrame(all_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"CSV climático guardado: {args.output} ({len(df)} filas)")
    if failures:
        failure_path = args.output.with_suffix(".errors.json")
        failure_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Algunas parcelas fallaron. Detalle: {failure_path}")


if __name__ == "__main__":
    main()
