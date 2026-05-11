import json
import re
from pathlib import Path

import pandas as pd

from src.config import GEOJSON_WARNING_FILE, PARCELS_GEOJSON_FILE, RAW_COORDS_FILE


def _parse_coords(value: str) -> list[list[float]]:
    numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(value))]
    coords = []
    for i in range(0, len(numbers) - 2, 3):
        lon, lat, _alt = numbers[i : i + 3]
        coords.append([lon, lat])
    if len(coords) >= 3 and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def excel_to_geojson(
    excel_path: Path = RAW_COORDS_FILE,
    output_path: Path = PARCELS_GEOJSON_FILE,
    warning_path: Path = GEOJSON_WARNING_FILE,
) -> bool:
    try:
        if not excel_path.exists():
            raise FileNotFoundError(f"Coordinate Excel not found: {excel_path}")
        df = pd.read_excel(excel_path)
        name_col = next((c for c in df.columns if str(c).strip().lower() in {"nombre", "parcela", "nombre_parcela"}), None)
        coord_col = next((c for c in df.columns if "coord" in str(c).strip().lower()), None)
        if coord_col is None:
            raise ValueError("No se encontró una columna de coordenadas en el Excel.")
        features = []
        generated_counter = 1
        for idx, row in df.iterrows():
            coords = _parse_coords(row.get(coord_col, ""))
            if len(coords) < 4:
                continue
            raw_name = str(row.get(name_col, "")).strip() if name_col else ""
            if raw_name:
                name = raw_name
            else:
                name = f"parcela_excel_{generated_counter:03d}"
                generated_counter += 1
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "nombre_parcela": name,
                        "excel_row": int(idx) + 2,
                        "generated_name": not bool(raw_name),
                    },
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                }
            )
        if not features:
            raise ValueError("No se pudo extraer ningún polígono válido del Excel.")
        geojson = {"type": "FeatureCollection", "features": features}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)
        if warning_path.exists():
            warning_path.unlink()
        return True
    except Exception as exc:
        warning_path.parent.mkdir(parents=True, exist_ok=True)
        warning_path.write_text(
            "No se pudo generar parcels.geojson. El resto del proyecto puede ejecutarse.\n"
            f"Motivo: {exc}\n",
            encoding="utf-8",
        )
        return False
