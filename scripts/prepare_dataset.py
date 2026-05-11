from pathlib import Path
import argparse
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DEFAULT_HORIZON_DAYS, PARCELS_GEOJSON_FILE, PROCESSED_DATA_FILE, ensure_directories, processed_data_file
from src.data_loading import read_raw_dataset
from src.feature_engineering import build_modeling_dataset
from src.geo_utils import excel_to_geojson
from src.temporal_split import add_temporal_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara el dataset de modelado para un horizonte objetivo.")
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    args = parser.parse_args()

    ensure_directories()
    raw = read_raw_dataset()
    dataset = build_modeling_dataset(raw, horizon_days=args.horizon_days)
    dataset = add_temporal_split(dataset)
    output_path = processed_data_file(args.horizon_days)
    dataset.to_csv(output_path, index=False)
    if args.horizon_days == DEFAULT_HORIZON_DAYS:
        shutil.copyfile(output_path, PROCESSED_DATA_FILE)
    geo_ok = excel_to_geojson()
    print(f"Dataset procesado guardado: {output_path} ({len(dataset)} filas)")
    if args.horizon_days == DEFAULT_HORIZON_DAYS:
        print(f"Alias de compatibilidad actualizado: {PROCESSED_DATA_FILE}")
    if geo_ok:
        print(f"GeoJSON de parcelas guardado: {PARCELS_GEOJSON_FILE}")
    else:
        print("No se pudo crear parcels.geojson. Revisa data/processed/parcels_geojson_warning.txt")


if __name__ == "__main__":
    main()
