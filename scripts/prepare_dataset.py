from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PARCELS_GEOJSON_FILE, PROCESSED_DATA_FILE, ensure_directories
from src.data_loading import read_raw_dataset
from src.feature_engineering import build_modeling_dataset
from src.geo_utils import excel_to_geojson
from src.temporal_split import add_temporal_split


def main() -> None:
    ensure_directories()
    raw = read_raw_dataset()
    dataset = build_modeling_dataset(raw)
    dataset = add_temporal_split(dataset)
    dataset.to_csv(PROCESSED_DATA_FILE, index=False)
    geo_ok = excel_to_geojson()
    print(f"Saved processed dataset: {PROCESSED_DATA_FILE} ({len(dataset)} rows)")
    if geo_ok:
        print(f"Saved parcel GeoJSON: {PARCELS_GEOJSON_FILE}")
    else:
        print("Could not create parcels.geojson. See data/processed/parcels_geojson_warning.txt")


if __name__ == "__main__":
    main()
