"""
Extract 512x512 pre-crop tiles from orthomosaics.

Two tile types:
  1. Spray-centered: centred on GPS points (300 pts x 3 months = 900 tiles)
  2. Eastern block: from 0.1ha unsprayed area east of experiment
"""

import argparse
import os

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.windows import Window

from patch.utils.config import GSD_M, MONTHS, PRECROP_SIZE, TILE_METERS

# Paths relative to project root
SCHEMA_DIR = "data/images/mpg_ranch"
ORTHO_DIR = "data/images/mpg_ranch/orthophotos"
OUTPUT_DIR = "data/images/mpg_ranch/tiles"

# UTM Zone 11N (Montana)
CRS_SRC = 4326  # WGS84
CRS_DST = 32611  # UTM 11N


def extract_spray_tiles(month: str, output_dir: str = OUTPUT_DIR):
    """Extract 512x512 tiles centred on spray GPS points for one month."""
    schema_path = os.path.join(SCHEMA_DIR, f"{month}.csv")
    ortho_path = os.path.join(ORTHO_DIR, f"{month}.tif")
    tile_dir = os.path.join(output_dir, month, "sprayed")
    os.makedirs(tile_dir, exist_ok=True)

    df = pd.read_csv(schema_path)
    transformer = Transformer.from_crs(CRS_SRC, CRS_DST, always_xy=True)

    # Convert lat/lon to UTM
    df["easting"], df["northing"] = transformer.transform(
        df["longitude"].values, df["latitude"].values
    )

    half_m = TILE_METERS / 2  # 1.792 m

    records = []
    with rasterio.open(ortho_path) as src:
        for _, row in df.iterrows():
            easting = row["easting"]
            northing = row["northing"]

            # Bounding box in UTM
            e_min, e_max = easting - half_m, easting + half_m
            n_min, n_max = northing - half_m, northing + half_m

            # Convert to pixel indices (Y inverted: northing_max -> row_min)
            row_min, col_min = src.index(e_min, n_max)
            row_max, col_max = src.index(e_max, n_min)

            # Enforce exact 512x512
            height = row_max - row_min
            width = col_max - col_min
            if height != PRECROP_SIZE or width != PRECROP_SIZE:
                # Adjust to centre on the GPS point
                center_row, center_col = src.index(easting, northing)
                row_min = center_row - PRECROP_SIZE // 2
                col_min = center_col - PRECROP_SIZE // 2
                row_max = row_min + PRECROP_SIZE
                col_max = col_min + PRECROP_SIZE

            window = Window.from_slices((row_min, row_max), (col_min, col_max))
            crop = src.read(window=window)  # (bands, H, W)

            if crop.shape[1] != PRECROP_SIZE or crop.shape[2] != PRECROP_SIZE:
                print(f"Skipping {row['name']}: got shape {crop.shape}")
                continue

            # Take first 3 bands (RGB)
            crop = np.moveaxis(crop[:3], 0, -1)  # (H, W, 3)

            # Save tile
            name = row["name"]
            tile_path = os.path.join(tile_dir, f"{name}.png")
            Image.fromarray(crop.astype(np.uint8)).save(tile_path)

            # Spray centroid is at the tile centre
            records.append({
                "tile_path": tile_path,
                "tile_type": "sprayed",
                "month": month,
                "color": row.get("color", "none"),
                "concentration": row.get("concentration", "none"),
                "spray_size_m": float(row.get("size", 0.0)),
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "point_name": int(name) if str(name).isdigit() else name,
            })

    return pd.DataFrame(records)


def extract_eastern_block_tiles(month: str, output_dir: str = OUTPUT_DIR):
    """Extract 512x512 tiles from unsprayed eastern block.

    Tiles are laid on a non-overlapping grid across the eastern block area.
    """
    schema_path = os.path.join(SCHEMA_DIR, f"{month}.csv")
    ortho_path = os.path.join(ORTHO_DIR, f"{month}.tif")
    tile_dir = os.path.join(output_dir, month, "eastern_block")
    os.makedirs(tile_dir, exist_ok=True)

    df = pd.read_csv(schema_path)
    transformer = Transformer.from_crs(CRS_SRC, CRS_DST, always_xy=True)
    df["easting"], df["northing"] = transformer.transform(
        df["longitude"].values, df["latitude"].values
    )

    # Eastern block: east of main plots
    max_easting = df["easting"].max()
    max_northing = df["northing"].max()
    min_northing = df["northing"].min()

    # 5m offset from eastern edge of spray grid
    west_limit = max_easting + 5
    ns_range = max_northing - min_northing

    # Tile the block with non-overlapping 3.584m tiles
    n_cols = max(1, int(10 / TILE_METERS))  # ~10m wide strip
    n_rows = max(1, int(ns_range / TILE_METERS))

    records = []
    with rasterio.open(ortho_path) as src:
        tile_idx = 0
        for ci in range(n_cols):
            for ri in range(n_rows):
                center_e = west_limit + (ci + 0.5) * TILE_METERS
                center_n = min_northing + (ri + 0.5) * TILE_METERS

                center_row, center_col = src.index(center_e, center_n)
                r0 = center_row - PRECROP_SIZE // 2
                c0 = center_col - PRECROP_SIZE // 2

                window = Window.from_slices(
                    (r0, r0 + PRECROP_SIZE), (c0, c0 + PRECROP_SIZE)
                )

                try:
                    crop = src.read(window=window)
                except Exception:
                    continue

                if crop.shape[1] != PRECROP_SIZE or crop.shape[2] != PRECROP_SIZE:
                    continue

                crop = np.moveaxis(crop[:3], 0, -1)

                tile_path = os.path.join(tile_dir, f"east_{tile_idx:04d}.png")
                Image.fromarray(crop.astype(np.uint8)).save(tile_path)

                # Compute lat/lon of tile centre
                inv_transformer = Transformer.from_crs(CRS_DST, CRS_SRC, always_xy=True)
                lon, lat = inv_transformer.transform(center_e, center_n)

                records.append({
                    "tile_path": tile_path,
                    "tile_type": "eastern_block",
                    "month": month,
                    "color": "none",
                    "concentration": "none",
                    "spray_size_m": 0.0,
                    "latitude": lat,
                    "longitude": lon,
                    "point_name": f"east_{tile_idx:04d}",
                })
                tile_idx += 1

    return pd.DataFrame(records)


def extract_all(output_dir: str = OUTPUT_DIR):
    """Extract all tiles for all months."""
    all_records = []
    for month in MONTHS:
        print(f"Extracting {month} spray tiles...")
        spray_df = extract_spray_tiles(month, output_dir)
        all_records.append(spray_df)

        print(f"Extracting {month} eastern block tiles...")
        east_df = extract_eastern_block_tiles(month, output_dir)
        all_records.append(east_df)

    manifest = pd.concat(all_records, ignore_index=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"Saved manifest ({len(manifest)} tiles) to {manifest_path}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 512px tiles from orthomosaics")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    extract_all(args.output_dir)
