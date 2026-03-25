"""
Extract 512×512 pre-crop tiles from European grassland wide-angle drone images.

Source data
-----------
Quattrini et al. (2025) "Standardised Drone Procedures for Phytosociological
Data Collection." Applied Vegetation Science, 28:e70032.
https://doi.org/10.1111/avsc.70032

Platform: DJI Mavic 3 Enterprise (M3E) with RTK positioning.
Camera: 20 MP wide-angle Hasselblad (4/3-inch CMOS sensor), nadir orientation.
Flight height: 5 m above canopy.
Image dimensions: 5280×3956 px (wide-angle mode).
GSD at 5 m: ~2 mm (estimated from sensor geometry; paper reports 0.3 mm for
    the 7× telephoto at the same altitude, so wide-angle is ~7× coarser).
Coverage per image: ~10.6 m × 7.9 m footprint.

Sites: Two Natura 2000 grassland sites in Marche, central Italy.
  - Mount Conero (SAC IT5320007): 6 plots, Bromopsis erecta xeric grassland.
  - Mount Valmontagnana (SAC Gola di Frasassi, IT5320003): 4 plots, same.
  - 10 grassland plots total, each ~22 m² circular area.
  - Plant associations: Convolvulo elegantissimi–Brometum erecti (Conero),
    Asperulo purpureae–Brometum erecti (Valmontagnana).

Each plot folder originally contained 8 images: 1 wide-angle overview + 7
telephoto (7× zoom, 4000×3000, oblique angles). Only the wide-angle overview
is used here. Telephoto images have been deleted from the source directory.

Preprocessing
-------------
  1. No border crop (minor lens vignette accepted at tile edges).
  2. Resize from ~2 mm to 7 mm GSD (scale factor 2/7 ≈ 0.286) to match
     MPG Ranch drone imagery resolution.
     Resized dimensions: ~1508 × 1130 px.
  3. Tile with 50% overlap (stride 256 px) into 512×512 pre-crops.
     Yields ~12 tiles per image (4 cols × 3 rows), ~120 tiles total.

Output: data/images/european_grassland/tiles/ + offsite_manifest.csv
"""

import argparse
import os

import pandas as pd
from PIL import Image

from patch.utils.config import PRECROP_SIZE

SOURCE_DIR = "data/images/european_grassland"
OUTPUT_DIR = "data/images/european_grassland/tiles"

WIDE_ANGLE_SIZE = (5280, 3956)
SOURCE_GSD_MM = 2
TARGET_GSD_MM = 7
SCALE_FACTOR = SOURCE_GSD_MM / TARGET_GSD_MM  # 2/7

STRIDE = PRECROP_SIZE // 2  # 256px = 50% overlap


def _find_wide_angle_images(source_dir: str) -> list[dict]:
    """Find wide-angle images (5280×3956) in each plot folder."""
    images = []
    for root, dirs, files in os.walk(source_dir):
        if "tiles" in root:
            continue
        for f in sorted(files):
            if not f.upper().endswith((".JPG", ".JPEG")):
                continue
            path = os.path.join(root, f)
            img = Image.open(path)
            if img.size == WIDE_ANGLE_SIZE:
                rel = os.path.relpath(root, source_dir).replace("\\", "/")
                parts = rel.split("/")
                site = parts[0] if len(parts) > 0 else "unknown"
                plot = parts[-1] if len(parts) > 1 else "unknown"
                images.append({
                    "path": path,
                    "site": site,
                    "plot": plot,
                    "filename": f,
                })
    return images


def extract_tiles(source_dir: str = SOURCE_DIR, output_dir: str = OUTPUT_DIR):
    """Process wide-angle images and tile with 50% overlap."""
    os.makedirs(output_dir, exist_ok=True)

    images = _find_wide_angle_images(source_dir)
    print(f"Found {len(images)} wide-angle images")

    records = []
    tile_idx = 0

    for img_info in images:
        img = Image.open(img_info["path"]).convert("RGB")

        # Resize to target GSD (no border crop)
        w, h = img.size
        new_w = int(w * SCALE_FACTOR)
        new_h = int(h * SCALE_FACTOR)
        resized = img.resize((new_w, new_h), Image.LANCZOS)

        # Tile with 50% overlap
        n_cols = (new_w - PRECROP_SIZE) // STRIDE + 1
        n_rows = (new_h - PRECROP_SIZE) // STRIDE + 1

        for ri in range(n_rows):
            for ci in range(n_cols):
                x0 = ci * STRIDE
                y0 = ri * STRIDE

                # Ensure tile fits within image
                if x0 + PRECROP_SIZE > new_w or y0 + PRECROP_SIZE > new_h:
                    continue

                tile = resized.crop((x0, y0, x0 + PRECROP_SIZE, y0 + PRECROP_SIZE))

                tile_name = f"offsite_{tile_idx:06d}.png"
                tile_path = os.path.join(output_dir, tile_name)
                tile.save(tile_path)

                records.append({
                    "tile_path": tile_path,
                    "tile_type": "offsite",
                    "month": "none",
                    "color": "none",
                    "concentration": "none",
                    "spray_size_m": 0.0,
                    "latitude": 0.0,
                    "longitude": 0.0,
                    "point_name": f"offsite_{tile_idx:06d}",
                    "source_image": img_info["filename"],
                    "site": img_info["site"],
                    "plot": img_info["plot"],
                })
                tile_idx += 1

        print(f"  {img_info['site']}/{img_info['plot']}: "
              f"{n_cols}×{n_rows} = {n_cols * n_rows} tiles "
              f"(resized {new_w}×{new_h}, stride {STRIDE})")

    manifest = pd.DataFrame(records)
    manifest_path = os.path.join(output_dir, "offsite_manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"\nTotal: {len(manifest)} tiles from {len(images)} images → {manifest_path}")

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract offsite grassland tiles")
    parser.add_argument("--source-dir", default=SOURCE_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    extract_tiles(args.source_dir, args.output_dir)
