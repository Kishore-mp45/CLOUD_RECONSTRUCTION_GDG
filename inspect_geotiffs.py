"""Inspect optical/SAR GeoTIFFs in an AllClear-style dataset.

Examples
--------
python inspect_geotiffs.py --dataset-root allclear_dataset
python inspect_geotiffs.py --cloudy path/to/cloudy.tif --target path/to/target.tif --sar path/to/sar.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio


def describe(path: Path, role: str) -> None:
    """Print the metadata and finite per-band range for one GeoTIFF."""
    with rasterio.open(path) as src:
        print(f"\n{role}: {path}")
        print(f"  bands: {src.count}")
        print(f"  size: {src.width} x {src.height}")
        print(f"  dtype(s): {', '.join(src.dtypes)}")
        print(f"  CRS: {src.crs}")
        print(f"  band descriptions: {src.descriptions}")
        for band in src.indexes:
            pixels = src.read(band, masked=True).compressed()
            finite = pixels[np.isfinite(pixels)]
            lo, hi = (float('nan'), float('nan')) if finite.size == 0 else (finite.min(), finite.max())
            print(f"  band {band:2d} ({src.descriptions[band - 1] or 'unnamed'}): min={lo:g}, max={hi:g}")


def find_samples(root: Path) -> tuple[Path, Path]:
    s2 = next(root.glob("roi*/**/s2_toa/*.tif"), None)
    s1 = next(root.glob("roi*/**/s1/*.tif"), None)
    if not s2 or not s1:
        raise FileNotFoundError("Could not find both roi*/**/s2_toa/*.tif and roi*/**/s1/*.tif")
    return s2, s1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, help="AllClear dataset root; finds one S2 and one S1 sample")
    parser.add_argument("--cloudy", type=Path, help="Cloudy optical GeoTIFF")
    parser.add_argument("--target", type=Path, help="Cloud-free optical target GeoTIFF")
    parser.add_argument("--sar", type=Path, help="Sentinel-1 SAR GeoTIFF")
    args = parser.parse_args()

    if args.dataset_root:
        s2, s1 = find_samples(args.dataset_root)
        describe(s2, "Optical S2 TOA sample (cloudy/target role is not encoded in this layout)")
        describe(s1, "SAR S1 sample")
        print("\nNo cloud-free target directory was inferred from folder names. "
              "Use --target or resolve roles through the AllClear metadata manifest.")
    else:
        if not all((args.cloudy, args.target, args.sar)):
            parser.error("pass --dataset-root or all of --cloudy, --target, and --sar")
        describe(args.cloudy, "Cloudy optical")
        describe(args.target, "Cloud-free optical target")
        describe(args.sar, "SAR")


if __name__ == "__main__":
    main()
