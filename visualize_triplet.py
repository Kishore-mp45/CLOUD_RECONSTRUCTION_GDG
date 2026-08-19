"""Create a contrast-stretched cloudy / target / SAR GeoTIFF PNG preview.

The script discovers RGB by GeoTIFF band descriptions (B4/B3/B2), rather than
assuming that the first three bands are RGB. It can select a labelled triplet
directly from the supplied AllClear JSON metadata manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def percentile_stretch(image: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    """Stretch each channel independently to [0, 1], ignoring NaN/inf values."""
    result = np.empty_like(image, dtype=np.float32)
    for channel in range(image.shape[0]):
        values = image[channel]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            result[channel] = 0
            continue
        lo, hi = np.percentile(finite, (low, high))
        result[channel] = 0 if hi <= lo else np.clip((values - lo) / (hi - lo), 0, 1)
    return result


def optical_rgb(path: Path) -> tuple[np.ndarray, str]:
    with rasterio.open(path) as src:
        descriptions = [d.upper() if d else "" for d in src.descriptions]
        required = ("B4", "B3", "B2")
        if not all(name in descriptions for name in required):
            raise ValueError(f"{path} has descriptions {src.descriptions}; cannot identify B4/B3/B2 RGB bands")
        indexes = [descriptions.index(name) + 1 for name in required]
        data = src.read(indexes).astype(np.float32)
    return np.moveaxis(percentile_stretch(data), 0, -1), f"RGB: {required} (bands {indexes})"


def sar_display(path: Path) -> tuple[np.ndarray, str]:
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        descriptions = tuple(d or f"band {i}" for i, d in enumerate(src.descriptions, start=1))
    stretched = percentile_stretch(data)
    if stretched.shape[0] >= 2:
        # VV/VH false colour; both channels are independently stretched.
        image = np.stack((stretched[0], stretched[1], stretched[1]), axis=-1)
        return image, f"False colour: {descriptions[0]} / {descriptions[1]}"
    return stretched[0], f"Grayscale: {descriptions[0]}"


def local_path(dataset_root: Path, metadata_path: str) -> Path:
    """Map an absolute dataset path recorded on the source machine to this checkout."""
    parts = Path(metadata_path.replace("\\", "/")).parts
    # Metadata paths end roi/year_month/{s1,s2_toa}/filename.tif.
    return dataset_root.joinpath(*parts[-4:])


def triplet_from_metadata(metadata: Path, dataset_root: Path, sample_key: str | None) -> tuple[Path, Path, Path, str]:
    with metadata.open(encoding="utf-8") as handle:
        samples = json.load(handle)
    candidates = [(sample_key, samples[sample_key])] if sample_key else samples.items()
    for key, sample in candidates:
        if not sample["s1"]:
            continue
        cloudy = local_path(dataset_root, sample["s2_toa"][0][1])
        target = local_path(dataset_root, sample["target"][0][1])
        sar = local_path(dataset_root, sample["s1"][0][1])
        if all(path.is_file() for path in (cloudy, target, sar)):
            return cloudy, target, sar, key
    message = f"No complete local input/target/SAR triplet found for {sample_key or 'any metadata sample'}"
    raise FileNotFoundError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloudy", type=Path, help="Cloudy Sentinel-2 GeoTIFF")
    parser.add_argument("--sar", type=Path, help="Matched Sentinel-1 GeoTIFF")
    parser.add_argument("--target", type=Path, help="Matched cloud-free Sentinel-2 target GeoTIFF")
    parser.add_argument("--metadata", type=Path, help="AllClear JSON manifest; selects a verified triplet")
    parser.add_argument("--dataset-root", type=Path, help="Local root used with --metadata")
    parser.add_argument("--sample-key", help="Optional exact metadata key; defaults to first complete local triplet")
    parser.add_argument("--output", type=Path, default=Path("triplet_preview.png"))
    args = parser.parse_args()

    if args.metadata:
        if not args.dataset_root:
            parser.error("--metadata requires --dataset-root")
        args.cloudy, args.target, args.sar, selected_key = triplet_from_metadata(
            args.metadata, args.dataset_root, args.sample_key
        )
        print(f"Selected metadata sample: {selected_key}")
        print(f"Input S2: {args.cloudy}\nTarget S2: {args.target}\nS1: {args.sar}")
    elif not all((args.cloudy, args.target, args.sar)):
        parser.error("pass --cloudy, --target, and --sar, or pass --metadata with --dataset-root")

    cloudy, cloudy_note = optical_rgb(args.cloudy)
    sar, sar_note = sar_display(args.sar)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(cloudy)
    axes[0].set_title(f"S2 input / cloudy observation\n{cloudy_note}")
    target, target_note = optical_rgb(args.target)
    axes[1].imshow(target)
    axes[1].set_title(f"Cloud-free S2 target\n{target_note}")
    axes[2].imshow(sar, cmap=None if sar.ndim == 3 else "gray")
    axes[2].set_title(f"Sentinel-1 SAR\n{sar_note}")
    for axis in axes:
        axis.set_axis_off()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Saved {args.output.resolve()}")
    print("Assumptions: optical RGB is identified from B4/B3/B2 band descriptions; "
          "display uses per-channel 2nd–98th percentile stretching. "
          "SAR uses per-channel 2nd–98th percentile stretching and VV/VH false colour when available.")


if __name__ == "__main__":
    main()
