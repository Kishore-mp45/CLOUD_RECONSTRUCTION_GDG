"""Live Sentinel scene retrieval through the official Earth Engine Python API."""

from __future__ import annotations

import json
import re
import uuid
from datetime import date, timedelta
from pathlib import Path

import rasterio
import requests
from sqlalchemy.orm import Session

from api.db.models import Scene
from cloudremoval.config import get_settings

S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
S1_BANDS = ["VV", "VH"]


class LiveFetchError(RuntimeError):
    """A recoverable live-data error that can be shown directly to the user."""


def _ee():
    try:
        import ee
    except ImportError as exc:  # pragma: no cover - setup guard
        raise LiveFetchError("Earth Engine client is not installed. Run `uv sync`.") from exc
    settings = get_settings()
    if not settings.EARTH_ENGINE_PROJECT:
        raise LiveFetchError("Earth Engine is not configured. Set EARTH_ENGINE_PROJECT in .env and authenticate.")
    try:
        ee.Initialize(project=settings.EARTH_ENGINE_PROJECT)
    except Exception as exc:
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            raise LiveFetchError(
                "Google sign-in was approved, but Python cannot trust your Windows network certificate. "
                "Install the Windows certificate bridge with `uv add --system-certs pip-system-certs`, then restart the API."
            ) from exc
        raise LiveFetchError(
            "Earth Engine is not authenticated. Run `uv run python scripts/authenticate_earth_engine.py` and approve the Google sign-in."
        ) from exc
    return ee


def resolve_location(location: str) -> tuple[float, float, str]:
    """Accept `lat, lon` or resolve a place name through OpenStreetMap Nominatim."""
    match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*", location)
    if match:
        lat, lon = map(float, match.groups())
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise LiveFetchError("Coordinates are outside the valid latitude/longitude range.")
        return lat, lon, f"{lat:.5f}, {lon:.5f}"

    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": location, "format": "jsonv2", "limit": 1},
            headers={"User-Agent": "ClearView-cloud-removal-demo/1.0"},
            timeout=15,
        )
        response.raise_for_status()
        matches = response.json()
    except requests.RequestException as exc:
        raise LiveFetchError("Could not look up that place. Please enter coordinates as latitude, longitude.") from exc
    if not matches:
        raise LiveFetchError("Place not found. Try a more specific name or enter latitude, longitude.")
    item = matches[0]
    return float(item["lat"]), float(item["lon"]), item["display_name"]


def _download(image, bands: list[str], region, destination: Path, scale: int = 10) -> None:
    """Download one multiband GeoTIFF and validate the expected band count."""
    try:
        url = image.select(bands).getDownloadURL(
            {"name": destination.stem, "region": region, "scale": scale, "format": "GEO_TIFF"}
        )
        response = requests.get(url, timeout=180)
        response.raise_for_status()
        destination.write_bytes(response.content)
        with rasterio.open(destination) as src:
            if src.count != len(bands):
                raise LiveFetchError(f"Download had {src.count} bands; expected {len(bands)}.")
    except LiveFetchError:
        destination.unlink(missing_ok=True)
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        detail = str(exc)
        if "CERTIFICATE_VERIFY_FAILED" in detail:
            raise LiveFetchError(
                "Earth Engine found the imagery, but Python cannot trust your Windows network certificate. "
                "Install `pip-system-certs` and restart the API."
            ) from exc
        raise LiveFetchError(f"Earth Engine could not download the selected imagery as a GeoTIFF: {detail}") from exc


def fetch_live_scene(db: Session, location: str, acquisition_date: date) -> Scene:
    """Fetch a cloud-containing Sentinel-2 and nearest VV/VH Sentinel-1 scene."""
    ee = _ee()
    lat, lon, location_label = resolve_location(location)
    start = acquisition_date - timedelta(days=14)
    end = acquisition_date + timedelta(days=15)
    target_millis = int(__import__("datetime").datetime.combine(acquisition_date, __import__("datetime").time.min).timestamp() * 1000)
    region = ee.Geometry.Rectangle([lon - 0.012, lat - 0.012, lon + 0.012, lat + 0.012])

    s2_collection = (
        # The model was trained on 13-band Sentinel-2 TOA inputs.  The SR
        # collection omits B10, while this harmonized TOA collection provides
        # the complete B1..B12/B8A band set required by the inference pipeline.
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(region)
        .filterDate(start.isoformat(), end.isoformat())
        .filter(ee.Filter.gte("CLOUDY_PIXEL_PERCENTAGE", 15))
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 95))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    if s2_collection.size().getInfo() == 0:
        raise LiveFetchError("No suitably cloudy Sentinel-2 image was found within 14 days of that date.")
    s2 = ee.Image(s2_collection.first())
    s2_info = s2.getInfo()
    cloud_cover = float(s2_info.get("properties", {}).get("CLOUDY_PIXEL_PERCENTAGE", 0.0))

    def add_difference(image):
        return image.set("date_difference", image.date().millis().subtract(target_millis).abs())

    s1_collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(start.isoformat(), end.isoformat())
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .map(add_difference)
        .sort("date_difference")
    )
    if s1_collection.size().getInfo() == 0:
        raise LiveFetchError("No nearby Sentinel-1 VV/VH radar pass was found within 14 days of that date.")
    s1 = ee.Image(s1_collection.first())

    scene_id = f"live_{acquisition_date:%Y%m%d}_{uuid.uuid4().hex[:8]}"
    output_dir = get_settings().OUTPUT_DIR / get_settings().EARTH_ENGINE_OUTPUT_SUBDIR / scene_id
    output_dir.mkdir(parents=True, exist_ok=False)
    s2_path, s1_path = output_dir / "cloudy_s2.tif", output_dir / "sar_s1.tif"
    _download(s2, S2_BANDS, region, s2_path)
    _download(s1, S1_BANDS, region, s1_path)

    with rasterio.open(s2_path) as src:
        crs = str(src.crs) if src.crs else "EPSG:4326"
        width, height = src.width, src.height
        resolution = float(abs(src.transform.a))
        bounds = list(src.bounds)

    scene = Scene(
        scene_id=scene_id,
        external_scene_id=str(s2_info.get("id", "")),
        roi_id=location_label[:64],
        acquisition_time=str(s2_info.get("properties", {}).get("system:time_start", acquisition_date.isoformat())),
        source_provider="Google Earth Engine",
        s2_path=str(s2_path), s1_path=str(s1_path), target_path=None,
        cloud_density_percent=cloud_cover, cloud_probability_threshold=0.0, is_eligible=True,
        crs=crs, width=width, height=height, resolution=resolution,
        bounds_json=json.dumps(bounds),
        extra_metadata=json.dumps({"source_type": "live", "latitude": lat, "longitude": lon, "requested_date": acquisition_date.isoformat()}),
    )
    db.add(scene)
    db.commit()
    db.refresh(scene)
    return scene
