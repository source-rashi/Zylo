"""Validators used by the fraud detection engine."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from PIL import Image


def _as_mapping(photo_exif_data: Any) -> Mapping[str, Any]:
	if isinstance(photo_exif_data, Mapping):
		return photo_exif_data
	return {}


def _parse_timestamp(value: Any) -> Optional[datetime]:
	if value is None:
		return None
	if isinstance(value, datetime):
		return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
	if isinstance(value, (int, float)):
		return datetime.fromtimestamp(float(value), tz=timezone.utc)

	text_value = str(value).strip()
	for pattern in (
		"%Y:%m:%d %H:%M:%S",
		"%Y-%m-%d %H:%M:%S",
		"%Y-%m-%dT%H:%M:%S",
		"%Y-%m-%dT%H:%M:%S%z",
	):
		try:
			parsed = datetime.strptime(text_value, pattern)
			return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
		except ValueError:
			continue
	return None


def _decimal_from_dms(value: Any) -> Optional[float]:
	if value is None:
		return None
	if isinstance(value, (int, float)):
		return float(value)
	if isinstance(value, (tuple, list)) and len(value) == 3:
		degrees, minutes, seconds = value
		return float(degrees) + (float(minutes) / 60.0) + (float(seconds) / 3600.0)
	return None


def _coerce_gps_pair(value: Any) -> Optional[Tuple[float, float]]:
	if value is None:
		return None
	if isinstance(value, (tuple, list)) and len(value) >= 2:
		latitude = _decimal_from_dms(value[0])
		longitude = _decimal_from_dms(value[1])
		if latitude is None or longitude is None:
			return None
		return float(latitude), float(longitude)
	if isinstance(value, Mapping):
		latitude = value.get("lat") or value.get("latitude")
		longitude = value.get("lng") or value.get("lon") or value.get("longitude")
		if latitude is None or longitude is None:
			return None
		return float(latitude), float(longitude)
	return None


def extract_exif_timestamp_and_gps(photo_exif_data: Any) -> dict[str, Any]:
	"""Extract a normalized capture timestamp and GPS coordinates from EXIF data.

	The function supports already-parsed EXIF dictionaries, PIL image objects,
	and common key names used by metadata extraction libraries.
	"""

	metadata = _as_mapping(photo_exif_data)
	timestamp = _parse_timestamp(
		metadata.get("DateTimeOriginal")
		or metadata.get("datetime_original")
		or metadata.get("timestamp")
		or metadata.get("capture_time")
	)
	gps_coords = _coerce_gps_pair(
		metadata.get("GPSCoordinates")
		or metadata.get("gps_coords")
		or metadata.get("gps")
		or metadata.get("location")
	)

	if timestamp is None and hasattr(photo_exif_data, "getexif"):
		try:
			exif = photo_exif_data.getexif()
			if exif:
				raw_timestamp = exif.get(36867) or exif.get(306)
				timestamp = _parse_timestamp(raw_timestamp)
		except Exception:
			pass

	if gps_coords is None and hasattr(photo_exif_data, "getexif"):
		try:
			exif = photo_exif_data.getexif()
			if exif:
				gps_info = exif.get(34853)
				if isinstance(gps_info, Mapping):
					latitude = gps_info.get("GPSLatitude")
					longitude = gps_info.get("GPSLongitude")
					gps_coords = _coerce_gps_pair((latitude, longitude))
		except Exception:
			pass

	return {
		"timestamp": timestamp,
		"gps_coords": gps_coords,
	}
