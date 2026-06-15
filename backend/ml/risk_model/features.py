"""Feature engineering pipeline for the risk scoring model."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


ZONE_RISK_INDEX = {
	"zone_01": 0.92,
	"zone_02": 0.84,
	"zone_03": 0.76,
	"zone_04": 0.68,
	"zone_05": 0.59,
	"zone_06": 0.52,
	"zone_07": 0.43,
	"zone_08": 0.35,
}

TIME_SLOT_MULTIPLIER = {
	"morning": 0.92,
	"afternoon": 1.00,
	"night": 1.18,
}


def _normalize_zone_id(zone_id: str) -> str:
	return str(zone_id).strip().lower()


def _zone_risk_index(zone_id: str) -> float:
	normalized_zone_id = _normalize_zone_id(zone_id)
	return float(ZONE_RISK_INDEX.get(normalized_zone_id, 0.5))


def _time_slot_multiplier(time_slot: str) -> float:
	normalized_time_slot = str(time_slot).strip().lower()
	return float(TIME_SLOT_MULTIPLIER.get(normalized_time_slot, 1.0))


def _historical_disruption_rate(rider_profile: Mapping[str, Any]) -> float:
	avg_orders_per_day = float(rider_profile.get("avg_orders_per_day", 12.0) or 12.0)
	days_active = float(rider_profile.get("days_active", 30.0) or 30.0)

	disruption_signal = 0.16 + min(avg_orders_per_day / 120.0, 0.28)
	tenure_discount = max(0.72, 1.0 - (days_active / 3650.0))
	return float(np.clip(disruption_signal * tenure_discount, 0.03, 0.48))


def _rider_tenure_factor(rider_profile: Mapping[str, Any]) -> float:
	days_active = float(rider_profile.get("days_active", 30.0) or 30.0)
	rider_tenure_days = float(rider_profile.get("rider_tenure_days", days_active) or days_active)
	normalized_tenure = max(days_active, rider_tenure_days)

	if normalized_tenure < 30:
		return 1.22
	if normalized_tenure < 90:
		return 1.12
	if normalized_tenure < 180:
		return 1.03
	if normalized_tenure < 365:
		return 0.97
	return 0.91


def build_feature_vector(zone_id: str, time_slot: str, rider_profile: Mapping[str, Any]) -> np.ndarray:
	"""Build a numeric feature vector for premium risk scoring."""

	features = np.array(
		[
			_zone_risk_index(zone_id),
			_time_slot_multiplier(time_slot),
			_historical_disruption_rate(rider_profile),
			_rider_tenure_factor(rider_profile),
		],
		dtype=np.float32,
	)
	return features


def build_feature_dict(zone_id: str, time_slot: str, rider_profile: Mapping[str, Any]) -> dict[str, float]:
	"""Return a named feature mapping for inspection and model debugging."""

	return {
		"zone_risk_index": _zone_risk_index(zone_id),
		"time_slot_multiplier": _time_slot_multiplier(time_slot),
		"historical_disruption_rate": _historical_disruption_rate(rider_profile),
		"rider_tenure_factor": _rider_tenure_factor(rider_profile),
	}
