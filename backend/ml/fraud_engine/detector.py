"""Isolation Forest backed fraud detection engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import IsolationForest

from .validators import (
	extract_exif_timestamp_and_gps,
	validate_gps_delta,
	validate_timezone_consistency,
	validate_velocity_impossibility,
)


@dataclass(frozen=True)
class FraudResult:
	"""Structured output from fraud detection."""

	spam_score: float
	decision: str
	flags: list[str]


@dataclass
class FraudDetector:
	"""Isolation Forest wrapper for claim anomaly detection."""

	model: IsolationForest = field(
		default_factory=lambda: IsolationForest(
			n_estimators=200,
			contamination=0.08,
			random_state=42,
		)
	)
	is_trained: bool = False

	def train(self, feature_rows: Sequence[Sequence[float]] | np.ndarray) -> "FraudDetector":
		"""Fit the anomaly detector on historical feature vectors."""

		features = np.asarray(feature_rows, dtype=np.float32)
		self.model.fit(features)
		self.is_trained = True
		return self

	def anomaly_score(self, feature_vector: Sequence[float] | np.ndarray) -> float:
		"""Return a normalized anomaly score between 0.0 and 1.0."""

		if not self.is_trained:
			raise RuntimeError("FraudDetector must be trained before calling anomaly_score().")

		vector = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
		raw_score = float(self.model.decision_function(vector)[0])
		normalized = 1.0 - ((raw_score + 0.5) / 1.0)
		return float(np.clip(normalized, 0.0, 1.0))

	def score_claim(self, claim: dict[str, Any]) -> FraudResult:
		"""Score a claim dictionary and return a composite fraud result."""

		if not self.is_trained:
			raise RuntimeError("FraudDetector must be trained before calling score_claim().")

		claim_gps_coords = claim.get("gps_coords")
		photo_exif_data = claim.get("photo_exif_data")
		extractions = extract_exif_timestamp_and_gps(photo_exif_data)
		flags: list[str] = []
		score_components = [self.anomaly_score(claim.get("feature_vector", [0.0, 0.0, 0.0, 0.0]))]

		if extractions.get("timestamp") is None:
			flags.append("missing_exif_timestamp")
			score_components.append(0.12)
		else:
			claim_time = claim.get("timestamp")
			if claim_time is not None and isinstance(claim_time, datetime):
				time_delta_minutes = abs((claim_time - extractions["timestamp"]).total_seconds()) / 60.0
				score_components.append(min(time_delta_minutes / 180.0, 0.25))

		if extractions.get("gps_coords") is None:
			flags.append("missing_exif_gps")
			score_components.append(0.12)
		else:
			gps_check = validate_gps_delta(claim_gps_coords, claim.get("zone_gps_coords"), radius_km=claim.get("radius_km", 3.0))
			if not gps_check["within_radius"]:
				flags.append(gps_check["flag"] or "gps_mismatch")
				score_components.append(0.30)

		timezone_check = validate_timezone_consistency(claim.get("device_timezone"), expected_timezone=claim.get("expected_timezone", "IST"))
		if not timezone_check["consistent"]:
			flags.append(timezone_check["flag"] or "timezone_mismatch")
			score_components.append(0.18)

		velocity_check = validate_velocity_impossibility(
			claim.get("previous_zone_gps_coords"),
			claim.get("current_zone_gps_coords"),
			claim.get("previous_timestamp"),
			claim.get("timestamp"),
			max_speed_kmph=claim.get("max_speed_kmph", 80.0),
		)
		if velocity_check["impossible"]:
			flags.append(velocity_check["flag"] or "velocity_impossible")
			score_components.append(0.32)

		spam_score = float(np.clip(sum(score_components) / len(score_components), 0.0, 1.0))
		decision = self._decision_from_score(spam_score)
		return FraudResult(spam_score=spam_score, decision=decision, flags=sorted(set(flags)))

	@staticmethod
	def _decision_from_score(spam_score: float) -> str:
		if spam_score < 0.3:
			return "auto_approve"
		if spam_score <= 0.7:
			return "review"
		return "auto_reject"


def build_fraud_detector() -> FraudDetector:
	"""Factory for the default fraud detector."""

	return FraudDetector()
