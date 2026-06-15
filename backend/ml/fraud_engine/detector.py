"""Isolation Forest backed fraud detection engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from sklearn.ensemble import IsolationForest


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


def build_fraud_detector() -> FraudDetector:
	"""Factory for the default fraud detector."""

	return FraudDetector()
