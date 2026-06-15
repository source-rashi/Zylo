"""LightGBM-based premium risk scoring model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import joblib
from lightgbm import LGBMClassifier


@dataclass
class RiskModel:
	"""Wrapper around a LightGBM classifier used for risk scoring."""

	model: LGBMClassifier = field(
		default_factory=lambda: LGBMClassifier(
			n_estimators=120,
			learning_rate=0.08,
			max_depth=4,
			subsample=0.9,
			colsample_bytree=0.85,
			random_state=42,
			class_weight="balanced",
		)
	)
	is_trained: bool = False
	last_trained_at: Optional[str] = None

	def train(self, X: Sequence[Sequence[float]] | np.ndarray, y: Sequence[int] | np.ndarray) -> "RiskModel":
		"""Train the underlying classifier on feature vectors and labels."""

		features = np.asarray(X, dtype=np.float32)
		targets = np.asarray(y, dtype=np.int32)
		self.model.fit(features, targets)
		self.is_trained = True
		self.last_trained_at = np.datetime64("now").astype(str)
		return self

	def predict_risk_score(self, feature_vector: Sequence[float] | np.ndarray) -> float:
		"""Return a normalized risk score between 0.0 and 1.0."""

		if not self.is_trained:
			raise RuntimeError("RiskModel must be trained before calling predict_risk_score().")

		features = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
		probabilities = self.model.predict_proba(features)
		positive_probability = float(probabilities[0][1] if probabilities.shape[1] > 1 else probabilities[0][0])
		return float(np.clip(positive_probability, 0.0, 1.0))

	@staticmethod
	def predict_premium(risk_score: float, base_premium: float) -> float:
		"""Convert a risk score into a weekly INR premium amount."""

		normalized_score = float(np.clip(risk_score, 0.0, 1.0))
		premium_multiplier = 1.0 + (normalized_score * 1.35)
		weekly_premium = float(base_premium) * premium_multiplier
		return float(np.round(np.clip(weekly_premium, base_premium, base_premium * 3.0), 2))

	def save(self, model_path: str | Path) -> Path:
		"""Persist the trained model to disk."""

		if not self.is_trained:
			raise RuntimeError("RiskModel must be trained before calling save().")

		path = Path(model_path)
		path.parent.mkdir(parents=True, exist_ok=True)
		joblib.dump(
			{
				"model": self.model,
				"is_trained": self.is_trained,
				"last_trained_at": self.last_trained_at,
			},
			path,
		)
		return path

	@classmethod
	def load(cls, model_path: str | Path) -> "RiskModel":
		"""Restore a model from disk."""

		payload = joblib.load(Path(model_path))
		instance = cls(model=payload["model"])
		instance.is_trained = bool(payload.get("is_trained", False))
		instance.last_trained_at = payload.get("last_trained_at")
		return instance


def build_risk_model() -> RiskModel:
	"""Factory for the default risk scoring model."""

	return RiskModel()


def load_risk_model(model_path: str | Path) -> RiskModel:
	"""Convenience loader for persisted risk models."""

	return RiskModel.load(model_path)
