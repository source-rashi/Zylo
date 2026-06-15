"""Manual claim validation pipeline backed by the fraud detector."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from ml.claim_validator.schemas import ClaimSubmission, ClaimValidationResult
from ml.fraud_engine.detector import FraudDetector, FraudResult, build_fraud_detector


_ZONE_COORDINATES: dict[str, tuple[float, float]] = {
    "zone_01": (12.9716, 77.5946),
    "zone_02": (12.9352, 77.6245),
    "zone_03": (12.9784, 77.6408),
    "zone_04": (12.9279, 77.6271),
    "zone_05": (12.9141, 77.6762),
    "zone_06": (12.9952, 77.6965),
    "zone_07": (12.9609, 77.6387),
    "zone_08": (12.8898, 77.6390),
}

_FRAUD_DETECTOR: FraudDetector | None = None


def _zone_coordinates(zone_id: str) -> tuple[float, float]:
    return _ZONE_COORDINATES.get(str(zone_id).strip().lower(), (12.9716, 77.5946))


def _build_feature_vector(submission: ClaimSubmission) -> np.ndarray:
    latitude, longitude = submission.gps_coords
    return np.asarray(
        [
            len(str(submission.zone_id)) / 16.0,
            submission.submitted_at.hour / 23.0,
            (latitude + 90.0) / 180.0,
            (longitude + 180.0) / 360.0,
        ],
        dtype=np.float32,
    )


def _bootstrap_detector() -> FraudDetector:
    rng = np.random.default_rng(42)
    feature_rows = rng.normal(loc=0.5, scale=0.08, size=(96, 4)).clip(0.0, 1.0)
    detector = build_fraud_detector()
    detector.train(feature_rows)
    return detector


def get_fraud_detector() -> FraudDetector:
    """Lazily build and cache the detector used by claim validation."""

    global _FRAUD_DETECTOR
    if _FRAUD_DETECTOR is None:
        _FRAUD_DETECTOR = _bootstrap_detector()
    return _FRAUD_DETECTOR


def _rejection_reason(result: FraudResult) -> str:
    if not result.flags:
        return "Claim rejected by AI validation."
    return f"Claim rejected by AI validation: {', '.join(result.flags)}"


def validate_claim_submission(submission: ClaimSubmission) -> ClaimValidationResult:
    """Run the fraud detector pipeline for a claim submission."""

    detector = get_fraud_detector()
    claim_payload = submission.model_dump()
    claim_payload["zone_gps_coords"] = _zone_coordinates(submission.zone_id)
    claim_payload["feature_vector"] = _build_feature_vector(submission).tolist()
    claim_payload["expected_timezone"] = "IST"

    fraud_result = detector.score_claim(claim_payload)
    rejection_reason = _rejection_reason(fraud_result) if fraud_result.decision == "auto_reject" else None

    return ClaimValidationResult(
        decision=fraud_result.decision,
        spam_score=fraud_result.spam_score,
        flags=fraud_result.flags,
        rejection_reason=rejection_reason,
        manual_review_queue=None,
        fraud_metadata={
            "zone_id": submission.zone_id,
            "submitted_at": submission.submitted_at.isoformat(),
            "zone_gps_coords": _zone_coordinates(submission.zone_id),
        },
    )
