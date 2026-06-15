"""ML inference endpoints for premium and claim validation workflows."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ml.claim_validator.schemas import ClaimSubmission, ClaimValidationResult
from ml.claim_validator.validator import validate_claim_submission
from ml.risk_model.features import build_feature_vector
from ml.risk_model.model import train_on_synthetic_data

router = APIRouter()

_RISK_MODEL = None


class PremiumCalculationRequest(BaseModel):
    """Request body for premium calculation."""

    zone_id: str
    time_slot: str
    rider_id: str
    rider_profile: dict[str, object] = Field(default_factory=dict)
    base_premium: float = 250.0


@router.post("/ml/premium/calculate")
async def calculate_premium(request: PremiumCalculationRequest) -> dict[str, object]:
    """Calculate weekly premium using the risk model."""

    zone_id = request.zone_id
    time_slot = request.time_slot
    rider_id = request.rider_id
    rider_profile = request.rider_profile
    base_premium = float(request.base_premium)

    feature_vector = build_feature_vector(zone_id, time_slot, rider_profile)

    global _RISK_MODEL
    if _RISK_MODEL is None:
        _RISK_MODEL = train_on_synthetic_data(sample_size=512, seed=42)

    risk_score = _RISK_MODEL.predict_risk_score(feature_vector)
    weekly_premium_inr = _RISK_MODEL.predict_premium(risk_score, base_premium)

    if risk_score < 0.35:
        zone_risk_label = "low"
    elif risk_score < 0.7:
        zone_risk_label = "medium"
    else:
        zone_risk_label = "high"

    return {
        "weekly_premium_inr": weekly_premium_inr,
        "risk_score": risk_score,
        "zone_risk_label": zone_risk_label,
        "rider_id": rider_id,
        "zone_id": zone_id,
        "time_slot": time_slot,
        "base_premium": base_premium,
    }


@router.post("/ml/claim/validate", response_model=ClaimValidationResult)
async def validate_claim(submission: ClaimSubmission) -> ClaimValidationResult:
    """Validate a manual claim submission using the fraud engine pipeline."""

    return await validate_claim_submission(submission)


@router.get("/ml/health")
async def ml_health() -> dict[str, object]:
    """Report ML model load status and fraud engine readiness."""

    global _RISK_MODEL
    if _RISK_MODEL is None:
        return {
            "model_loaded": False,
            "last_trained_at": None,
            "fraud_engine_status": "not_loaded",
        }

    return {
        "model_loaded": True,
        "last_trained_at": _RISK_MODEL.last_trained_at,
        "fraud_engine_status": "ready",
    }
