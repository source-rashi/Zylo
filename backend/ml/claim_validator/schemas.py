"""Schemas for the claim validation pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Sequence
from uuid import UUID

from pydantic import BaseModel, Field


class ClaimSubmission(BaseModel):
    """Input payload for AI-assisted claim validation."""

    rider_id: UUID
    zone_id: str
    photo_base64: str
    gps_coords: tuple[float, float]
    submitted_at: datetime
    photo_exif_data: Optional[dict[str, Any]] = None
    device_timezone: Optional[str] = None
    previous_zone_gps_coords: Optional[tuple[float, float]] = None
    previous_timestamp: Optional[datetime] = None


class ClaimValidationResult(BaseModel):
    """Structured result returned from the validation pipeline."""

    decision: str = Field(default="review")
    spam_score: float
    flags: list[str] = Field(default_factory=list)
    rejection_reason: Optional[str] = None
    manual_review_queue: Optional[str] = None
    fraud_metadata: Optional[dict[str, Any]] = None
