"""Synthetic dataset generation utilities for ML training.

This module produces pandas DataFrames for rider profiles, weather history,
traffic history, and historical payouts. The generated data is intentionally
correlated so it can be used as a lightweight training seed for the phase 2
ML pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


TIME_SLOTS = ("morning", "afternoon", "night")
TRIGGER_TYPES = ("weather", "traffic", "platform", "manual_review")
DEFAULT_ZONE_IDS = tuple(f"zone_{index:02d}" for index in range(1, 9))
IST = timezone(timedelta(hours=5, minutes=30))


@dataclass(frozen=True)
class GeneratorConfig:
    """Config values used across the synthetic data generators."""

    seed: int = 42
    rider_count: int = 5_000
    weather_rows: int = 25_000
    traffic_rows: int = 25_000
    payout_rows: int = 20_000


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _zone_weights(zone_ids: Iterable[str]) -> np.ndarray:
    zone_ids = tuple(zone_ids)
    weights = np.linspace(1.2, 0.8, num=len(zone_ids))
    return weights / weights.sum()


def _random_timestamps(rng: np.random.Generator, rows: int, days_back: int = 180) -> pd.DatetimeIndex:
    end = datetime.now(tz=IST)
    offsets = rng.integers(0, days_back * 24 * 60, size=rows)
    stamps = [end - timedelta(minutes=int(offset)) for offset in offsets]
    return pd.to_datetime(stamps)


def generate_rider_profiles(
    n_riders: int = 5_000,
    zone_ids: Optional[Iterable[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate rider profile data."""

    zone_ids = tuple(zone_ids or DEFAULT_ZONE_IDS)
    rng = _rng(seed)
    weights = _zone_weights(zone_ids)

    time_slot_preferences = rng.choice(TIME_SLOTS, size=n_riders, p=[0.42, 0.38, 0.20])
    zone_choices = rng.choice(zone_ids, size=n_riders, p=weights)

    days_active = rng.integers(7, 365 * 3, size=n_riders)
    avg_orders_per_day = np.clip(
        rng.normal(loc=18, scale=6, size=n_riders),
        a_min=2,
        a_max=45,
    ).round(1)

    df = pd.DataFrame(
        {
            "rider_id": [f"rider_{index:06d}" for index in range(1, n_riders + 1)],
            "zone_id": zone_choices,
            "time_slot": time_slot_preferences,
            "days_active": days_active,
            "avg_orders_per_day": avg_orders_per_day,
        }
    )
    return df


def generate_weather_history(
    rows: int = 25_000,
    zone_ids: Optional[Iterable[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic weather history with monsoon-like skew."""

    zone_ids = tuple(zone_ids or DEFAULT_ZONE_IDS)
    rng = _rng(seed + 1)
    zone_choices = rng.choice(zone_ids, size=rows)
    timestamps = _random_timestamps(rng, rows)

    seasonality = np.where(timestamps.month.isin([6, 7, 8, 9]), 1.7, 1.0)
    rainfall_mm = np.clip(rng.gamma(shape=1.6, scale=6.0, size=rows) * seasonality, 0, 180).round(1)
    wind_speed = np.clip(rng.normal(loc=12, scale=5, size=rows) + rainfall_mm * 0.04, 0, 65).round(1)
    visibility_km = np.clip(12 - rainfall_mm * 0.055 - wind_speed * 0.03 + rng.normal(0, 0.8, rows), 0.2, 15).round(1)

    return pd.DataFrame(
        {
            "zone_id": zone_choices,
            "timestamp": timestamps,
            "rainfall_mm": rainfall_mm,
            "wind_speed": wind_speed,
            "visibility_km": visibility_km,
        }
    )


def generate_traffic_history(
    rows: int = 25_000,
    zone_ids: Optional[Iterable[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate traffic history with incident clustering during peak hours."""

    zone_ids = tuple(zone_ids or DEFAULT_ZONE_IDS)
    rng = _rng(seed + 2)
    zone_choices = rng.choice(zone_ids, size=rows)
    timestamps = _random_timestamps(rng, rows)

    hour = timestamps.hour
    peak_multiplier = np.where(((hour >= 8) & (hour <= 11)) | ((hour >= 17) & (hour <= 21)), 1.35, 1.0)
    base_traffic = np.clip(rng.normal(loc=52, scale=16, size=rows) * peak_multiplier, 0, 100)
    incident_probability = np.clip((base_traffic / 140) + np.where(peak_multiplier > 1.0, 0.08, 0.02), 0, 0.95)
    incident_flag = (rng.random(rows) < incident_probability).astype(int)

    traffic_index = np.clip(base_traffic + incident_flag * rng.uniform(8, 24, size=rows), 0, 100).round(1)

    return pd.DataFrame(
        {
            "zone_id": zone_choices,
            "timestamp": timestamps,
            "traffic_index": traffic_index,
            "incident_flag": incident_flag,
        }
    )


def generate_historical_payouts(
    rows: int = 20_000,
    rider_profiles: Optional[pd.DataFrame] = None,
    zone_ids: Optional[Iterable[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate historical payout records with a small fraud tail."""

    zone_ids = tuple(zone_ids or DEFAULT_ZONE_IDS)
    rng = _rng(seed + 3)
    timestamps = _random_timestamps(rng, rows)

    if rider_profiles is None:
        rider_profiles = generate_rider_profiles(n_riders=max(rows // 2, 1_000), zone_ids=zone_ids, seed=seed)

    rider_sample = rider_profiles.sample(n=rows, replace=True, random_state=seed).reset_index(drop=True)
    trigger_type = rng.choice(TRIGGER_TYPES, size=rows, p=[0.42, 0.31, 0.16, 0.11])

    base_amount = np.select(
        [trigger_type == "weather", trigger_type == "traffic", trigger_type == "platform", trigger_type == "manual_review"],
        [120.0, 90.0, 75.0, 60.0],
        default=80.0,
    )
    tenure_boost = np.clip(rider_sample["days_active"].to_numpy() / 365, 0.2, 3.0)
    payout_amount = np.clip(base_amount * (0.7 + tenure_boost * 0.45) + rng.normal(0, 12, size=rows), 25, 500).round(2)

    fraud_score = (
        (payout_amount > 220).astype(float) * 0.35
        + np.where(trigger_type == "manual_review", 0.18, 0.0)
        + np.where(rider_sample["days_active"].to_numpy() < 30, 0.12, 0.0)
        + rng.random(rows) * 0.25
    )
    was_fraud = (fraud_score > 0.55).astype(bool)

    return pd.DataFrame(
        {
            "rider_id": rider_sample["rider_id"].to_numpy(),
            "zone_id": rider_sample["zone_id"].to_numpy(),
            "timestamp": timestamps,
            "trigger_type": trigger_type,
            "payout_amount": payout_amount,
            "was_fraud": was_fraud,
        }
    )


def generate_training_datasets(config: Optional[GeneratorConfig] = None) -> Dict[str, pd.DataFrame]:
    """Generate the full synthetic dataset bundle."""

    config = config or GeneratorConfig()
    rider_profiles = generate_rider_profiles(
        n_riders=config.rider_count,
        seed=config.seed,
    )
    weather_history = generate_weather_history(
        rows=config.weather_rows,
        seed=config.seed,
    )
    traffic_history = generate_traffic_history(
        rows=config.traffic_rows,
        seed=config.seed,
    )
    payouts = generate_historical_payouts(
        rows=config.payout_rows,
        rider_profiles=rider_profiles,
        seed=config.seed,
    )

    return {
        "rider_profiles": rider_profiles,
        "weather_history": weather_history,
        "traffic_history": traffic_history,
        "historical_payouts": payouts,
    }
