"""
Mechanical Karma stream simulation.

Takes per-lap features and produces smoothed per-component "karma" scores
that mimic real-time risk tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import logging

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    feature_weights: Dict[str, float]
    description: str


COMPONENT_SPECS: List[ComponentSpec] = [
    ComponentSpec(
        name="engine",
        feature_weights={"speed_mean": 0.4, "nmot_mean": 0.6},
        description="RPM + sustained speed stress",
    ),
    ComponentSpec(
        name="gearbox",
        feature_weights={"gear_mean": 0.5, "accx_can_std": 0.5},
        description="Gear usage and longitudinal jolts",
    ),
    ComponentSpec(
        name="brakes",
        feature_weights={"pbrake_f_max": 0.6, "pbrake_r_max": 0.4},
        description="Brake pressure spikes front/rear",
    ),
    ComponentSpec(
        name="tires",
        feature_weights={"speed_mean": 0.3, "Steering_Angle_std": 0.7},
        description="Cornering + abrasion load",
    ),
]


def _column_stats(df: pd.DataFrame, columns: Iterable[str]) -> Dict[str, tuple[float, float]]:
    stats = {}
    for col in columns:
        series = df[col].dropna()
        if series.empty:
            stats[col] = (0.0, 0.0)
        else:
            stats[col] = (float(series.min()), float(series.max()))
    return stats


def _normalize(value: float | int | None, min_val: float, max_val: float) -> float:
    if value is None or pd.isna(value):
        return 0.0
    if max_val <= min_val:
        return 0.0
    return max(0.0, min(1.0, (float(value) - min_val) / (max_val - min_val)))


def _component_score(row: pd.Series, stats: Dict[str, tuple[float, float]], spec: ComponentSpec) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for feature, weight in spec.feature_weights.items():
        min_val, max_val = stats.get(feature, (0.0, 0.0))
        norm_val = _normalize(row.get(feature), min_val, max_val)
        weighted_sum += weight * norm_val
        total_weight += weight
    if total_weight == 0:
        return 0.0
    return weighted_sum / total_weight


def compute_stream(df: pd.DataFrame, *, smoothing: float = 0.6, wear_rate: float = 0.002) -> pd.DataFrame:
    """
    Compute Mechanical Karma scores from an in-memory per-lap dataframe.
    
    Args:
        df: Per-lap feature dataframe with columns: vehicle_id, lap, and required feature columns
        smoothing: EMA smoothing factor (0-1, higher = more smoothing)
        wear_rate: Base wear accumulation per lap (0-1, simulates natural degradation)
    
    Returns:
        DataFrame with columns: vehicle_id, lap, component, instant_score, karma_score
    """

    if df.empty:
        return pd.DataFrame(columns=["vehicle_id", "lap", "component", "instant_score", "karma_score"])

    # Check for required columns with flexible naming
    required_cols = set().union(*(spec.feature_weights.keys() for spec in COMPONENT_SPECS))
    
    # Map column names - handle case variations and alternative names
    column_map = {}
    for col in required_cols:
        # Try exact match first
        if col in df.columns:
            column_map[col] = col
            continue
        # Try case-insensitive match
        for df_col in df.columns:
            if df_col.lower() == col.lower():
                column_map[col] = df_col
                break
        # If still not found, try common variations
        if col not in column_map:
            variations = {
                "speed_mean": ["speed", "Speed", "SPEED", "speed_avg", "avg_speed"],
                "nmot_mean": ["nmot", "Nmot", "NMOT", "rpm", "RPM"],
                "gear_mean": ["gear", "Gear", "GEAR", "gear_avg"],
                "accx_can_std": ["accx_can", "accx", "AccX", "acceleration_x"],
                "pbrake_f_max": ["pbrake_f", "brake_f", "brake_front", "pbrake_front"],
                "pbrake_r_max": ["pbrake_r", "brake_r", "brake_rear", "pbrake_rear"],
                "Steering_Angle_std": ["Steering_Angle", "steering_angle", "steering", "Steering"],
            }
            for var in variations.get(col, []):
                if var in df.columns:
                    column_map[col] = var
                    break
    
    missing = [col for col in required_cols if col not in column_map]
    if missing:
        # Try to compute aggregations if we have base columns
        LOGGER.warning(f"Some required columns missing: {missing}. Attempting to compute from available data.")
        # For now, we'll proceed with what we have and use defaults for missing
    
    # Create a working dataframe with mapped column names
    working_df = df.copy()
    for original_col, mapped_col in column_map.items():
        if mapped_col != original_col and mapped_col in working_df.columns:
            working_df[original_col] = working_df[mapped_col]
    
    # For missing columns, try to use defaults or skip that component
    available_cols = set(working_df.columns)
    stats = _column_stats(working_df, [col for col in required_cols if col in available_cols])

    records: List[dict] = []
    sorted_df = working_df.sort_values(["vehicle_id", "lap"])
    smoothed_scores: Dict[tuple[str, str], float] = {}
    
    # Track starting lap for each vehicle to calculate relative wear
    vehicle_start_laps: Dict[str, int] = {}

    for _, row in sorted_df.iterrows():
        vehicle_id = str(row["vehicle_id"])
        lap = int(row["lap"])
        
        # Track starting lap
        if vehicle_id not in vehicle_start_laps:
            vehicle_start_laps[vehicle_id] = lap
        
        # Calculate relative lap position (for wear accumulation)
        relative_lap = lap - vehicle_start_laps[vehicle_id] + 1
        
        for spec in COMPONENT_SPECS:
            # Check if we have the required features for this component
            component_features = set(spec.feature_weights.keys())
            available_features = component_features.intersection(available_cols)
            
            if not available_features:
                # Skip this component if no features available
                continue
            
            # Adjust weights based on available features
            available_weights = {f: spec.feature_weights[f] for f in available_features}
            total_weight_sum = sum(available_weights.values())
            if total_weight_sum > 0:
                # Normalize weights
                normalized_weights = {f: w / total_weight_sum for f, w in available_weights.items()}
            else:
                normalized_weights = available_weights
            
            # Compute instant score with available features
            instant = 0.0
            total_weight = 0.0
            for feature, weight in normalized_weights.items():
                if feature in stats:
                    min_val, max_val = stats[feature]
                    norm_val = _normalize(row.get(feature), min_val, max_val)
                    instant += weight * norm_val
                    total_weight += weight
            
            if total_weight > 0:
                instant = instant / total_weight
            
            key = (vehicle_id, spec.name)
            prev = smoothed_scores.get(key, instant)
            
            # Apply EMA smoothing to instant stress
            smoothed = smoothing * prev + (1 - smoothing) * instant
            
            # Add cumulative wear that increases over time
            # Wear accumulates based on lap number, simulating natural degradation
            wear_accumulation = min(0.5, wear_rate * relative_lap)  # Cap wear at 0.5
            
            # Combine smoothed stress with accumulated wear
            # Higher instant stress increases wear rate
            stress_multiplier = 1.0 + (instant * 0.5)  # Stress increases wear by up to 50%
            total_wear = wear_accumulation * stress_multiplier
            
            # Final karma is smoothed stress + accumulated wear (capped at 1.0)
            final_karma = min(1.0, smoothed + total_wear)
            
            smoothed_scores[key] = final_karma
            records.append(
                {
                    "vehicle_id": vehicle_id,
                    "lap": lap,
                    "component": spec.name,
                    "instant_score": instant,
                    "karma_score": final_karma,
                }
            )

    return pd.DataFrame(records)


__all__ = ["compute_stream", "COMPONENT_SPECS"]


