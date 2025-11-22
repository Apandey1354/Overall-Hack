"""
Track Coach base class
Phase 3.1: Provides reusable coaching utilities for track-specific AI coaches.

Responsibilities:
- Build a knowledge base per track (Track DNA + historical data)
- Provide sector-by-sector recommendations
- Generate racing line and weather adaptation tips
- Deliver driver-specific advice using embeddings and performance data
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data_processing.data_loader import DataLoader
from src.data_processing.track_dna_extractor import TrackDNAExtractor
from src.data_processing.driver_embedder import create_driver_embeddings


@dataclass
class TrackCoachConfig:
    """Configuration for TrackCoach."""

    track_id: str
    driver_embeddings_df: Optional[pd.DataFrame] = None
    track_dna_df: Optional[pd.DataFrame] = None
    weather_window_minutes: int = 30


class TrackCoach:
    """
    Base class for track-specific coaches.

    Usage:
        coach = TrackCoach(TrackCoachConfig(track_id="barber_Race 1"))
        overview = coach.get_track_overview()
        sector_tips = coach.get_sector_recommendations(driver_number=13)
        weather_strategy = coach.get_weather_strategy(weather_condition="rain")
        advice = coach.get_driver_advice(driver_number=13)
    """

    def __init__(
        self,
        config: TrackCoachConfig,
        data_loader: Optional[DataLoader] = None,
    ) -> None:
        self.config = config
        self.track_id = config.track_id
        self.venue, self.race = self._parse_track_id(self.track_id)
        self.loader = data_loader or DataLoader()
        self.track_dna_extractor = TrackDNAExtractor(self.loader)

        # Lazy caches
        self._track_dna_cache: Optional[Dict] = None
        self._driver_embeddings_cache: Optional[pd.DataFrame] = config.driver_embeddings_df
        self._knowledge_base_cache: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_track_overview(self) -> Dict:
        """High-level summary of the track characteristics."""
        dna = self._get_track_dna()
        knowledge = self._build_knowledge_base()

        overview = {
            "track_id": self.track_id,
            "venue": self.venue,
            "race": self.race,
            "complexity_score": dna["technical_complexity"]["complexity_score"],
            "track_length_km": dna["physical_characteristics"]["track_length"].get(
                "estimated_length_km", 0
            ),
            "num_sectors": dna["physical_characteristics"]["num_sectors"],
            "dominant_features": knowledge["dominant_features"],
            "recommended_focus": knowledge["recommended_focus"],
        }
        overview["local_notes"] = self.get_local_track_notes()
        return overview

    def get_sector_recommendations(
        self, driver_number: Optional[int] = None
    ) -> List[Dict]:
        """
        Provide sector-by-sector tips.

        If driver_number is provided, compare driver consistency vs. track demands.
        """
        dna = self._get_track_dna()
        sectors = dna["technical_complexity"]["sector_time_variance"]
        recommendations = []

        driver_profile = None
        if driver_number is not None:
            driver_profile = self._get_driver_profile(driver_number)

        for sector_name, stats in sectors.items():
            std_seconds = stats.get("std_seconds")
            if std_seconds is None and "variance" in stats:
                std_seconds = float(np.sqrt(max(stats["variance"], 0)))
            cv = stats.get("cv", 0)

            if std_seconds is None:
                focus = "balanced"
            elif stats.get("variance", 0) > 50.0:
                focus = "stability"
            elif stats.get("variance", 0) >= 5.0:
                focus = "precision"
            else:
                focus = "attack speed"

            tip = {
                "sector": sector_name,
                "variance": stats.get("variance", 0),
                "cv": cv,
                "std_seconds": std_seconds,
                "focus": focus,
            }
            if driver_profile:
                sector_consistency = driver_profile["consistency_score"]
                variance = stats.get("variance", 0)
                if std_seconds is not None and variance > 0:
                    if variance > 50.0:
                        tip["driver_tip"] = (
                            "Critical inconsistency—review braking points and track limits; likely a major mistake in this sector."
                        )
                    elif variance >= 5.0:
                        tip["driver_tip"] = (
                            "Trim entry speed and smooth brake release to cut the spread between laps."
                        )
                    else:
                        tip["driver_tip"] = (
                            "Consistency is excellent—carry more apex speed and attack exit throttle earlier."
                        )
                elif std_seconds is not None:
                    tip["driver_tip"] = (
                        "Good baseline—focus on matching brake points to gain 1–2 km/h more through mid-corner."
                    )
                elif sector_consistency < 0.5:
                    tip["driver_tip"] = (
                        "Stabilize braking inputs before turn-in to reduce variance."
                    )
                else:
                    tip["driver_tip"] = (
                        "Consistency looks solid. Explore later braking reference."
                    )
            recommendations.append(tip)
        return recommendations

    def get_weather_strategy(self, weather_condition: str = "default") -> Dict:
        """
        Suggest weather adaptation strategy.
        weather_condition: "rain", "hot", "cold", "default"
        """
        dna = self._get_track_dna()
        base_strategy = {
            "condition": weather_condition,
            "focus": "balanced",
            "notes": [],
        }

        if weather_condition.lower() == "rain":
            base_strategy["focus"] = "traction"
            base_strategy["notes"].append("Prioritize smooth throttle and early upshifts.")
            base_strategy["notes"].append(
                "Use defensive lines through sectors with high variance."
            )
        elif weather_condition.lower() == "hot":
            base_strategy["focus"] = "tire management"
            base_strategy["notes"].append("Reduce sliding in Sector 2 to control temps.")
        elif weather_condition.lower() == "cold":
            base_strategy["focus"] = "warm-up"
            base_strategy["notes"].append("Build temperature on straights before push lap.")
        else:
            base_strategy["notes"].append(
                "Baseline setup: optimize balance between corner entry stability and rotation."
            )

        # Track-specific nuance
        if dna["physical_characteristics"]["num_sectors"] > 3:
            base_strategy["notes"].append(
                "Longer track: monitor delta per sector to catch time losses early."
            )

        return base_strategy

    def get_driver_advice(self, driver_number: int) -> Dict:
        """Provide driver-specific recommendations."""
        profile = self._get_driver_profile(driver_number)
        dna = self._get_track_dna()

        advice = {
            "driver_number": driver_number,
            "strengths": profile["strengths"],
            "weaknesses": profile["weaknesses"],
            "focus": [],
        }

        complexity = dna["technical_complexity"]["complexity_score"]
        speed_ratio = dna["speed_profile"]["straight_corner_ratio"].get("ratio", 1.0)
        sector_variances = dna["technical_complexity"].get("sector_time_variance", {})

        if complexity > 0.6 and "technical" not in profile["strengths"]:
            advice["focus"].append(
                "Build confidence in trail braking zones; focus on rotation without overslowing."
            )
        if speed_ratio > 1.2 and "speed-focused" not in profile["strengths"]:
            advice["focus"].append(
                "Work on maximizing exit speed; use transfer model predictions for optimal gears."
            )
        if not advice["focus"]:
            advice["focus"].append(
                "Maintain current approach; use telemetry overlays to refine braking markers."
            )
        high_variance_sectors = [
            name for name, stats in sector_variances.items()
            if isinstance(stats, dict) and stats.get("std_seconds", 0) >= 1.0
        ][:2]
        if high_variance_sectors:
            advice["focus"].append(
                f"High spread in {', '.join(high_variance_sectors)}—review data there to tame the swings."
            )
        return advice

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_track_id(self, track_id: str) -> (str, str):
        if "_" not in track_id:
            raise ValueError(
                f"Invalid track_id '{track_id}'. Expected format 'venue_Race X'."
            )
        venue, race = track_id.split("_", 1)
        return venue, race

    def _get_track_dna(self) -> Dict:
        if self._track_dna_cache is None:
            if (
                self.config.track_dna_df is not None
                and "track_id" in self.config.track_dna_df.columns
            ):
                row = self.config.track_dna_df[
                    self.config.track_dna_df["track_id"] == self.track_id
                ]
                if not row.empty:
                    self._track_dna_cache = row.iloc[0].to_dict()
            if self._track_dna_cache is None:
                self._track_dna_cache = self.track_dna_extractor.extract_track_dna(
                    self.venue, self.race
                )
        return self._track_dna_cache

    def _build_knowledge_base(self) -> Dict:
        if self._knowledge_base_cache is not None:
            return self._knowledge_base_cache

        dna = self._get_track_dna()
        complexity = dna["technical_complexity"]["complexity_score"]
        speed_ratio = dna["speed_profile"]["straight_corner_ratio"].get("ratio", 1.0)
        num_sectors = dna["physical_characteristics"].get("num_sectors", 0)

        dominant = []
        focus = []
        if complexity >= 0.55:
            dominant.append("technical")
            focus.append("Link apexes and manage rotation through tighter corners.")
        elif complexity <= 0.35:
            dominant.append("low-complexity")
            focus.append("Push braking references; track rewards aggression.")

        if speed_ratio >= 1.15:
            dominant.append("high-speed")
            focus.append("Maximize exit speed and straighten steering before throttle.")
        elif speed_ratio <= 0.9:
            dominant.append("corner-heavy")
            focus.append("Prioritize mid-corner balance and rotate the car early.")

        if num_sectors >= 5:
            dominant.append("flowing")
            focus.append("Maintain rhythm over long linked sectors; avoid micro-mistakes.")

        if not dominant:
            dominant.append("balanced")
            focus.append("Consistency across all sectors.")

        self._knowledge_base_cache = {
            "dominant_features": dominant,
            "recommended_focus": focus,
        }
        return self._knowledge_base_cache

    def _get_driver_embeddings(self) -> pd.DataFrame:
        if self._driver_embeddings_cache is None:
            self._driver_embeddings_cache = create_driver_embeddings(self.loader)
        return self._driver_embeddings_cache

    def _get_driver_profile(self, driver_number: int) -> Dict:
        df = self._get_driver_embeddings()
        row = df[df["driver_number"] == driver_number]
        if row.empty:
            raise ValueError(f"No embedding found for driver #{driver_number}")

        entry = row.iloc[0]
        return {
            "technical_score": entry.get("technical_proficiency", 0.0),
            "speed_score": entry.get("high_speed_proficiency", 0.0),
            "consistency_score": entry.get("consistency_score", 0.0),
            "strengths": entry.get("strengths", "").split(", ") if entry.get("strengths") else [],
            "weaknesses": entry.get("full_embedding", {})
            .get("track_specific_strengths", {})
            .get("weaknesses", []),
        }

    def get_local_track_notes(self) -> Dict:
        """
        Override in subclasses to provide track-specific knowledge.
        Default is empty.
        """
        return {
            "track_highlights": [],
            "setup_focus": [],
            "risk_zones": [],
            "complexity_score": self._get_track_dna()["technical_complexity"]["complexity_score"],
        }


def create_track_coach(
    track_id: str,
    driver_embeddings_df: Optional[pd.DataFrame] = None,
    track_dna_df: Optional[pd.DataFrame] = None,
    data_loader: Optional[DataLoader] = None,
) -> TrackCoach:
    """Factory helper to instantiate a TrackCoach."""
    config = TrackCoachConfig(
        track_id=track_id,
        driver_embeddings_df=driver_embeddings_df,
        track_dna_df=track_dna_df,
    )
    return TrackCoach(config=config, data_loader=data_loader)

