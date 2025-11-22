"""
Indianapolis-specific track coach implementation.
"""
from __future__ import annotations

from typing import Dict

from src.models.track_coach import TrackCoach, TrackCoachConfig


class IndianapolisCoach(TrackCoach):
    """Specialized coach for Indianapolis Motor Speedway road course."""

    def get_local_track_notes(self) -> Dict:
        dna = self._get_track_dna()
        return {
            "track_highlights": [
                "Combines oval banking with stop-and-go infield.",
                "Tight hairpins require maximizing rotation without understeer.",
            ],
            "setup_focus": [
                "Brake stability for heavy stops (T1, T7).",
                "Rear grip for abrupt direction changes.",
            ],
            "risk_zones": [
                "T1 braking from the oval straight — easy to overcook.",
                "Transition back onto oval: watch rear traction.",
            ],
            "complexity_score": dna["technical_complexity"]["complexity_score"],
        }


def create_indianapolis_coach(
    config: TrackCoachConfig,
    data_loader=None,
) -> IndianapolisCoach:
    return IndianapolisCoach(config=config, data_loader=data_loader)

