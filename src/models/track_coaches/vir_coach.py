"""
VIR-specific track coach implementation.
"""
from __future__ import annotations

from typing import Dict

from src.models.track_coach import TrackCoach, TrackCoachConfig


class VIRCoach(TrackCoach):
    """Specialized coach for Virginia International Raceway."""

    def get_local_track_notes(self) -> Dict:
        dna = self._get_track_dna()
        return {
            "track_highlights": [
                "Fast, flowing sections like the Climbing Esses.",
                "High-risk zones combined with grass runoff.",
            ],
            "setup_focus": [
                "High-speed stability; avoid bottoming through esses.",
                "Predictable rear during curb strikes.",
            ],
            "risk_zones": [
                "Uphill Esses: one mistake ends in big time loss.",
                "Roller Coaster: downhill braking over crest.",
            ],
            "complexity_score": dna["technical_complexity"]["complexity_score"],
        }


def create_vir_coach(
    config: TrackCoachConfig,
    data_loader=None,
) -> VIRCoach:
    return VIRCoach(config=config, data_loader=data_loader)

