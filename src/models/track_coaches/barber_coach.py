"""
Barber-specific track coach implementation.
"""
from __future__ import annotations

from typing import Dict

from src.models.track_coach import TrackCoach, TrackCoachConfig


class BarberCoach(TrackCoach):
    """Specialized coach for Barber Motorsports Park."""

    def get_local_track_notes(self) -> Dict:
        dna = self._get_track_dna()
        return {
            "track_highlights": [
                "Flowing technical track with elevation changes.",
                "Sequence-heavy S turns punish sloppy rhythm.",
            ],
            "setup_focus": [
                "Stable front end for rapid direction changes.",
                "Traction control tuned for flowing exits.",
            ],
            "risk_zones": [
                "T2-T3 complex: easy to over-slow entry.",
                "Last sector elevation drop: brake release timing crucial.",
            ],
            "complexity_score": dna["technical_complexity"]["complexity_score"],
        }


def create_barber_coach(
    config: TrackCoachConfig,
    data_loader=None,
) -> BarberCoach:
    return BarberCoach(config=config, data_loader=data_loader)

