"""
COTA-specific track coach implementation.
"""
from __future__ import annotations

from typing import Dict

from src.models.track_coach import TrackCoach, TrackCoachConfig


class COTACoach(TrackCoach):
    """Specialized coach for Circuit of the Americas."""

    def get_local_track_notes(self) -> Dict:
        dna = self._get_track_dna()
        return {
            "track_highlights": [
                "Large elevation changes, especially T1 climb.",
                "First sector mimics Maggotts/Becketts rhythm.",
                "Long back straight requires optimal exit speed.",
            ],
            "setup_focus": [
                "Firm platform to handle aggressive chicanes.",
                "Aerodynamic efficiency for long straights.",
            ],
            "risk_zones": [
                "T1 uphill braking mis-judgment leads to deep lockups.",
                "Esses: missing the first apex cascades through entire S1.",
            ],
            "complexity_score": dna["technical_complexity"]["complexity_score"],
        }


def create_cota_coach(
    config: TrackCoachConfig,
    data_loader=None,
) -> COTACoach:
    return COTACoach(config=config, data_loader=data_loader)

