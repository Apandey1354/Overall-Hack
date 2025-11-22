"""
Utility to build cached JSON files for the Phase 6 React dashboard.

Generates:
- data/cache/track_dna_summary.json
- data/cache/championship_state.json
- data/cache/track_coach_data.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.cache_utils import generate_cache_bundle


def main():
    generate_cache_bundle()
    print("✔ Dashboard cache updated in data/cache/")


if __name__ == "__main__":
    main()


