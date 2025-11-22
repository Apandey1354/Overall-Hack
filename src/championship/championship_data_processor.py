"""
Championship Data Integration (Phase 4.1).

Responsibilities:
- Locate and load official championship standings CSVs
- Normalize driver/track identifiers
- Build per-race point tracking tables
- Produce calendar mappings that downstream simulators can consume
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_processing.data_loader import DataLoader

logger = logging.getLogger(__name__)

# Example column: "Barber_Race 1_Points"
POINT_COLUMN_PATTERN = re.compile(r"(?P<event>.+)_Race (?P<race>\d+)_Points", re.IGNORECASE)

# Canonical mapping between free-form venue names (CSV headers) and internal identifiers
TRACK_ALIAS_MAP: Dict[str, str] = {
    "barber": "barber",
    "barber motorsports park": "barber",
    "barber motorsports": "barber",
    "cota": "COTA",
    "circuit of the americas": "COTA",
    "americas": "COTA",
    "indy": "indianapolis",
    "indianapolis": "indianapolis",
    "indianapolis motor speedway": "indianapolis",
    "vir": "virginia-international-raceway",
    "virginia international raceway": "virginia-international-raceway",
    "virginia-international-raceway": "virginia-international-raceway",
    "virginia international": "virginia-international-raceway",
    "sonoma": "sonoma-raceway",
    "sonoma raceway": "sonoma-raceway",
    "sonoma-raceway": "sonoma-raceway",
    "road america": "road-america",
    "road-america": "road-america",
    "sebring": "sebring-international-raceway",
    "sebring-international-raceway": "sebring-international-raceway",
}


def _safe_number(value: object) -> float:
    """Convert championship CSV tokens (including blanks, '/', etc.) into floats."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float, np.number)):
        if pd.isna(value):
            return 0.0
        return float(value)
    if isinstance(value, str):
        token = value.strip()
        if not token or token in {"-", "NS", "DNS", "DNF", "DSQ", "NC", "/", "NA"}:
            return 0.0
        try:
            return float(token)
        except ValueError:
            return 0.0
    return 0.0


def _clean_status(value: object) -> str:
    """Normalize status strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    status = str(value).strip()
    if status in {"", "0"}:
        return ""
    return status


@dataclass
class ChampionshipDataset:
    """Container for fully-processed championship data."""

    standings: pd.DataFrame
    driver_points: pd.DataFrame
    calendar: pd.DataFrame


class ChampionshipDataProcessor:
    """
    Load and normalize championship datasets.

    The processor searches for files that include the word "Championship" under the workspace,
    parses semicolon-delimited standings, and constructs tidy tables suitable for simulation.
    """

    def __init__(
        self,
        data_loader: Optional[DataLoader] = None,
        base_path: Optional[Path] = None,
    ) -> None:
        self.loader = data_loader or DataLoader()
        self.base_path = Path(base_path or Path.cwd()).resolve()

        self._standings_df: Optional[pd.DataFrame] = None
        self._driver_points_df: Optional[pd.DataFrame] = None
        self._calendar_df: Optional[pd.DataFrame] = None
        self._event_order_lookup: Dict[Tuple[str, int], int] = {}
        self._championship_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_championship_standings(self) -> pd.DataFrame:
        """Return the raw championship standings table with normalized driver metadata."""
        if self._standings_df is not None:
            return self._standings_df.copy()

        championship_csv = self._find_championship_file()
        if championship_csv is None:
            raise FileNotFoundError(
                "Unable to locate a championship standings CSV. Expected a file that "
                "contains the word 'Championship' somewhere under the project root."
            )

        logger.info("Loading championship standings from %s", championship_csv)
        encodings = ["utf-8", "latin-1", "cp1252"]
        last_error: Optional[Exception] = None
        df = None
        for enc in encodings:
            try:
                logger.debug("Attempting to read standings with encoding=%s", enc)
                df = pd.read_csv(championship_csv, sep=";", engine="python", encoding=enc)
                break
            except UnicodeDecodeError as exc:
                last_error = exc
                logger.warning("Failed to read %s with encoding %s: %s", championship_csv, enc, exc)
        if df is None:
            raise last_error  # type: ignore[misc]

        rename_map = {
            "Pos": "position",
            "Participant": "participant",
            "Points": "season_points",
            "Number": "driver_number",
            "NAME": "first_name",
            "SURNAME": "last_name",
            "COUNTRY": "country",
            "TEAM": "team",
            "MANUFACTURER": "manufacturer",
            "CLASS": "driver_class",
        }
        for src, dst in rename_map.items():
            if src in df.columns:
                df.rename(columns={src: dst}, inplace=True)

        if "driver_number" in df.columns:
            df["driver_number"] = (
                pd.to_numeric(df["driver_number"], errors="coerce").fillna(0).astype(int)
            )
        else:
            df["driver_number"] = np.arange(1, len(df) + 1)

        if "season_points" in df.columns:
            df["season_points"] = pd.to_numeric(df["season_points"], errors="coerce").fillna(0.0)
        else:
            df["season_points"] = 0.0

        df["driver_name"] = df.apply(
            lambda row: row.get("participant")
            or " ".join(filter(None, [row.get("first_name", ""), row.get("last_name", "")])).strip(),
            axis=1,
        )

        self._standings_df = df
        return df.copy()

    def create_driver_point_tracker(self) -> pd.DataFrame:
        """
        Build a long-format table with one row per driver per race (Phase 4.1 requirement).
        """
        if self._driver_points_df is not None:
            return self._driver_points_df.copy()

        standings = self.load_championship_standings()
        if standings.empty:
            self._driver_points_df = pd.DataFrame()
            return self._driver_points_df.copy()

        records: List[Dict] = []
        expected_columns = [
            "driver_number",
            "driver_name",
            "team",
            "driver_class",
            "event_name",
            "race_number",
            "event_order",
            "venue_key",
            "track_id",
            "base_points",
            "pole_points",
            "fastest_lap_points",
            "extra_points",
            "status",
            "total_points",
        ]
        event_order = 0

        for column in standings.columns:
            match = POINT_COLUMN_PATTERN.match(column)
            if not match:
                continue
            event_name = match.group("event").strip()
            race_number = int(match.group("race"))
            event_key = (event_name, race_number)
            if event_key not in self._event_order_lookup:
                event_order += 1
                self._event_order_lookup[event_key] = event_order

            for _, row in standings.iterrows():
                record = self._build_event_record(
                    row=row,
                    event_name=event_name,
                    race_number=race_number,
                    event_order=self._event_order_lookup[event_key],
                )
                records.append(record)

        driver_points = pd.DataFrame(records)
        if driver_points.empty:
            driver_points = pd.DataFrame(columns=expected_columns)
        if driver_points.empty:
            self._driver_points_df = driver_points
            return driver_points

        driver_points.sort_values(["driver_number", "event_order"], inplace=True)
        driver_points["cumulative_points"] = driver_points.groupby("driver_number")[
            "total_points"
        ].cumsum()

        self._driver_points_df = driver_points
        return driver_points.copy()

    def build_calendar_map(self) -> pd.DataFrame:
        """Return ordered calendar of events (Phase 4.1 requirement)."""
        if self._calendar_df is not None:
            return self._calendar_df.copy()

        if not self._event_order_lookup:
            self.create_driver_point_tracker()

        if not self._event_order_lookup:
            logger.warning(
                "No championship events detected in standings; falling back to loader venues."
            )
            fallback_records = []
            order = 0
            for venue in getattr(self.loader, "venues", []):
                for race_number in [1, 2]:
                    order += 1
                    track_id = f"{venue}_Race {race_number}"
                    fallback_records.append(
                        {
                            "event_name": venue.replace("-", " ").title(),
                            "race_number": race_number,
                            "venue_key": venue,
                            "track_id": track_id,
                            "event_order": order,
                        }
                    )
            self._calendar_df = pd.DataFrame(fallback_records)
            return self._calendar_df.copy()

        # Get configured venues from loader to filter calendar
        # Normalize configured venues for comparison (case-insensitive)
        configured_venues_raw = getattr(self.loader, "venues", [])
        configured_venues_normalized = set()
        for venue in configured_venues_raw:
            # Normalize each configured venue the same way event names are normalized
            normalized = self._normalize_track_name(venue)
            configured_venues_normalized.add(normalized)
        
        if not configured_venues_normalized:
            logger.warning("No configured venues found in loader; including all events from CSV.")
        
        records = []
        filtered_order = 0
        for (event_name, race_number), original_order in sorted(
            self._event_order_lookup.items(), key=lambda item: item[1]
        ):
            venue_key = self._normalize_track_name(event_name)
            
            # Filter to only include configured venues (if venues are configured)
            if configured_venues_normalized and venue_key not in configured_venues_normalized:
                logger.debug(f"Skipping event {event_name} Race {race_number} - venue {venue_key} not in configured venues {configured_venues_normalized}")
                continue
            
            # Only include Race 1 and Race 2 for each venue
            if race_number not in [1, 2]:
                logger.debug(f"Skipping event {event_name} Race {race_number} - only Race 1 and Race 2 are included")
                continue
            
            filtered_order += 1
            track_id = f"{venue_key}_Race {race_number}"
            records.append(
                {
                    "event_name": event_name,
                    "race_number": race_number,
                    "venue_key": venue_key,
                    "track_id": track_id,
                    "event_order": filtered_order,
                }
            )

        self._calendar_df = pd.DataFrame(records).sort_values("event_order")
        logger.info(f"Built championship calendar with {len(self._calendar_df)} events from {len(configured_venues_normalized)} configured venues")
        return self._calendar_df.copy()

    def build_championship_dataset(self) -> ChampionshipDataset:
        """Return the combined standings, driver tracker, and calendar."""
        standings = self.load_championship_standings()
        driver_points = self.create_driver_point_tracker()
        calendar = self.build_calendar_map()
        return ChampionshipDataset(standings=standings, driver_points=driver_points, calendar=calendar)

    def export_processed_data(self, output_dir: Path) -> None:
        """
        Persist processed artifacts for downstream tools or notebooks.
        Files:
            - championship_standings.csv
            - championship_driver_points.csv
            - championship_calendar.csv
        """
        dataset = self.build_championship_dataset()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset.standings.to_csv(output_dir / "championship_standings.csv", index=False)
        dataset.driver_points.to_csv(output_dir / "championship_driver_points.csv", index=False)
        dataset.calendar.to_csv(output_dir / "championship_calendar.csv", index=False)
        logger.info("Exported processed championship data to %s", output_dir)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _find_championship_file(self) -> Optional[Path]:
        if self._championship_path and self._championship_path.exists():
            return self._championship_path

        candidates = list(self.base_path.glob("**/*Championship*.csv"))
        if not candidates:
            logger.warning("No championship CSV found under %s", self.base_path)
            return None

        # Prefer files in data directories first, fall back to any result
        candidates.sort()
        self._championship_path = candidates[0]
        return self._championship_path

    def _normalize_track_name(self, event_name: str) -> str:
        slug = event_name.strip().lower().replace("’", "").replace("'", "")
        slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
        canonical = TRACK_ALIAS_MAP.get(slug, TRACK_ALIAS_MAP.get(slug.replace("-", " "), slug))
        if canonical:
            return canonical

        # Attempt partial match against alias keys
        for alias, target in TRACK_ALIAS_MAP.items():
            if alias in slug:
                return target
        return slug or "unknown"

    def _build_event_record(
        self,
        row: pd.Series,
        event_name: str,
        race_number: int,
        event_order: int,
    ) -> Dict:
        base_prefix = f"{event_name}_Race {race_number}"

        base_points = _safe_number(row.get(f"{base_prefix}_Points"))
        pole_points = _safe_number(row.get(f"{base_prefix}_PolePoints"))
        fl_points = _safe_number(row.get(f"{base_prefix}_FastestLapPoints"))

        total_extra = _safe_number(row.get(f"{base_prefix}_TotalExtraPoints"))
        if total_extra == 0.0:
            extra_components = [
                f"{base_prefix}_Extra 1",
                f"{base_prefix}_Extra 2",
                f"{base_prefix}_ExtraParticipationPoints",
                f"{base_prefix}_ExtraNotStartedPoints",
                f"{base_prefix}_ExtraNotClassifiedPoints",
            ]
            extra_points = sum(_safe_number(row.get(col)) for col in extra_components)
        else:
            extra_points = total_extra

        status = _clean_status(row.get(f"{base_prefix}_Status"))
        venue_key = self._normalize_track_name(event_name)
        track_id = f"{venue_key}_Race {race_number}"

        record = {
            "driver_number": int(row.get("driver_number", 0)),
            "driver_name": row.get("driver_name", ""),
            "team": row.get("team", ""),
            "driver_class": row.get("driver_class", ""),
            "event_name": event_name,
            "race_number": race_number,
            "event_order": event_order,
            "venue_key": venue_key,
            "track_id": track_id,
            "base_points": base_points,
            "pole_points": pole_points,
            "fastest_lap_points": fl_points,
            "extra_points": extra_points,
            "status": status,
        }
        record["total_points"] = base_points + pole_points + fl_points + extra_points
        return record


