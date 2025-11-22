# butterfly_effect_analyzer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .championship_simulator import ChampionshipSimulator

# Default points table (FIA-style). Analyzer will prefer simulator.config.points_system
POINTS_TABLE = {
    1: 25,
    2: 18,
    3: 15,
    4: 12,
    5: 10,
    6: 8,
    7: 6,
    8: 4,
    9: 2,
    10: 1
}


@dataclass
class ImpactReport:
    event_order: int
    track_id: str
    impact_score: float
    champion_changed: bool
    champion_before: str
    champion_after: str
    max_points_delta: float
    key_movers: List[Dict[str, float]]


class ButterflyEffectAnalyzer:
    """
    Analyze how single-race result changes affect the final championship outcome.

    Key improvements in this implementation:
      - Positions forced by the user are honored (not overwritten).
      - Collisions (two drivers assigned the same position) are resolved deterministically
        by placing remaining drivers into the next available slot(s).
      - Points are stored in a unified `points` column and `total_points` (points + bonus_points).
      - Points mapping uses simulator.config.points_system if available, otherwise POINTS_TABLE.
    """

    def __init__(
        self,
        simulator: ChampionshipSimulator,
        include_advanced: bool = True,
        random_seed: int = 42,
    ) -> None:
        self.simulator = simulator
        self.include_advanced = include_advanced
        self.random_seed = random_seed
        # prefer simulator points_map if present; fallback to analyzer POINTS_TABLE
        self.points_map = getattr(simulator.config, "points_system", None) or POINTS_TABLE
        self._baseline_cache: Optional[Dict[str, pd.DataFrame]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_baseline(self) -> Dict[str, pd.DataFrame]:
        """Execute (or return cached) baseline season results."""
        if self._baseline_cache is None:
            self._baseline_cache = self.simulator.simulate_season(
                include_advanced=self.include_advanced,
                random_seed=self.random_seed,
            )
        return self._baseline_cache

    def compute_event_impact(
        self,
        event_order: int,
        changes: Optional[List[Dict]] = None,
        strategy: str = "swap_top_two",
    ) -> ImpactReport:
        """Apply a single-event modification and measure championship delta."""
        baseline = self.run_baseline()
        base_races = baseline["race_results"]
        base_standings = baseline["final_standings"]

        event_mask = base_races["event_order"] == event_order
        if not event_mask.any():
            raise ValueError(f"No race data found for event_order {event_order}")
        track_id = base_races.loc[event_mask, "track_id"].iloc[0]

        modified_event = self._apply_changes_to_event(
            base_races.loc[event_mask],
            changes=changes,
            strategy=strategy,
        )
        updated_races = base_races.copy()
        # replace event rows with modified event (index alignment maintained by mask)
        updated_races.loc[event_mask, :] = modified_event

        alt_standings = self._recalculate_standings(updated_races)
        merged = self._merge_standings(base_standings, alt_standings)

        champion_changed = (
            merged.sort_values("season_points_base", ascending=False).iloc[0]["driver_number"]
            != merged.sort_values("season_points_alt", ascending=False).iloc[0]["driver_number"]
        )
        champion_before = merged.sort_values("season_points_base", ascending=False).iloc[0]["driver_name"]
        champion_after = merged.sort_values("season_points_alt", ascending=False).iloc[0]["driver_name"]

        merged["points_delta_abs"] = merged["season_points_alt"] - merged["season_points_base"]
        merged["rank_delta"] = merged["rank_alt"] - merged["rank_base"]
        max_points_delta = float(merged["points_delta_abs"].abs().max())

        key_movers = (
            merged.loc[merged["points_delta_abs"].abs() > 0]
            .sort_values("points_delta_abs", ascending=False)
            .head(5)[["driver_number", "driver_name", "points_delta_abs", "rank_delta"]]
            .to_dict("records")
        )

        impact_score = max_points_delta + (5.0 if champion_changed else 0.0)

        return ImpactReport(
            event_order=event_order,
            track_id=track_id,
            impact_score=float(impact_score),
            champion_changed=champion_changed,
            champion_before=champion_before,
            champion_after=champion_after,
            max_points_delta=max_points_delta,
            key_movers=key_movers,
        )

    def rank_event_impacts(
        self,
        strategy: str = "swap_top_two",
        max_events: int = 5,
    ) -> List[ImpactReport]:
        """Evaluate every event and rank by impact_score."""
        baseline = self.run_baseline()
        event_orders = sorted(baseline["race_results"]["event_order"].unique())
        reports = [
            self.compute_event_impact(event_order=order, strategy=strategy)
            for order in event_orders
        ]
        reports.sort(key=lambda r: r.impact_score, reverse=True)
        return reports[:max_events]

    def apply_custom_changes(
        self,
        adjustments: List[Dict],
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply custom scenario adjustments (multi-event) and return recalculated data.

        adjustments: List of dicts with keys:
            - event_order (int)
            - changes: List of {driver_number, new_position? or position_delta?}
        """
        baseline = self.run_baseline()
        updated_races = baseline["race_results"].copy()
        for adj in adjustments:
            event_order = adj["event_order"]
            changes = adj.get("changes", [])
            mask = updated_races["event_order"] == event_order
            if not mask.any():
                continue
            modified_event = self._apply_changes_to_event(
                updated_races.loc[mask],
                changes=changes,
                strategy=adj.get("strategy"),
            )
            updated_races.loc[mask, :] = modified_event

        new_standings = self._recalculate_standings(updated_races)
        return {
            "race_results": updated_races,
            "final_standings": new_standings,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_changes_to_event(
        self,
        event_df: pd.DataFrame,
        changes: Optional[List[Dict]] = None,
        strategy: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Apply a change strategy or explicit changes to an event dataframe.

        Behavior differences vs. prior versions:
        - Manual `new_position` or `position_delta` values are honored.
        - We do NOT globally reassign all positions as a repair step. Instead we
          perform localized, deterministic conflict resolution so user changes stick.
        """
        df = event_df.copy().reset_index(drop=True)

        # ensure final_position exists and is integer where possible
        if "final_position" not in df.columns:
            df["final_position"] = np.arange(1, len(df) + 1)
        else:
            # coerce numeric positions when possible; keep NaNs if present
            df["final_position"] = pd.to_numeric(df["final_position"], errors="coerce")

        # Strategy-based change
        if strategy == "swap_top_two" and len(df) >= 2:
            # swap the first two positions by their current ordering in df
            # determine current order by sorting by final_position (NaNs at end)
            ordered = df.sort_values("final_position", na_position="last").reset_index()
            if len(ordered) >= 2:
                idx0 = ordered.loc[0, "index"]
                idx1 = ordered.loc[1, "index"]
                # swap final_position values
                v0 = df.at[idx0, "final_position"]
                v1 = df.at[idx1, "final_position"]
                df.at[idx0, "final_position"], df.at[idx1, "final_position"] = v1, v0

        # Build manual change map and mark requested targets so we can prioritize them
        manual_targets: Dict[int, int] = {}  # driver_number -> requested_position
        if changes:
            for change in changes:
                driver_num = change.get("driver_number")
                if driver_num is None:
                    continue
                # find driver row index
                mask = df["driver_number"] == driver_num
                if not mask.any():
                    # driver not present in this event: skip
                    continue
                idx = df.index[mask][0]
                current_pos = df.at[idx, "final_position"]
                # determine requested target
                if change.get("new_position") is not None:
                    try:
                        target = int(change["new_position"])
                    except Exception:
                        continue
                elif change.get("position_delta") is not None:
                    try:
                        delta = int(change["position_delta"])
                        if pd.isna(current_pos):
                            # if current position is missing, place at end + delta
                            target = max(1, len(df) + delta)
                        else:
                            target = int(current_pos) + delta
                    except Exception:
                        continue
                else:
                    continue
                # clamp target into [1, N]
                target = max(1, min(len(df), int(target)))
                manual_targets[driver_num] = target
                # set temporarily so conflict resolution sees the request
                df.loc[mask, "final_position"] = target

        # Resolve conflicts deterministically without global reset
        df = self._resolve_position_conflicts(df, manual_targets)

        # Recalculate unified points and total_points
        points_map = self.points_map or POINTS_TABLE
        df["points"] = df["final_position"].map(points_map).fillna(0).astype(float)
        if "bonus_points" in df.columns:
            df["bonus_points"] = df["bonus_points"].fillna(0)
            df["total_points"] = df["points"] + df["bonus_points"]
        else:
            df["total_points"] = df["points"]

        # Keep final_position integer and sorted by final_position
        df["final_position"] = df["final_position"].astype(int)
        df = df.sort_values("final_position").reset_index(drop=True)
        return df

    def _resolve_position_conflicts(self, df: pd.DataFrame, manual_targets: Dict[int, int]) -> pd.DataFrame:
        """
        Resolve duplicates/gaps by honoring manual targets and filling remaining drivers into
        the next available slot(s). This avoids a destructive global reassign.

        Rules:
          - Every driver with a manual target keeps that requested position if possible.
          - If multiple drivers request the same position, the one that appears first in the `changes`
            application order (i.e., insertion order in manual_targets) will get the slot; the others
            will be shifted to the next free slots (ascending).
          - Drivers without manual targets keep their original relative ordering (by original final_position
            then by existing row order) and will be placed into remaining free slots in that order.
        """
        n = len(df)
        # Build workspace: map requested pos -> list of driver_numbers (ordered by manual_targets insertion)
        requested: Dict[int, List[int]] = {}
        # preserve insertion order of manual_targets
        for drv, pos in manual_targets.items():
            requested.setdefault(pos, []).append(drv)

        # Build a mapping driver_number -> original order priority for non-manual drivers
        original_order = []
        for idx, row in df.reset_index().iterrows():
            original_order.append((row["driver_number"], row.get("final_position", np.nan), row["index"]))
        # sort non-manual drivers by existing final_position (NaNs last) then by original index
        original_order_sorted = sorted(original_order, key=lambda t: (np.inf if pd.isna(t[1]) else t[1], t[2]))

        # Set to collect assigned drivers
        assigned_positions: Dict[int, int] = {}  # driver_number -> assigned_pos
        occupied = set()

        # 1) Assign manual-requested slots (in the order they were provided)
        for pos in range(1, n + 1):
            if pos in requested:
                # assign the first requester to this pos, others will be queued later
                requesters = requested[pos]
                for i, driver_num in enumerate(requesters):
                    if i == 0 and driver_num not in assigned_positions:
                        assigned_positions[driver_num] = pos
                        occupied.add(pos)
                    else:
                        # queue the remaining requesters into a list to place later
                        # we'll append them to a pending_manual list
                        pass

        # Build pending queue: remaining manual drivers that weren't assigned yet (preserve order)
        pending_manual = []
        for pos in range(1, n + 1):
            if pos in requested:
                for driver_num in requested[pos]:
                    if driver_num not in assigned_positions:
                        pending_manual.append(driver_num)

        # 2) Build list of non-manual drivers in their relative order
        non_manual_drivers = []
        manual_set = set(manual_targets.keys())
        for driver_num, _, _ in original_order_sorted:
            if driver_num not in manual_set:
                non_manual_drivers.append(driver_num)

        # 3) Create a single fill-order queue: first pending manual requesters, then non-manual drivers
        fill_queue = pending_manual + non_manual_drivers

        # 4) Fill remaining free positions (ascending)
        free_positions = [p for p in range(1, n + 1) if p not in occupied]
        free_iter = iter(free_positions)
        for driver_num in fill_queue:
            try:
                pos = next(free_iter)
            except StopIteration:
                # should not happen because free_positions length should match remaining drivers count
                raise RuntimeError("Position resolution exhausted slots unexpectedly.")
            assigned_positions[driver_num] = pos
            occupied.add(pos)

        # 5) As a final step ensure that any manual driver that was assigned a different position than requested
        #    (because the requested was already taken) is placed as close as possible to their request.
        #    Our algorithm above already attempts to honor first-come requesters; others got next slots.

        # Now write assigned positions back into dataframe deterministically
        df = df.copy()
        # If a driver was not present in assigned_positions for some reason, keep their original pos or assign end slots
        for idx, row in df.iterrows():
            dnum = row["driver_number"]
            if dnum in assigned_positions:
                df.at[idx, "final_position"] = int(assigned_positions[dnum])
            else:
                # fallback: place them into next available slot
                remaining = [p for p in range(1, n + 1) if p not in occupied]
                fallback_pos = remaining[0] if remaining else n
                df.at[idx, "final_position"] = int(fallback_pos)
                occupied.add(fallback_pos)

        return df

    def _apply_manual_changes(self, df: pd.DataFrame, changes: Sequence[Dict]) -> pd.DataFrame:
        """
        Backwards-compatible wrapper that produces the `changes` mapping for _apply_changes_to_event.
        Kept for compatibility with earlier calls that used this method signature.
        """
        # convert to the same form processed in _apply_changes_to_event
        # simply forward to _apply_changes_to_event for consistent behavior
        event_df = df.copy()
        return self._apply_changes_to_event(event_df, changes=list(changes))

    def _recalculate_standings(self, race_results: pd.DataFrame) -> pd.DataFrame:
        """
        Given a full race_results DataFrame (possibly updated by one or more events),
        compute season standings grouped by driver_number and return a dataframe with:
          - driver_number
          - driver_name
          - season_points (float)
          - rank (int)
          - position (1-based index after sorting)
        """
        df = race_results.copy()
        # Ensure final_position numeric and sane
        if "final_position" in df.columns:
            df["final_position"] = pd.to_numeric(df["final_position"], errors="coerce")
        else:
            df["final_position"] = np.nan

        # Recompute unified points column based on final_position and configured points map
        points_map = self.points_map or POINTS_TABLE
        df["points"] = df["final_position"].map(points_map).fillna(0).astype(float)
        if "bonus_points" in df.columns:
            df["bonus_points"] = df["bonus_points"].fillna(0)
            df["total_points"] = df["points"] + df["bonus_points"]
        else:
            df["total_points"] = df["points"]

        # Sum season points
        grouped = (
            df.groupby("driver_number")[["total_points"]]
            .sum()
            .rename(columns={"total_points": "season_points"})
            .reset_index()
        )
        # Attach driver names if mapping exists
        grouped["driver_name"] = grouped["driver_number"].map(getattr(self.simulator, "driver_name_map", {}))
        grouped["season_points"] = grouped["season_points"].astype(float)

        # Sort and assign ranks (ties get same rank, rank as int)
        grouped.sort_values(["season_points", "driver_number"], ascending=[False, True], inplace=True)
        grouped["rank"] = grouped["season_points"].rank(method="min", ascending=False).astype(int)
        grouped["position"] = range(1, len(grouped) + 1)
        return grouped

    @staticmethod
    def _merge_standings(base: pd.DataFrame, alt: pd.DataFrame) -> pd.DataFrame:
        """
        Merge baseline and alternate standings into a single frame for delta calculations.
        Expects `base` to have columns including `driver_number`, `season_points` and `rank`
        and `alt` to have the same; result will include `_base` and `_alt` suffix columns.
        """
        merged = base.merge(
            alt,
            on="driver_number",
            how="outer",
            suffixes=("_base", "_alt"),
        ).fillna(0)
        if "driver_name_base" in merged.columns:
            merged["driver_name"] = merged["driver_name_base"].replace("", np.nan).fillna(merged.get("driver_name_alt", ""))
        else:
            merged["driver_name"] = merged.get("driver_name_alt", "")
        return merged


class ScenarioGenerator:
    """Create multi-event what-if scenarios using the ButterflyEffectAnalyzer."""

    def __init__(self, analyzer: ButterflyEffectAnalyzer) -> None:
        self.analyzer = analyzer

    def run_scenario(
        self,
        name: str,
        adjustments: List[Dict],
        description: str = "",
    ) -> Dict:
        outcome = self.analyzer.apply_custom_changes(adjustments)
        timeline = self._build_timeline(outcome["race_results"])
        return {
            "scenario_name": name,
            "description": description,
            "final_standings": outcome["final_standings"],
            "timeline": timeline,
            "adjustments": adjustments,
        }

    @staticmethod
    def _build_timeline(race_results: pd.DataFrame) -> List[Dict]:
        if race_results.empty:
            return []
        points_map = race_results["driver_number"].unique()
        cumulative = {driver: 0.0 for driver in points_map}
        timeline = []
        for event_order, group in race_results.sort_values(["event_order", "final_position"]).groupby("event_order"):
            for row in group.itertuples():
                cumulative[row.driver_number] += row.total_points
            leader = max(cumulative.items(), key=lambda item: item[1])
            timeline.append(
                {
                    "event_order": int(event_order),
                    "track_id": group["track_id"].iloc[0],
                    "leader_driver_number": int(leader[0]),
                    "leader_points": float(leader[1]),
                }
            )
        return timeline
