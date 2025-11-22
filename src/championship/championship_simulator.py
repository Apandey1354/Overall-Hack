"""
Championship Simulation Engine (Phase 4.2 & 4.3).

Components:
- RaceOutcomePredictor: blends driver embeddings, track DNA, and transfer learning outputs
- PointsCalculator: encodes championship point/bouns systems
- ChampionshipSimulator: orchestrates season simulations + Monte Carlo experiments
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.data_processing.data_loader import DataLoader
from src.data_processing.driver_embedder import create_driver_embeddings
from src.data_processing.track_dna_extractor import extract_all_tracks_dna
from src.models.transfer_learning_model import create_transfer_model, TrackTransferModel

from .championship_data_processor import (
    ChampionshipDataProcessor,
    ChampionshipDataset,
)

logger = logging.getLogger(__name__)

POSITION_KEYWORDS = [
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "sixteenth",
    "seventeenth",
    "eighteenth",
    "nineteenth",
    "twentieth",
]


@dataclass
class ChampionshipSimulatorConfig:
    points_system: Dict[int, int]
    bonus_points: Dict[str, int]
    monte_carlo_iterations: int = 500
    weather_variation: float = 0.15
    mechanical_wear_rate: float = 0.03
    mechanical_recovery: float = 0.08
    momentum_gain: float = 0.08
    momentum_decay: float = 0.15
    pressure_penalty: float = 0.05
    fatigue_penalty: float = 0.02


class PointsCalculator:
    """Convert finishing positions and bonuses into championship points."""

    def __init__(self, points_map: Dict[int, int], bonus_points: Optional[Dict[str, int]] = None):
        self.points_map = points_map
        self.bonus_points = bonus_points or {}

    def points_for_position(self, position: int) -> int:
        return self.points_map.get(position, 0)

    def assign_points(
        self,
        race_predictions: pd.DataFrame,
        rng: np.random.Generator,
        include_bonus: bool = True,
    ) -> pd.DataFrame:
        if race_predictions.empty:
            return race_predictions

        df = race_predictions.copy()
        df.sort_values("performance_score", ascending=False, inplace=True)
        df["final_position"] = np.arange(1, len(df) + 1)
        df["base_points"] = df.apply(
            lambda row: self.points_for_position(int(row["final_position"])) if row["did_finish"] else 0,
            axis=1,
        )
        df["bonus_points"] = 0

        if include_bonus:
            # Fix issue #2: Drop NaNs first before finding minimum
            valid_lap_times = df["predicted_lap_time"].dropna()
            if not valid_lap_times.empty:
                pole_idx = valid_lap_times.idxmin()
                if pd.notna(pole_idx):
                    df.at[pole_idx, "bonus_points"] += self.bonus_points.get("pole_position", 0)

            # Fastest lap only for drivers who finished
            finished_mask = df["did_finish"] & df["predicted_lap_time"].notna()
            if finished_mask.any():
                finished_lap_times = df.loc[finished_mask, "predicted_lap_time"].dropna()
                if not finished_lap_times.empty:
                    fl_idx = finished_lap_times.idxmin()
                    if pd.notna(fl_idx):
                        df.at[fl_idx, "bonus_points"] += self.bonus_points.get("fastest_lap", 0)

        df["total_points"] = df["base_points"] + df["bonus_points"]
        df["status"] = np.where(df["did_finish"], "FIN", "DNF")
        return df


class RaceOutcomePredictor:
    """
    Estimate finishing order using driver embeddings, track DNA, and (optional) transfer learning predictions.
    """

    def __init__(
        self,
        data_loader: DataLoader,
        driver_embeddings_df: pd.DataFrame,
        driver_points_df: pd.DataFrame,
        calendar_df: pd.DataFrame,
        track_dna_df: pd.DataFrame,
        transfer_model_path: Optional[Path] = None,
    ) -> None:
        self.loader = data_loader
        self.driver_embeddings_df = driver_embeddings_df
        self.driver_points_df = driver_points_df
        self.calendar_df = calendar_df
        self.track_dna_df = track_dna_df
        self.transfer_model = self._load_transfer_model(transfer_model_path)
        self.driver_pool = sorted(driver_embeddings_df["driver_number"].unique().tolist())
        self._track_cache: Dict[str, Dict] = {}
        self._results_cache = self._build_results_cache()

    def predict_race(
        self,
        track_id: str,
        weather_condition: str,
        driver_states: Dict[int, Dict[str, float]],
        include_advanced: bool,
        rng: np.random.Generator,
        current_points: Optional[Dict[int, float]] = None,
        event_order: Optional[int] = None,
    ) -> pd.DataFrame:
        target_dna = self._get_track_dna(track_id)
        track_vector = self._flatten_track_dna(target_dna)
        track_profile = self._summarize_track_profile(target_dna)
        event_order = event_order if event_order is not None else 0

        predictions: List[Dict] = []
        for _, driver_row in self.driver_embeddings_df.iterrows():
            driver_number = int(driver_row["driver_number"])
            if driver_number not in driver_states:
                continue

            skill_vector = np.array(driver_row["skill_vector"], dtype=np.float32)
            driver_state = driver_states[driver_number]
            source_perf, source_track = self._get_recent_performance(driver_number, event_order)

            model_output = None
            if (
                self.transfer_model is not None
                and source_perf is not None
                and track_vector is not None
            ):
                model_output = self._predict_with_transfer_model(skill_vector, source_perf, track_vector)

            base_prediction = self._heuristic_prediction(
                skill_vector=skill_vector,
                track_profile=track_profile,
                model_output=model_output,
            )

            adjusted = self._apply_contextual_adjustments(
                base_prediction=base_prediction,
                driver_state=driver_state,
                weather_condition=weather_condition,
                include_advanced=include_advanced,
                rng=rng,
                current_points=current_points or {},
                driver_number=driver_number,
                event_order=event_order,
            )

            predictions.append(
                {
                    "driver_number": driver_number,
                    "driver_name": driver_row.get("driver_name", driver_row.get("participant", driver_number)),
                    "track_id": track_id,
                    "event_order": event_order,
                    "weather_condition": weather_condition,
                    "predicted_position": adjusted["predicted_position"],
                    "predicted_lap_time": adjusted["predicted_lap_time"],
                    "predicted_speed": adjusted["predicted_speed"],
                    "predicted_rank": adjusted["predicted_rank"],
                    "finish_probability": adjusted["finish_probability"],
                    "mechanical_health": driver_state.get("mechanical_health", 1.0),
                    "momentum": driver_state.get("momentum", 0.0),
                    "did_finish": adjusted["did_finish"],
                    "performance_score": adjusted["performance_score"],
                    "source_track_id": source_track,
                }
            )

        return pd.DataFrame(predictions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_transfer_model(self, model_path: Optional[Path]) -> Optional[TrackTransferModel]:
        candidate = Path(model_path) if model_path else Path("models") / "transfer_model.pt"
        if not candidate.exists():
            logger.warning("Transfer learning model %s not found. Falling back to heuristic predictor.", candidate)
            return None

        model = create_transfer_model()
        try:
            checkpoint = torch.load(candidate, map_location=torch.device("cpu"))
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            model.eval()
            logger.info("Loaded transfer learning model from %s", candidate)
            return model
        except Exception:
            logger.exception("Failed to load transfer model from %s", candidate)
            return None

    def _build_results_cache(self) -> Dict[Tuple[str, int], Dict]:
        cache: Dict[Tuple[str, int], Dict] = {}
        for venue in getattr(self.loader, "venues", []):
            for race in ["Race 1", "Race 2"]:
                track_id = f"{venue}_{race}"
                df = self.loader.load_results_file(venue, race, "provisional")
                if df.empty:
                    df = self.loader.load_results_file(venue, race, "official")
                if df.empty or "NUMBER" not in df.columns:
                    continue
                for _, row in df.iterrows():
                    number = row.get("NUMBER")
                    if pd.isna(number):
                        continue
                    cache[(track_id, int(number))] = row.to_dict()
        return cache

    def _get_track_dna(self, track_id: str) -> Dict:
        if track_id in self._track_cache:
            return self._track_cache[track_id]

        if "track_id" in self.track_dna_df.columns:
            row = self.track_dna_df[self.track_dna_df["track_id"] == track_id]
            if not row.empty:
                dna_dict = row.iloc[0].to_dict()
                self._track_cache[track_id] = dna_dict
                return dna_dict

        # Try to compute on the fly
        try:
            venue, race = track_id.split("_", 1)
            extractor = self.loader  # type: ignore
            from src.data_processing.track_dna_extractor import TrackDNAExtractor

            dna = TrackDNAExtractor(self.loader).extract_track_dna(venue, race)
            self._track_cache[track_id] = dna
            return dna
        except Exception:
            logger.exception("Failed to fetch track DNA for %s", track_id)
            self._track_cache[track_id] = {}
            return {}

    @staticmethod
    def _flatten_track_dna(dna_dict: Dict) -> Optional[np.ndarray]:
        if not dna_dict:
            return None
        features: List[float] = []
        tech = dna_dict.get("technical_complexity", {})
        if isinstance(tech, dict):
            features.append(float(tech.get("complexity_score", 0.0)))
            sector_std = tech.get("overall_sector_std")
            if sector_std is None and "overall_sector_variance" in tech:
                variance_val = tech.get("overall_sector_variance", 0.0)
                sector_std = float(np.sqrt(max(variance_val, 0.0)))
            features.append(float(sector_std or 0.0))
            features.append(float(tech.get("braking_zones", {}).get("count", 0)) / 20.0)
        speed = dna_dict.get("speed_profile", {})
        if isinstance(speed, dict):
            top_speed = speed.get("top_speed", {})
            features.append(float(top_speed.get("max", 0.0)) / 200.0 if isinstance(top_speed, dict) else 0.0)
            features.append(float(speed.get("straight_corner_ratio", {}).get("ratio", 1.0)) / 2.0)
        physical = dna_dict.get("physical_characteristics", {})
        if isinstance(physical, dict):
            track_len = physical.get("track_length", {})
            features.append(float(track_len.get("estimated_length_km", 0.0)) / 10.0 if isinstance(track_len, dict) else 0.0)
            features.append(float(physical.get("num_sectors", 0)) / 10.0)
        patterns = dna_dict.get("performance_patterns", {})
        if isinstance(patterns, dict):
            lap_var = patterns.get("lap_time_variance", {})
            features.append(float(lap_var.get("cv", 0.0)) * 10.0 if isinstance(lap_var, dict) else 0.0)

        while len(features) < 20:
            features.append(0.0)
        return np.array(features[:20], dtype=np.float32)

    @staticmethod
    def _summarize_track_profile(dna_dict: Dict) -> Dict[str, float]:
        tech_score = float(dna_dict.get("technical_complexity", {}).get("complexity_score", 0.0)) if dna_dict else 0.0
        speed_ratio = float(
            dna_dict.get("speed_profile", {}).get("straight_corner_ratio", {}).get("ratio", 1.0)
        ) if dna_dict else 1.0
        length = float(
            dna_dict.get("physical_characteristics", {}).get("track_length", {}).get("estimated_length_km", 0.0)
        ) if dna_dict else 0.0
        return {
            "technicality": tech_score,
            "speed_ratio": speed_ratio,
            "length": length,
        }

    def _get_recent_performance(self, driver_number: int, event_order: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
        if self.driver_points_df.empty or "driver_number" not in self.driver_points_df.columns:
            return None, None
        history = self.driver_points_df[self.driver_points_df["driver_number"] == driver_number]
        if history.empty:
            return None, None
        past = history[history["event_order"] < event_order]
        if past.empty:
            past = history.sort_values("event_order").head(1)
        latest = past.sort_values("event_order").tail(1).iloc[0]
        track_id = latest.get("track_id")
        result_row = self._results_cache.get((track_id, driver_number), {})
        perf_vector = self._build_performance_vector(result_row, latest)
        return perf_vector, track_id

    @staticmethod
    def _build_performance_vector(result_row: Dict, championship_row: Optional[pd.Series]) -> Optional[np.ndarray]:
        if not result_row and championship_row is None:
            return None

        def _parse_lap_time(raw: object) -> Optional[float]:
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                return None
            token = str(raw).strip()
            if not token:
                return None
            try:
                if ":" in token:
                    minutes, seconds = token.split(":", 1)
                    seconds = float(seconds)
                    return int(minutes) * 60 + seconds
                return float(token)
            except Exception:
                return None

        position = result_row.get("POSITION")
        if pd.isna(position) and championship_row is not None:
            # Fix issue #4: Better position approximation from points
            # Use inverse points mapping: higher points = better position
            points = float(championship_row.get("base_points", 0.0))
            if points >= 25:
                position = 1.0
            elif points >= 18:
                position = 2.0
            elif points >= 15:
                position = 3.0
            elif points >= 12:
                position = 4.0
            elif points >= 10:
                position = 5.0
            elif points >= 8:
                position = 6.0
            elif points >= 6:
                position = 7.0
            elif points >= 4:
                position = 8.0
            elif points >= 2:
                position = 9.0
            elif points >= 1:
                position = 10.0
            else:
                # No points = lower positions, but not necessarily last
                position = 15.0
        lap_time = result_row.get("FL_TIME_seconds") or _parse_lap_time(result_row.get("FL_TIME"))
        speed = result_row.get("FL_KPH")
        status = str(result_row.get("STATUS", "")).lower()
        finished = 0.0 if status in {"dnf", "dsq", "nc", "dns"} else 1.0
        laps = result_row.get("LAPS", 0)

        features = []
        if position:
            pos_score = max(0.0, 1.0 - (float(position) - 1) / 19.0)
        else:
            pos_score = 0.3
        features.append(pos_score)

        if lap_time:
            # Fix issue #5: Better lap time scoring - faster times get higher scores
            # Normalize: 80s = 1.0, 150s = 0.0 (typical range for GT racing)
            lap_time_val = float(lap_time)
            if lap_time_val <= 80.0:
                lap_score = 1.0
            elif lap_time_val >= 150.0:
                lap_score = 0.0
            else:
                # Linear interpolation: (150 - lap_time) / (150 - 80)
                lap_score = (150.0 - lap_time_val) / 70.0
            lap_score = max(0.0, min(1.0, lap_score))
        else:
            lap_score = 0.0
        features.append(lap_score)

        if pd.notna(speed):
            # Fix issue #10: Better speed mapping with wider range
            # Normalize: 120 km/h = 0.0, 200 km/h = 1.0 (typical GT range)
            speed_val = float(speed)
            if speed_val <= 120.0:
                speed_score = 0.0
            elif speed_val >= 200.0:
                speed_score = 1.0
            else:
                speed_score = (speed_val - 120.0) / 80.0
            speed_score = max(0.0, min(1.0, speed_score))
        else:
            speed_score = 0.0
        features.append(speed_score)

        features.append(float(finished))
        features.append(min(1.0, float(laps or 0) / 30.0))
        return np.array(features, dtype=np.float32)

    def _predict_with_transfer_model(
        self,
        skill_vector: np.ndarray,
        source_performance: np.ndarray,
        target_track_vector: np.ndarray,
    ) -> np.ndarray:
        try:
            # Fix transfer model dimension issues: ensure correct input sizes
            # Model expects: driver_embedding_dim=8, performance_features=5, track_dna_dim=20
            if skill_vector is None or len(skill_vector) == 0:
                return None
            skill_8d = skill_vector[:8] if len(skill_vector) >= 8 else np.pad(skill_vector, (0, max(0, 8 - len(skill_vector))), 'constant')
            
            if source_performance is None or len(source_performance) == 0:
                return None
            perf_5d = source_performance[:5] if len(source_performance) >= 5 else np.pad(source_performance, (0, max(0, 5 - len(source_performance))), 'constant')
            
            if target_track_vector is None or len(target_track_vector) == 0:
                return None
            track_20d = target_track_vector[:20] if len(target_track_vector) >= 20 else np.pad(target_track_vector, (0, max(0, 20 - len(target_track_vector))), 'constant')
            
            driver_tensor = torch.FloatTensor(skill_8d).unsqueeze(0)
            source_tensor = torch.FloatTensor(perf_5d).unsqueeze(0)
            target_tensor = torch.FloatTensor(track_20d).unsqueeze(0)
            with torch.no_grad():
                output = self.transfer_model(driver_tensor, source_tensor, target_tensor)
            return output.squeeze(0).numpy()
        except Exception:
            logger.exception("Transfer model prediction failed")
            return None

    @staticmethod
    def _heuristic_prediction(
        skill_vector: np.ndarray,
        track_profile: Dict[str, float],
        model_output: Optional[np.ndarray],
    ) -> Dict[str, float]:
        # Skill vector layout: [technical, high-speed, consistency, weather,
        #                       tech-track, speed-track, balanced-track, finish-rate]
        technical_component = np.clip(0.6 * skill_vector[0] + 0.4 * skill_vector[4], 0.0, 1.0)
        speed_component = np.clip(0.6 * skill_vector[1] + 0.4 * skill_vector[5], 0.0, 1.0)
        balance_component = np.clip(0.6 * skill_vector[2] + 0.4 * skill_vector[6], 0.0, 1.0)
        finish_component = skill_vector[7]

        technicality = float(track_profile.get("technicality", 0.5))
        speed_ratio = float(track_profile.get("speed_ratio", 1.0))
        track_length = float(track_profile.get("length", 4.0))  # km

        tech_weight = 0.25 + technicality * 0.5
        speed_weight = 0.25 + max(0.0, speed_ratio - 1.0) * 0.4
        balance_weight = max(0.15, 1.0 - (tech_weight + speed_weight))
        
        # Fix issue #3: Normalize weights to ensure denominator is always 1.0
        total_weight = tech_weight + speed_weight + balance_weight
        if total_weight > 0:
            tech_weight /= total_weight
            speed_weight /= total_weight
            balance_weight /= total_weight
        else:
            # Fallback if all weights are zero
            tech_weight = speed_weight = balance_weight = 1.0 / 3.0

        weighted_skill = (
            technical_component * tech_weight
            + speed_component * speed_weight
            + balance_component * balance_weight
        )
        weighted_skill = float(np.clip(weighted_skill, 0.0, 1.0))

        baseline_lap = 95.0
        # Fix issue #6: Better handling of track length
        if track_length <= 0:
            track_length = 4.0  # Default to 4km if missing
        
        lap_length_factor = np.clip(track_length / 4.0, 0.7, 1.5)
        lap_skill_factor = 1.0 + (1.0 - weighted_skill) * 0.35
        predicted_lap_time = baseline_lap * lap_length_factor * lap_skill_factor

        # Fix issue #6: Always use proper speed calculation with valid track length
        if predicted_lap_time > 0:
            predicted_speed = float(track_length / (predicted_lap_time / 3600.0))
            # Clamp to realistic racing speeds (120-220 km/h for GT cars)
            predicted_speed = float(np.clip(predicted_speed, 120.0, 220.0))
        else:
            # Fallback only if lap time is invalid
            predicted_speed = 170.0 * lap_length_factor * (0.8 + weighted_skill * 0.4)
        predicted_position = 1.0 + (1.0 - weighted_skill) * 19.0
        finish_probability = float(np.clip(0.45 + 0.55 * finish_component, 0.05, 0.995))

        if model_output is not None and len(model_output) >= 4:
            ml_lap, ml_pos, ml_speed, ml_finish = model_output[:4]
            predicted_lap_time = float(np.clip(ml_lap, 80.0, 150.0))
            predicted_position = float(np.clip(ml_pos, 1.0, 20.0))
            predicted_speed = float(np.clip(ml_speed, 150.0, 210.0))
            finish_probability = float(np.clip(ml_finish, 0.05, 0.99))

        return {
            "predicted_position": predicted_position,
            "predicted_lap_time": predicted_lap_time,
            "predicted_speed": predicted_speed,
            "finish_probability": finish_probability,
        }

    def _apply_contextual_adjustments(
        self,
        base_prediction: Dict[str, float],
        driver_state: Dict[str, float],
        weather_condition: str,
        include_advanced: bool,
        rng: np.random.Generator,
        current_points: Dict[int, float],
        driver_number: int,
        event_order: int,
    ) -> Dict[str, float]:
        prediction = base_prediction.copy()
        base_score = max(0.0, 21.0 - prediction["predicted_position"])
        
        # Fix issue #11: Better randomness model - skill-dependent and track-appropriate
        # Better drivers have less variance, worse drivers have more
        skill_level = 1.0 - (prediction["predicted_position"] - 1.0) / 19.0
        randomness_scale = 0.5 + (1.0 - skill_level) * 0.5  # 0.5 to 1.0 scale
        randomness = np.clip(rng.normal(loc=0.0, scale=randomness_scale), -2.5, 2.5)
        performance_score = np.clip(base_score + randomness, 0.0, 25.0)

        finish_probability = prediction["finish_probability"]
        weather_skill = driver_state.get("weather_skill", 0.5)
        if include_advanced:
            condition_penalties = {
                "rain": 0.18,
                "hot": 0.08,
                "cold": 0.06,
            }
            penalty_base = condition_penalties.get(weather_condition.lower(), 0.0)
            if penalty_base > 0.0:
                # Fix issue #7: Cap weather penalty to prevent blowouts
                weather_penalty = min(penalty_base * (1.0 - weather_skill), 0.25)  # Cap at 25%
                prediction["predicted_lap_time"] *= 1.0 + weather_penalty * 0.85
                prediction["predicted_speed"] *= 1.0 - weather_penalty * 0.6
                performance_score -= weather_penalty * 6.0
                finish_probability *= np.clip(1.0 - weather_penalty * 0.75, 0.4, 1.0)

            momentum = driver_state.get("momentum", 0.0)
            performance_score += momentum * 2.0
            finish_probability = np.clip(finish_probability + momentum * 0.1, 0.05, 0.99)

            mech = driver_state.get("mechanical_health", 1.0)
            # Fix issue #8: Reduce mechanical penalty to avoid double punishment
            performance_score -= (1.0 - mech) * 2.5  # Reduced from 4.0
            finish_probability *= np.clip(mech, 0.5, 1.0)  # Less harsh clipping

            fatigue = driver_state.get("fatigue", 0.0)
            performance_score -= fatigue * 1.5
            finish_probability *= max(0.6, 1.0 - fatigue * 0.3)  # Less harsh fatigue

            top_points = max(current_points.values()) if current_points else 0.0
            if top_points > 0 and current_points.get(driver_number, 0.0) >= top_points * 0.9:
                # Fix issue #9: Better pressure handling to avoid negative probabilities
                pressure = driver_state.get("pressure", 0.0)
                performance_score -= pressure * 1.5
                # Apply pressure more carefully to avoid underflow
                pressure_multiplier = max(0.7, 1.0 - pressure * 0.3)
                finish_probability = np.clip(finish_probability * pressure_multiplier, 0.05, 0.99)

        # Fix issue #9: Ensure finish_probability is valid before using it
        finish_probability = np.clip(finish_probability, 0.05, 0.99)
        did_finish = rng.random() < finish_probability
        
        # Fix issue #12: Less harsh DNF penalty
        if not did_finish:
            # Reduce DNF penalty from 15 to 8 points
            performance_score = max(performance_score - 8.0, 0.0)

        performance_score = float(np.clip(performance_score, 0.0, 25.0))

        predicted_rank = float(np.clip(prediction["predicted_position"], 1.0, 20.0))
        return {
            **prediction,
            "finish_probability": float(np.clip(finish_probability, 0.01, 0.995)),
            "performance_score": float(performance_score),
            "did_finish": bool(did_finish),
            "predicted_rank": predicted_rank,
        }


class ChampionshipSimulator:
    """High-level API for running single-season and Monte Carlo simulations."""

    def __init__(
        self,
        data_loader: Optional[DataLoader] = None,
        data_processor: Optional[ChampionshipDataProcessor] = None,
        driver_embeddings_df: Optional[pd.DataFrame] = None,
        track_dna_df: Optional[pd.DataFrame] = None,
        config: Optional[ChampionshipSimulatorConfig] = None,
    ) -> None:
        self.loader = data_loader or DataLoader()
        self.data_processor = data_processor or ChampionshipDataProcessor(self.loader)
        self.dataset: ChampionshipDataset = self.data_processor.build_championship_dataset()
        self.driver_embeddings_df = driver_embeddings_df or create_driver_embeddings(self.loader)
        if self.driver_embeddings_df.empty:
            raise ValueError("Driver embeddings dataframe is empty; cannot run simulations.")
        self.track_dna_df = track_dna_df or extract_all_tracks_dna(self.loader)
        self.driver_name_map = self._driver_number_to_name()

        if "driver_name" not in self.driver_embeddings_df.columns:
            self.driver_embeddings_df["driver_name"] = ""
        self.driver_embeddings_df["driver_name"] = self.driver_embeddings_df["driver_number"].apply(
            lambda num: self.driver_name_map.get(num, f"Driver #{int(num)}")
        )

        self.config = config or self._build_default_config()
        self.points_calculator = PointsCalculator(self.config.points_system, self.config.bonus_points)
        self.predictor = RaceOutcomePredictor(
            data_loader=self.loader,
            driver_embeddings_df=self.driver_embeddings_df,
            driver_points_df=self.dataset.driver_points,
            calendar_df=self.dataset.calendar,
            track_dna_df=self.track_dna_df,
        )
        self.driver_pool = sorted(self.driver_embeddings_df["driver_number"].unique().tolist())

    # ------------------------------------------------------------------
    # Simulation entry points
    # ------------------------------------------------------------------
    def simulate_season(
        self,
        include_advanced: bool = True,
        random_seed: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        rng = np.random.default_rng(random_seed)
        driver_states = self._initial_driver_states()
        cumulative_points = {driver: 0.0 for driver in self.driver_pool}
        race_records: List[pd.DataFrame] = []

        calendar = self.dataset.calendar.sort_values("event_order")
        for _, event in calendar.iterrows():
            track_id = event["track_id"]
            event_order = int(event["event_order"])
            weather = self._sample_weather_condition(track_id, rng)
            predictions = self.predictor.predict_race(
                track_id=track_id,
                weather_condition=weather["label"],
                driver_states=driver_states,
                include_advanced=include_advanced,
                rng=rng,
                current_points=cumulative_points,
                event_order=event_order,
            )
            if predictions.empty:
                continue
            race_result = self.points_calculator.assign_points(predictions, rng=rng, include_bonus=True)
            race_result["event_order"] = event_order
            race_result["track_id"] = track_id
            race_result["weather_condition"] = weather["label"]
            race_results_df = race_result.copy()

            for row in race_result.itertuples():
                cumulative_points[row.driver_number] += row.total_points
                self._update_driver_state(driver_states[row.driver_number], row)

            race_results_df["season_points"] = race_results_df["driver_number"].map(cumulative_points)
            race_records.append(race_results_df)

        combined = pd.concat(race_records, ignore_index=True) if race_records else pd.DataFrame()
        if combined.empty:
            final_standings = pd.DataFrame(
                {
                    "driver_number": self.driver_pool,
                    "season_points": [0.0] * len(self.driver_pool),
                }
            )
        else:
            final_standings = (
                combined.groupby("driver_number")[["total_points"]]
                .sum()
                .rename(columns={"total_points": "season_points"})
                .reset_index()
            )
        final_standings["driver_name"] = final_standings["driver_number"].map(self.driver_name_map)
        final_standings["driver_name"] = final_standings.apply(
            lambda row: row["driver_name"] if isinstance(row["driver_name"], str) and row["driver_name"] else f"Driver #{int(row['driver_number'])}",
            axis=1,
        )
        final_standings.sort_values("season_points", ascending=False, inplace=True)
        final_standings["rank"] = final_standings["season_points"].rank(
            method="min", ascending=False
        )
        final_standings["is_champion"] = final_standings["rank"] == 1
        final_standings["is_podium"] = final_standings["rank"] <= 3

        return {
            "race_results": combined,
            "final_standings": final_standings,
        }

    def run_monte_carlo(
        self,
        iterations: Optional[int] = None,
        include_advanced: bool = True,
        random_seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run multiple independent seasons to estimate probabilities.

        Note: Each iteration starts from a fresh driver state (no fatigue/momentum
        carry-over between seasons).
        """
        iterations = iterations or self.config.monte_carlo_iterations
        records: List[pd.DataFrame] = []
        base_seed = random_seed or 42

        for idx in range(iterations):
            result = self.simulate_season(include_advanced=include_advanced, random_seed=base_seed + idx)
            standings = result["final_standings"].copy()
            standings["iteration"] = idx
            records.append(standings)

        all_results = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
        if all_results.empty:
            return all_results

        summary = (
            all_results.groupby("driver_number")
            .agg(
                average_points=("season_points", "mean"),
                points_std=("season_points", "std"),
                win_probability=("is_champion", "mean"),
                podium_probability=("is_podium", "mean"),
            )
            .reset_index()
        )
        summary["driver_name"] = summary["driver_number"].map(self.driver_name_map)
        summary["driver_name"] = summary.apply(
            lambda row: row["driver_name"]
            if isinstance(row["driver_name"], str) and row["driver_name"]
            else f"Driver #{int(row['driver_number'])}",
            axis=1,
        )
        summary.sort_values("average_points", ascending=False, inplace=True)
        return summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_default_config(self) -> ChampionshipSimulatorConfig:
        sim_cfg = (
            self.loader.config.get("simulation", {}).get("championship", {})
            if hasattr(self.loader, "config")
            else {}
        )
        model_cfg = (
            self.loader.config.get("models", {}).get("championship", {})
            if hasattr(self.loader, "config")
            else {}
        )

        raw_points = sim_cfg.get("points_system", {})
        default_points_sequence = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
        default_points: Dict[int, int] = {
            pos: (default_points_sequence[pos - 1] if pos <= len(default_points_sequence) else 0)
            for pos in range(1, len(POSITION_KEYWORDS) + 1)
        }
        points_map: Dict[int, int] = default_points.copy()
        for idx, key in enumerate(POSITION_KEYWORDS, start=1):
            if key in raw_points:
                points_map[idx] = int(raw_points[key])

        bonus_points = sim_cfg.get("bonus_points", {"pole_position": 1, "fastest_lap": 1})
        monte_carlo_iterations = int(model_cfg.get("monte_carlo_iterations", 500))

        return ChampionshipSimulatorConfig(
            points_system=points_map,
            bonus_points=bonus_points,
            monte_carlo_iterations=monte_carlo_iterations,
        )

    def _initial_driver_states(self) -> Dict[int, Dict[str, float]]:
        states = {}
        for driver in self.driver_pool:
            skill_entry = self.driver_embeddings_df.loc[
                self.driver_embeddings_df["driver_number"] == driver, "skill_vector"
            ]
            weather_skill = 0.5
            if not skill_entry.empty:
                vector = skill_entry.iloc[0]
                if isinstance(vector, (list, tuple, np.ndarray)) and len(vector) > 3:
                    weather_skill = float(vector[3])

            states[driver] = {
                "mechanical_health": 1.0,
                "momentum": 0.0,
                "fatigue": 0.0,
                "pressure": 0.0,
                "weather_skill": weather_skill,
            }
        return states

    def _sample_weather_condition(self, track_id: str, rng: np.random.Generator) -> Dict[str, float]:
        venue, race = track_id.split("_", 1)
        weather_df = self.loader.load_weather_data(venue, race)
        if weather_df.empty:
            return {"label": "default", "rain_probability": 0.15}

        rain_prob = 0.0
        if "RAIN" in weather_df.columns:
            rain_prob = float((weather_df["RAIN"] > 0).mean())

        avg_temp = float(weather_df["AIR_TEMP"].mean()) if "AIR_TEMP" in weather_df.columns else 24.0
        base_label = "default"
        if avg_temp >= 32:
            base_label = "hot"
        elif avg_temp <= 15:
            base_label = "cold"

        if rng.random() < rain_prob * (1.0 + self.config.weather_variation):
            label = "rain"
        else:
            label = base_label

        return {"label": label, "rain_probability": rain_prob, "average_temp": avg_temp}

    def _update_driver_state(self, state: Dict[str, float], race_row: pd.Series) -> None:
        wear_rate = self.config.mechanical_wear_rate
        recovery = self.config.mechanical_recovery
        momentum_gain = self.config.momentum_gain
        momentum_decay = self.config.momentum_decay

        delta = -wear_rate
        if race_row.did_finish:
            delta += recovery
        else:
            delta -= wear_rate * 1.5
        state["mechanical_health"] = float(np.clip(state["mechanical_health"] + delta, 0.3, 1.0))
        if int(race_row.final_position) <= 3 and race_row.did_finish:
            state["momentum"] = float(np.clip(state["momentum"] + momentum_gain, 0.0, 0.5))
        else:
            state["momentum"] = float(np.clip(state["momentum"] * (1.0 - momentum_decay), 0.0, 0.5))

        state["fatigue"] = float(np.clip(state.get("fatigue", 0.0) + self.config.fatigue_penalty, 0.0, 0.3))
        state["pressure"] = float(np.clip(state.get("pressure", 0.0) * 0.8, 0.0, 0.4))

    def _driver_number_to_name(self, source_df: Optional[pd.DataFrame] = None) -> Dict[int, str]:
        mapping: Dict[int, str] = {}
        source = source_df if source_df is not None else (
            self.dataset.standings
            if "driver_name" in self.dataset.standings.columns
            else self.driver_embeddings_df
        )
        for _, row in source.iterrows():
            driver_number = int(row.get("driver_number", 0))
            name_candidates = [
                row.get("driver_name"),
                row.get("participant"),
                row.get("first_name"),
                row.get("last_name"),
            ]
            label = ""
            for candidate in name_candidates:
                if candidate is None:
                    continue
                if isinstance(candidate, float) and pd.isna(candidate):
                    continue
                value = str(candidate).strip()
                if value and value.lower() != "nan":
                    label = value
                    break
            if not label:
                label = f"Driver #{driver_number}"
            mapping[driver_number] = label
        return mapping


