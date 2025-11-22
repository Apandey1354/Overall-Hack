"""
Shared utilities for building dashboard cache data structures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data_processing.data_loader import DataLoader
from src.data_processing.track_dna_extractor import extract_all_tracks_dna
from src.data_processing.track_clustering import TrackClusterer
from src.data_processing.driver_embedder import create_driver_embeddings
from src.models.track_coach import create_track_coach
from src.championship.championship_simulator import ChampionshipSimulator
from src.championship.butterfly_effect_analyzer import ButterflyEffectAnalyzer

CACHE_DIR = Path("data/cache")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return str(obj)


def ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def save_json(filename: str, payload):
    import json

    ensure_cache_dir()
    path = CACHE_DIR / filename
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    return path


def build_track_dna_summary(loader: DataLoader) -> List[Dict]:
    dna_df = extract_all_tracks_dna(loader)
    if dna_df.empty:
        return []

    clusterer = TrackClusterer(loader)
    cluster_df = clusterer.classify_tracks()
    cluster_map = cluster_df.set_index("track_id").to_dict(orient="index") if not cluster_df.empty else {}

    summary = []
    for _, row in dna_df.iterrows():
        track_id = f"{row['venue']}_{row['race']}"
        tech = row.get("technical_complexity", {}) or {}
        speed = row.get("speed_profile", {}) or {}
        physical = row.get("physical_characteristics", {}) or {}
        classification = cluster_map.get(track_id, {})

        summary.append(
            {
                "track_id": track_id,
                "venue": row.get("venue"),
                "race": row.get("race"),
                "complexity_score": tech.get("complexity_score"),
                "overall_sector_std": tech.get("overall_sector_std") or tech.get("overall_sector_variance"),
                "braking_zones": tech.get("braking_zones", {}).get("count"),
                "straight_corner_ratio": speed.get("straight_corner_ratio", {}).get("ratio"),
                "top_speed": speed.get("top_speed", {}).get("max"),
                "track_length_km": physical.get("track_length", {}).get("estimated_length_km"),
                "num_sectors": physical.get("num_sectors"),
                "cluster_label": classification.get("cluster_label"),
                "cluster": classification.get("cluster"),
            }
        )
    return summary


def build_championship_state(simulator: ChampionshipSimulator) -> Dict:
    single_run = simulator.simulate_season(include_advanced=True, random_seed=42)
    monte_carlo = simulator.run_monte_carlo(iterations=300, include_advanced=True, random_seed=99)

    analyzer = ButterflyEffectAnalyzer(simulator, include_advanced=True, random_seed=42)
    analyzer._baseline_cache = single_run  # reuse simulation result
    top_impacts = analyzer.rank_event_impacts(max_events=5)

    def _report_to_dict(report):
        return {
            "event_order": report.event_order,
            "track_id": report.track_id,
            "impact_score": report.impact_score,
            "champion_changed": report.champion_changed,
            "champion_before": report.champion_before,
            "champion_after": report.champion_after,
            "max_points_delta": report.max_points_delta,
            "key_movers": report.key_movers,
        }

    return {
        "final_standings": single_run["final_standings"].to_dict(orient="records"),
        "race_results": single_run["race_results"].to_dict(orient="records"),
        "monte_carlo_summary": monte_carlo.to_dict(orient="records"),
        "impact_reports": [_report_to_dict(r) for r in top_impacts],
    }


def build_driver_embeddings_summary(driver_embeddings_df: pd.DataFrame) -> List[Dict]:
    """Build a summary of driver embeddings for visualization."""
    if driver_embeddings_df.empty:
        return []
    
    summary = []
    for _, row in driver_embeddings_df.iterrows():
        skill_vector = row.get("skill_vector", [])
        if isinstance(skill_vector, np.ndarray):
            skill_vector = skill_vector.tolist()
        
        summary.append({
            "driver_number": int(row.get("driver_number", 0)),
            "driver_name": row.get("driver_name", f"Driver #{int(row.get('driver_number', 0))}"),
            "skill_vector": skill_vector,
            "technical_proficiency": float(row.get("technical_proficiency", 0.0)),
            "high_speed_proficiency": float(row.get("high_speed_proficiency", 0.0)),
            "consistency_score": float(row.get("consistency_score", 0.0)),
            "weather_adaptability": float(row.get("weather_adaptability", 0.0)),
            "best_track_type": row.get("best_track_type", "unknown"),
            "strengths": row.get("strengths", ""),
        })
    
    return summary


def build_track_coach_data(
    loader: DataLoader,
    driver_embeddings_df: pd.DataFrame,
    track_dna_df: pd.DataFrame,
) -> List[Dict]:
    track_ids = track_dna_df["track_id"].unique().tolist() if "track_id" in track_dna_df.columns else []
    results = []
    for track_id in track_ids:
        try:
            coach = create_track_coach(
                track_id=track_id,
                driver_embeddings_df=driver_embeddings_df,
                track_dna_df=track_dna_df,
                data_loader=loader,
            )
            overview = coach.get_track_overview()
            sectors = coach.get_sector_recommendations()
            weather = {
                cond: coach.get_weather_strategy(cond)
                for cond in ["default", "rain", "hot", "cold"]
            }
            driver_sample = driver_embeddings_df["driver_number"].iloc[0] if not driver_embeddings_df.empty else None
            advice = coach.get_driver_advice(driver_sample) if driver_sample is not None else {}
            results.append(
                {
                    "track_id": track_id,
                    "overview": overview,
                    "sector_recommendations": sectors,
                    "weather_strategies": weather,
                    "driver_advice_sample": advice,
                }
            )
        except Exception as exc:
            print(f"Warning: failed to build coach data for {track_id}: {exc}")
    return results


def generate_cache_bundle():
    loader = DataLoader()
    track_summary = build_track_dna_summary(loader)
    simulator = ChampionshipSimulator(data_loader=loader)
    championship_state = build_championship_state(simulator)
    driver_embeddings_df = create_driver_embeddings(loader)
    track_dna_df = extract_all_tracks_dna(loader)
    coach_data = build_track_coach_data(loader, driver_embeddings_df, track_dna_df)
    driver_embeddings_summary = build_driver_embeddings_summary(driver_embeddings_df)

    save_json("track_dna_summary.json", track_summary)
    save_json("championship_state.json", championship_state)
    save_json("track_coach_data.json", coach_data)
    save_json("driver_embeddings.json", driver_embeddings_summary)

    return {
        "track_summary": track_summary,
        "championship_state": championship_state,
        "coach_data": coach_data,
        "driver_embeddings": driver_embeddings_summary,
    }

