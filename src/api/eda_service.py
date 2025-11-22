from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data_processing.data_loader import DataLoader, validate_data_completeness

RACES = ["Race 1", "Race 2"]


def build_eda_payload(loader: DataLoader) -> Dict[str, object]:
    validation_df = validate_data_completeness(loader)
    results_df = _load_all_results(loader)
    best_laps_df = _load_all_best_laps(loader)
    weather_df = _load_all_weather(loader)
    telemetry_df = _load_all_telemetry(loader)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "completeness": _summarize_completeness(validation_df, loader),
        "results": _summarize_results(results_df, best_laps_df),
        "weather": _summarize_weather(weather_df),
        "telemetry": _summarize_telemetry(telemetry_df),
    }


def _summarize_completeness(validation_df: pd.DataFrame, loader: DataLoader) -> Dict[str, object]:
    summary = {
        "total_checks": int(len(validation_df)),
        "files_found": int(validation_df["found"].sum()) if "found" in validation_df else 0,
        "files_missing": int((~validation_df["found"]).sum()) if "found" in validation_df else 0,
        "overall_pct": float(validation_df["found"].mean() * 100) if len(validation_df) and "found" in validation_df else 0.0,
    }

    breakdown = []
    for venue in loader.venues:
        venue_df = validation_df[validation_df["venue"] == venue] if "venue" in validation_df else pd.DataFrame()
        total = len(venue_df)
        found = int(venue_df["found"].sum()) if total and "found" in venue_df else 0
        pct = (found / total) * 100 if total else 0.0
        breakdown.append({
            "venue": venue,
            "found": found,
            "total": total,
            "pct": pct,
        })

    heatmap_cells: List[Dict[str, object]] = []
    if {"venue", "file_type", "found"}.issubset(validation_df.columns):
        heatmap_cells = [
            {
                "venue": row["venue"],
                "file_type": row["file_type"],
                "found": bool(row["found"]),
            }
            for _, row in validation_df.iterrows()
        ]

    file_types = sorted(validation_df["file_type"].dropna().unique().tolist()) if "file_type" in validation_df else []
    venues = sorted(validation_df["venue"].dropna().unique().tolist()) if "venue" in validation_df else []

    return {
        "summary": summary,
        "breakdown": breakdown,
        "heatmap": {
            "cells": heatmap_cells,
            "venues": venues,
            "file_types": file_types,
        },
    }


def _summarize_results(results_df: pd.DataFrame, best_laps_df: pd.DataFrame) -> Dict[str, object]:
    summary = {
        "total_races": int(results_df.groupby(["venue", "race"]).ngroups) if not results_df.empty else 0,
        "total_drivers": int(results_df["NUMBER"].nunique()) if "NUMBER" in results_df else 0,
        "total_entries": int(len(results_df)),
    }
    laps_box = _build_boxplot(results_df, "venue", "LAPS")
    fastest_hist = _build_histogram(results_df, category_col="venue", value_col="FL_TIME_seconds", bins=15)
    best_lap_hist = _build_histogram(best_laps_df, category_col="venue", value_col="BESTLAP_1_seconds", bins=15)

    summary["laps_stats"] = _describe_series(results_df.get("LAPS"))
    summary["fastest_lap_stats"] = _describe_series(results_df.get("FL_TIME_seconds"))
    summary["best_lap_races"] = int(best_laps_df.groupby(["venue", "race"]).ngroups) if not best_laps_df.empty else 0
    summary["best_lap_drivers"] = int(best_laps_df["NUMBER"].nunique()) if "NUMBER" in best_laps_df else 0

    return {
        "summary": summary,
        "lap_boxplot": laps_box,
        "fastest_lap_histogram": fastest_hist,
        "best_lap_histogram": best_lap_hist,
    }


def _summarize_weather(weather_df: pd.DataFrame) -> Dict[str, object]:
    metrics = {}
    for column, title in [
        ("AIR_TEMP", "Air Temperature"),
        ("TRACK_TEMP", "Track Temperature"),
        ("HUMIDITY", "Humidity"),
        ("WIND_SPEED", "Wind Speed"),
    ]:
        if column in weather_df:
            metrics[column] = {
                "title": title,
                "summary": _describe_series(weather_df[column]),
                "boxplot": _build_boxplot(weather_df, "venue", column),
            }
    return {"metrics": metrics}


def _summarize_telemetry(telemetry_df: pd.DataFrame) -> Dict[str, object]:
    if telemetry_df is None or telemetry_df.empty:
        return {
            "venues_covered": 0,
            "samples": 0,
            "unique_parameters": 0,
            "top_parameters": [],
        }

    telemetry_df = telemetry_df.copy()
    telemetry_df["venue"] = telemetry_df["venue"].fillna("unknown")
    param_col = None
    for candidate in ["parameter_name", "PARAMETER_NAME", "Parameter"]:
        if candidate in telemetry_df.columns:
            param_col = candidate
            break

    top_parameters: List[Dict[str, object]] = []
    unique_params = 0
    if param_col:
        params = telemetry_df[param_col].dropna().astype(str)
        counts = Counter(params)
        unique_params = len(counts)
        top_parameters = [
            {"parameter": name, "count": int(count)}
            for name, count in counts.most_common(20)
        ]

    return {
        "venues_covered": int(telemetry_df["venue"].nunique()) if "venue" in telemetry_df else 0,
        "samples": int(len(telemetry_df)),
        "unique_parameters": unique_params,
        "top_parameters": top_parameters,
    }


def _load_all_results(loader: DataLoader) -> pd.DataFrame:
    frames = []
    for venue in loader.venues:
        for race in RACES:
            df = loader.load_results_file(venue, race, "provisional")
            if df.empty:
                continue
            df = df.copy()
            df["venue"] = venue
            df["race"] = race
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_all_best_laps(loader: DataLoader) -> pd.DataFrame:
    frames = []
    for venue in loader.venues:
        for race in RACES:
            df = loader.load_best_laps(venue, race)
            if df.empty:
                continue
            df = df.copy()
            df["venue"] = venue
            df["race"] = race
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_all_weather(loader: DataLoader) -> pd.DataFrame:
    frames = []
    for venue in loader.venues:
        for race in RACES:
            df = loader.load_weather_data(venue, race)
            if df.empty:
                continue
            df = df.copy()
            df["venue"] = venue
            df["race"] = race
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_all_telemetry(loader: DataLoader) -> pd.DataFrame:
    frames = []
    for venue in loader.venues:
        for race in RACES:
            try:
                df = loader.load_telemetry(venue, race, sample_size=200000)
            except Exception:
                continue
            if df.empty:
                continue
            df = df.copy()
            df["venue"] = venue
            df["race"] = race
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_boxplot(df: pd.DataFrame, group_col: str, value_col: str) -> List[Dict[str, object]]:
    if df.empty or value_col not in df:
        return []
    stats = []
    grouped = df.groupby(group_col)
    for label, group in grouped:
        values = group[value_col].dropna().astype(float).to_numpy()
        if values.size == 0:
            continue
        percentiles = np.percentile(values, [0, 25, 50, 75, 100])
        stats.append({
            "label": label,
            "min": float(percentiles[0]),
            "q1": float(percentiles[1]),
            "median": float(percentiles[2]),
            "q3": float(percentiles[3]),
            "max": float(percentiles[4]),
            "count": int(values.size),
        })
    return stats


def _build_histogram(df: pd.DataFrame, *, category_col: str, value_col: str, bins: int = 15) -> Dict[str, object]:
    if df.empty or value_col not in df or category_col not in df:
        return {"bins": [], "categories": []}

    series = df[value_col].dropna().astype(float)
    if series.empty:
        return {"bins": [], "categories": []}

    min_val = float(series.min())
    max_val = float(series.max())
    if min_val == max_val:
        max_val += 1.0
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    categories = sorted(df[category_col].dropna().unique().tolist())
    per_cat_counts = {}
    for category in categories:
        cat_series = df.loc[df[category_col] == category, value_col].dropna().astype(float)
        counts, _ = np.histogram(cat_series, bins=bin_edges)
        per_cat_counts[category] = counts.astype(int)

    rows: List[Dict[str, object]] = []
    for idx in range(bins):
        row = {
            "bin": f"{bin_edges[idx]:.1f}-{bin_edges[idx + 1]:.1f}",
            "midpoint": float((bin_edges[idx] + bin_edges[idx + 1]) / 2),
        }
        for category in categories:
            row[category] = int(per_cat_counts.get(category, np.zeros(bins, dtype=int))[idx])
        rows.append(row)

    return {"bins": rows, "categories": categories, "range": [min_val, max_val]}


def _describe_series(series: pd.Series | None) -> Dict[str, float]:
    if series is None or series.empty:
        return {}
    desc = series.describe()
    return {key: float(value) for key, value in desc.to_dict().items()}


