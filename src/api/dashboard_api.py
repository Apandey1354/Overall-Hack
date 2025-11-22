from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os

import pandas as pd
import logging
from src.data_processing.data_loader import DataLoader

logger = logging.getLogger(__name__)
from src.data_processing.track_dna_extractor import extract_all_tracks_dna
from src.data_processing.driver_embedder import create_driver_embeddings
from src.models.track_coach import create_track_coach
from src.championship.championship_simulator import ChampionshipSimulator
from src.championship.butterfly_effect_analyzer import ButterflyEffectAnalyzer, ScenarioGenerator
from .eda_service import build_eda_payload
from .ai_recommendation_service import (
    generate_coach_recommendations,
    generate_scenario_recommendations,
    generate_combined_recommendations,
)
from .chatbot_service import generate_chatbot_response
from .cache_utils import (
    CACHE_DIR,
    build_track_dna_summary,
    build_championship_state,
    build_track_coach_data,
    build_driver_embeddings_summary,
    _json_default,
)
from src.data_processing.karma_stream import compute_stream, COMPONENT_SPECS
def _to_plain(obj):
    raw = json.loads(json.dumps(obj, default=_json_default, allow_nan=True))
    return _replace_nonfinite(raw)


def _replace_nonfinite(value):
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, list):
        return [_replace_nonfinite(v) for v in value]
    if isinstance(value, dict):
        return {k: _replace_nonfinite(v) for k, v in value.items()}
    return value



app = FastAPI(title="GR Cup Dashboard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


loader = DataLoader()
simulator = ChampionshipSimulator(data_loader=loader)
analyzer = ButterflyEffectAnalyzer(simulator, include_advanced=True, random_seed=21)
scenario_generator = ScenarioGenerator(analyzer)
driver_embeddings_df = create_driver_embeddings(loader)
track_dna_df = extract_all_tracks_dna(loader)

# Store uploaded telemetry data in memory
_uploaded_per_lap_data: Optional[pd.DataFrame] = None


class DriverChange(BaseModel):
    driver_number: int
    new_position: Optional[int] = None
    position_delta: Optional[int] = None


class EventAdjustment(BaseModel):
    event_order: int
    changes: List[DriverChange]


class ScenarioRequest(BaseModel):
    name: str = "Custom Scenario"
    description: str = ""
    adjustments: List[EventAdjustment]
    target_driver_number: Optional[int] = None  # For championship win analysis


class CoachRequest(BaseModel):
    track_id: str
    driver_number: int


class AIRecommendationRequest(BaseModel):
    coach_insights: Optional[Dict] = None
    scenario_changes: Optional[Dict] = None


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatbotRequest(BaseModel):
    question: str
    conversation_history: Optional[List[ChatMessage]] = None


class SimulationRequest(BaseModel):
    iterations: Optional[int] = 200
    include_advanced: bool = True
    random_seed: Optional[int] = None
    run_monte_carlo: bool = True


def _load_or_rebuild(filename: str, builder):
    path = CACHE_DIR / filename
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    data = builder()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


@app.get("/api/status")
def status():
    return {"status": "ok"}


@app.get("/api/cache/track-dna")
def get_track_dna():
    return _load_or_rebuild(
        "track_dna_summary.json",
        lambda: build_track_dna_summary(loader),
    )


@app.get("/api/cache/championship")
def get_championship_state():
    return _load_or_rebuild(
        "championship_state.json",
        lambda: build_championship_state(simulator),
    )


@app.get("/api/cache/track-coach")
def get_track_coach():
    return _load_or_rebuild(
        "track_coach_data.json",
        lambda: build_track_coach_data(loader, driver_embeddings_df, track_dna_df),
    )


@app.get("/api/cache/driver-embeddings")
def get_driver_embeddings():
    return _load_or_rebuild(
        "driver_embeddings.json",
        lambda: build_driver_embeddings_summary(driver_embeddings_df),
    )


@app.post("/api/scenario/run")
def run_custom_scenario(req: ScenarioRequest):
    analyzer.run_baseline()
    adjustments = [
        {
            "event_order": event.event_order,
            "changes": [
                change.model_dump(exclude_none=True)
                for change in event.changes
            ],
        }
        for event in req.adjustments
    ]
    result = scenario_generator.run_scenario(
        name=req.name,
        description=req.description,
        adjustments=adjustments,
    )
    return _to_plain(result)


@app.post("/api/coach/advice")
def get_driver_advice(req: CoachRequest):
    coach = create_track_coach(
        track_id=req.track_id,
        driver_embeddings_df=driver_embeddings_df,
        track_dna_df=track_dna_df,
        data_loader=loader,
    )
    try:
        advice = coach.get_driver_advice(req.driver_number)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _to_plain({
        "track_id": req.track_id,
        "driver_number": req.driver_number,
        "advice": advice,
    })


@app.get("/api/eda/dashboard")
def get_eda_dashboard():
    data = build_eda_payload(loader)
    return _to_plain(data)


@app.post("/api/ai/coach-recommendations")
def get_coach_ai_recommendations(req: CoachRequest):
    """Generate AI recommendations based on Coach Lab insights."""
    coach = create_track_coach(
        track_id=req.track_id,
        driver_embeddings_df=driver_embeddings_df,
        track_dna_df=track_dna_df,
        data_loader=loader,
    )
    try:
        advice = coach.get_driver_advice(req.driver_number)
        sectors = coach.get_sector_recommendations(req.driver_number)
        weather = {
            cond: coach.get_weather_strategy(cond)
            for cond in ["default", "rain", "hot", "cold"]
        }
        
        # Get driver name safely
        driver_name = f"Driver #{req.driver_number}"
        try:
            driver_row = driver_embeddings_df[driver_embeddings_df["driver_number"] == req.driver_number]
            if not driver_row.empty:
                if "driver_name" in driver_row.columns:
                    name_val = driver_row["driver_name"].iloc[0]
                    if pd.notna(name_val) and str(name_val).strip():
                        driver_name = str(name_val).strip()
        except Exception:
            pass  # Use default driver_name if extraction fails
        
        # Get driver embedding data
        driver_embedding = None
        try:
            driver_row = driver_embeddings_df[driver_embeddings_df["driver_number"] == req.driver_number]
            if not driver_row.empty:
                skill_vector = driver_row["skill_vector"].iloc[0] if "skill_vector" in driver_row.columns else []
                # Convert numpy array to list if needed
                if hasattr(skill_vector, 'tolist'):
                    skill_vector = skill_vector.tolist()
                elif not isinstance(skill_vector, list):
                    skill_vector = list(skill_vector) if skill_vector else []
                
                driver_embedding = {
                    "driver_number": req.driver_number,
                    "driver_name": driver_name,
                    "skill_vector": skill_vector,
                    "technical_proficiency": float(driver_row["technical_proficiency"].iloc[0]) if "technical_proficiency" in driver_row.columns and pd.notna(driver_row["technical_proficiency"].iloc[0]) else 0.0,
                    "high_speed_proficiency": float(driver_row["high_speed_proficiency"].iloc[0]) if "high_speed_proficiency" in driver_row.columns and pd.notna(driver_row["high_speed_proficiency"].iloc[0]) else 0.0,
                    "consistency_score": float(driver_row["consistency_score"].iloc[0]) if "consistency_score" in driver_row.columns and pd.notna(driver_row["consistency_score"].iloc[0]) else 0.0,
                    "weather_adaptability": float(driver_row["weather_adaptability"].iloc[0]) if "weather_adaptability" in driver_row.columns and pd.notna(driver_row["weather_adaptability"].iloc[0]) else 0.0,
                    "best_track_type": str(driver_row["best_track_type"].iloc[0]) if "best_track_type" in driver_row.columns and pd.notna(driver_row["best_track_type"].iloc[0]) else "unknown",
                    "strengths": str(driver_row["strengths"].iloc[0]) if "strengths" in driver_row.columns and pd.notna(driver_row["strengths"].iloc[0]) else "",
                }
        except Exception as e:
            print(f"Warning: Could not extract driver embedding: {e}")
            pass  # Continue without embedding data if extraction fails
        
        recommendation = generate_coach_recommendations(
            track_id=req.track_id,
            coach_advice=advice,
            sector_recommendations=sectors,
            weather_strategies=weather,
            driver_number=req.driver_number,
            driver_name=driver_name,
            driver_embedding=driver_embedding,
        )
        
        return _to_plain({
            "recommendation": recommendation,
            "track_id": req.track_id,
            "driver_number": req.driver_number,
        })
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/ai/scenario-recommendations")
def get_scenario_ai_recommendations(req: ScenarioRequest):
    """Generate AI recommendations based on Scenario Lab changes."""
    analyzer.run_baseline()
    adjustments = [
        {
            "event_order": event.event_order,
            "changes": [
                change.model_dump(exclude_none=True)
                for change in event.changes
            ],
        }
        for event in req.adjustments
    ]
    
    try:
        result = scenario_generator.run_scenario(
            name=req.name,
            description=req.description,
            adjustments=adjustments,
        )
        
        # Get baseline standings for comparison
        baseline_standings = analyzer._baseline_cache.get("final_standings", [])
        if hasattr(baseline_standings, "to_dict"):
            baseline_standings = baseline_standings.to_dict(orient="records")
        
        scenario_standings = result.get("final_standings", [])
        if hasattr(scenario_standings, "to_dict"):
            scenario_standings = scenario_standings.to_dict(orient="records")
        
        # Get driver embeddings if target driver is specified
        driver_embeddings_list = None
        if req.target_driver_number:
            driver_embeddings_list = build_driver_embeddings_summary(driver_embeddings_df)
        
        recommendation = generate_scenario_recommendations(
            scenario_name=req.name,
            scenario_description=req.description,
            adjustments=adjustments,
            baseline_standings=baseline_standings[:10] if baseline_standings else [],
            scenario_standings=scenario_standings[:10] if scenario_standings else [],
            target_driver_number=req.target_driver_number,
            driver_embeddings=driver_embeddings_list,
        )
        
        return _to_plain({
            "recommendation": recommendation,
            "scenario_result": result,
        })
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/ai/combined-recommendations")
def get_combined_ai_recommendations(req: AIRecommendationRequest):
    """Generate AI recommendations combining Coach Lab and Scenario Lab insights."""
    try:
        recommendation = generate_combined_recommendations(
            coach_insights=req.coach_insights,
            scenario_changes=req.scenario_changes,
        )
        return _to_plain({
            "recommendation": recommendation,
        })
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/chatbot/ask")
def ask_chatbot(req: ChatbotRequest):
    """Ask Shu Todoroki chatbot a question about the project."""
    try:
        # Convert conversation history to the format expected by the service
        history = None
        if req.conversation_history:
            history = [
                {"role": msg.role, "content": msg.content}
                for msg in req.conversation_history
            ]
        
        response = generate_chatbot_response(req.question, history)
        return {
            "response": response,
            "assistant": "Shu Todoroki"
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class SimulationRequest(BaseModel):
    iterations: Optional[int] = 200
    include_advanced: bool = True
    random_seed: Optional[int] = None
    run_monte_carlo: bool = True


@app.post("/api/simulation/run")
def run_simulation(req: SimulationRequest):
    """Run a championship simulation on-demand with butterfly effect analysis."""
    try:
        # Default to cached seed (42) if not specified
        single_seed = req.random_seed if req.random_seed is not None else 42
        # For Monte Carlo, use a different seed (99) if not specified
        mc_seed = req.random_seed if req.random_seed is not None else 99
        
        # Run single season simulation
        single_run = simulator.simulate_season(
            include_advanced=req.include_advanced,
            random_seed=single_seed
        )
        
        result = {
            "single_run": {
                "final_standings": single_run["final_standings"].to_dict(orient="records"),
                "race_results": single_run["race_results"].to_dict(orient="records"),
            }
        }
        
        # Run Monte Carlo if requested
        if req.run_monte_carlo:
            monte_carlo = simulator.run_monte_carlo(
                iterations=req.iterations or 200,
                include_advanced=req.include_advanced,
                random_seed=mc_seed
            )
            result["monte_carlo"] = monte_carlo.to_dict(orient="records")
        
        # Run butterfly effect analysis
        butterfly_analyzer = ButterflyEffectAnalyzer(
            simulator, 
            include_advanced=req.include_advanced, 
            random_seed=single_seed
        )
        butterfly_analyzer._baseline_cache = single_run  # Reuse simulation result
        top_impacts = butterfly_analyzer.rank_event_impacts(max_events=5)
        
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
        
        result["impact_reports"] = [_report_to_dict(r) for r in top_impacts]
        
        return _to_plain(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(exc)}")


@app.get("/api/simulation/quick")
def run_quick_simulation():
    """Run a quick simulation (single run + 100 Monte Carlo iterations + butterfly effect)."""
    try:
        # Use same seed as cached data for consistency
        single_run = simulator.simulate_season(include_advanced=True, random_seed=42)
        monte_carlo = simulator.run_monte_carlo(iterations=100, include_advanced=True, random_seed=99)
        
        # Run butterfly effect analysis
        butterfly_analyzer = ButterflyEffectAnalyzer(
            simulator, 
            include_advanced=True, 
            random_seed=42
        )
        butterfly_analyzer._baseline_cache = single_run  # Reuse simulation result
        top_impacts = butterfly_analyzer.rank_event_impacts(max_events=5)
        
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
        
        return _to_plain({
            "single_run": {
                "final_standings": single_run["final_standings"].to_dict(orient="records"),
                "race_results": single_run["race_results"].to_dict(orient="records"),
            },
            "monte_carlo": monte_carlo.to_dict(orient="records"),
            "impact_reports": [_report_to_dict(r) for r in top_impacts],
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Quick simulation failed: {str(exc)}")


def _process_uploaded_telemetry(filepath: Path) -> pd.DataFrame:
    """
    Process uploaded telemetry CSV file into per-lap features.
    Handles both long-format (parameter_name/value) and wide-format data.
    """
    # Read CSV
    df = pd.read_csv(filepath, low_memory=False)
    
    # Check if data is in long format (parameter_name, value)
    if "parameter_name" in df.columns and "value" in df.columns:
        # Find vehicle and lap columns
        vehicle_col = None
        for col in ["vehicle_id", "original_vehicle_id", "vehicle"]:
            if col in df.columns:
                vehicle_col = col
                break
        
        if not vehicle_col or "lap" not in df.columns:
            raise ValueError("Missing required columns: vehicle_id and lap")
        
        # Rename for consistency
        if vehicle_col != "vehicle_id":
            df = df.rename(columns={vehicle_col: "vehicle_id"})
        
        # Convert value to numeric
        df["value_numeric"] = pd.to_numeric(df["value"], errors="coerce")
        
        # Aggregate per lap and parameter
        per_lap_features = df.groupby(["vehicle_id", "lap", "parameter_name"]).agg({
            "value_numeric": ["mean", "max", "min", "std"]
        }).reset_index()
        
        # Flatten MultiIndex columns
        per_lap_features.columns = [
            col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
            for col in per_lap_features.columns.values
        ]
        
        # Pivot to wide format
        pivot_cols = [c for c in per_lap_features.columns if c not in ["vehicle_id", "lap", "parameter_name"]]
        per_lap_wide = per_lap_features.pivot_table(
            index=["vehicle_id", "lap"],
            columns="parameter_name",
            values=pivot_cols,
            aggfunc="first"
        ).reset_index()
        
        # Flatten column names
        per_lap_wide.columns = [
            f"{col[1]}_{col[0]}" if isinstance(col, tuple) and len(col) == 2 and col[0] != "" and col[1] != ""
            else col[0] if isinstance(col, tuple) and len(col) == 2 and col[1] != ""
            else str(col)
            for col in per_lap_wide.columns.values
        ]
        
        # Map parameter names to karma feature names
        column_mapping = {}
        for col in per_lap_wide.columns:
            col_lower = col.lower()
            if "speed" in col_lower and "mean" in col_lower:
                column_mapping[col] = "speed_mean"
            elif "speed" in col_lower and "max" in col_lower:
                column_mapping[col] = "speed_max"
            elif "nmot" in col_lower or "rpm" in col_lower:
                if "mean" in col_lower:
                    column_mapping[col] = "nmot_mean"
            elif "gear" in col_lower and "mean" in col_lower:
                column_mapping[col] = "gear_mean"
            elif "accx" in col_lower or "acceleration_x" in col_lower:
                if "std" in col_lower:
                    column_mapping[col] = "accx_can_std"
            elif "pbrake_f" in col_lower or "brake_front" in col_lower:
                if "max" in col_lower:
                    column_mapping[col] = "pbrake_f_max"
            elif "pbrake_r" in col_lower or "brake_rear" in col_lower:
                if "max" in col_lower:
                    column_mapping[col] = "pbrake_r_max"
            elif "steering" in col_lower:
                if "std" in col_lower:
                    column_mapping[col] = "Steering_Angle_std"
        
        per_lap_wide = per_lap_wide.rename(columns=column_mapping)
        
    else:
        # Data is already in wide format
        per_lap_wide = df.copy()
        
        # Ensure vehicle_id and lap columns exist
        if "vehicle_id" not in per_lap_wide.columns:
            vehicle_cols = [c for c in per_lap_wide.columns if "vehicle" in c.lower()]
            if vehicle_cols:
                per_lap_wide = per_lap_wide.rename(columns={vehicle_cols[0]: "vehicle_id"})
            else:
                raise ValueError("Could not find vehicle identifier column")
        
        if "lap" not in per_lap_wide.columns:
            lap_cols = [c for c in per_lap_wide.columns if "lap" in c.lower()]
            if lap_cols:
                per_lap_wide = per_lap_wide.rename(columns={lap_cols[0]: "lap"})
            else:
                raise ValueError("Could not find lap column")
    
    # Ensure required columns exist
    required_features = set().union(*(spec.feature_weights.keys() for spec in COMPONENT_SPECS))
    for feature in required_features:
        if feature not in per_lap_wide.columns:
            per_lap_wide[feature] = 0.0
    
    return per_lap_wide


def _prepare_per_lap_features_for_karma(loader: DataLoader) -> pd.DataFrame:
    """
    Prepare per-lap features from available data sources for karma computation.
    Attempts to aggregate telemetry data or use analysis data if available.
    """
    import numpy as np
    
    # Try to get telemetry data from all venues and aggregate per lap
    all_per_lap = []
    
    for venue in loader.venues:
        for race in ["Race 1", "Race 2"]:
            try:
                # Try to load telemetry
                telemetry = loader.load_telemetry(venue, race, sample_size=50000)
                
                if telemetry.empty:
                    continue
                
                # Check if telemetry is in key-value format
                if "parameter_name" in telemetry.columns and "value" in telemetry.columns:
                    # Find vehicle identifier column
                    vehicle_col = None
                    for col in ["original_vehicle_id", "vehicle_id", "vehicle"]:
                        if col in telemetry.columns:
                            vehicle_col = col
                            break
                    
                    if not vehicle_col or "lap" not in telemetry.columns:
                        continue
                    
                    # Rename vehicle column for consistency
                    if vehicle_col != "vehicle_id":
                        telemetry = telemetry.rename(columns={vehicle_col: "vehicle_id"})
                    
                    # Convert value to numeric where possible
                    telemetry["value_numeric"] = pd.to_numeric(telemetry["value"], errors="coerce")
                    
                    # Filter out NaN values
                    telemetry_clean = telemetry[telemetry["value_numeric"].notna()].copy()
                    
                    if telemetry_clean.empty:
                        continue
                    
                    # Aggregate per lap and parameter
                    per_lap_features = telemetry_clean.groupby(["vehicle_id", "lap", "parameter_name"]).agg({
                        "value_numeric": ["mean", "max", "min", "std"]
                    }).reset_index()
                    
                    # Flatten MultiIndex columns
                    per_lap_features.columns = [
                        f"{col[0]}_{col[1]}" if col[1] != "" else col[0]
                        for col in per_lap_features.columns.values
                    ]
                    
                    # Pivot to wide format
                    try:
                        per_lap_wide = per_lap_features.pivot_table(
                            index=["vehicle_id", "lap"],
                            columns="parameter_name",
                            values="value_numeric_mean",  # Use mean as primary value
                            aggfunc="first"
                        ).reset_index()
                        
                        # Flatten column names
                        per_lap_wide.columns = [
                            str(col) if col not in ["vehicle_id", "lap"] else col
                            for col in per_lap_wide.columns.values
                        ]
                        
                        # Map parameter names to karma feature names
                        parameter_mapping = {
                            "speed": "speed_mean",
                            "nmot": "nmot_mean", 
                            "rpm": "nmot_mean",
                            "gear": "gear_mean",
                            "accx_can": "accx_can_mean",
                            "accx": "accx_can_mean",
                            "pbrake_f": "pbrake_f_max",
                            "pbrake_r": "pbrake_r_max",
                            "Steering_Angle": "Steering_Angle_mean",
                            "steering_angle": "Steering_Angle_mean",
                        }
                        
                        # Rename columns based on parameter mapping
                        rename_dict = {}
                        for col in per_lap_wide.columns:
                            if col not in ["vehicle_id", "lap"]:
                                col_lower = str(col).lower()
                                for param, karma_feature in parameter_mapping.items():
                                    if param.lower() in col_lower:
                                        rename_dict[col] = karma_feature
                                        break
                        
                        per_lap_wide = per_lap_wide.rename(columns=rename_dict)
                        
                        all_per_lap.append(per_lap_wide)
                    except Exception as pivot_error:
                        # If pivot fails, skip this venue/race
                        print(f"Pivot error for {venue} {race}: {pivot_error}")
                        continue
                    
            except Exception as e:
                # Continue if this venue/race fails
                continue
    
    if not all_per_lap:
        # Return empty dataframe with required structure
        return pd.DataFrame(columns=["vehicle_id", "lap"])
    
    # Combine all venues
    combined = pd.concat(all_per_lap, ignore_index=True)
    
    # Fill missing columns with defaults
    required_features = set().union(*(spec.feature_weights.keys() for spec in COMPONENT_SPECS))
    for feature in required_features:
        if feature not in combined.columns:
            combined[feature] = 0.0
    
    return combined


@app.post("/api/karma/upload")
async def upload_telemetry_file(file: UploadFile = File(...)):
    """Upload and process telemetry file."""
    global _uploaded_per_lap_data
    
    try:
        # Save uploaded file temporarily
        suffix = Path(file.filename).suffix if file.filename else ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        
        try:
            # Process the file
            logger.info(f"Processing uploaded file: {file.filename}")
            
            per_lap_df = _process_uploaded_telemetry(tmp_path)
            
            logger.info(f"Processed file: {len(per_lap_df)} rows, columns: {list(per_lap_df.columns)}")
            
            # Store in memory
            _uploaded_per_lap_data = per_lap_df
            
            # Get vehicle list
            vehicles = sorted(per_lap_df["vehicle_id"].unique().tolist()) if not per_lap_df.empty else []
            
            logger.info(f"Found {len(vehicles)} vehicles: {vehicles}")
            
            return _to_plain({
                "message": "File processed successfully",
                "filename": file.filename,
                "rows_processed": len(per_lap_df),
                "vehicles": vehicles,
                "total_laps": len(per_lap_df) if not per_lap_df.empty else 0,
                "min_lap": int(per_lap_df["lap"].min()) if not per_lap_df.empty and "lap" in per_lap_df.columns else 0,
                "max_lap": int(per_lap_df["lap"].max()) if not per_lap_df.empty and "lap" in per_lap_df.columns else 0,
                "columns": list(per_lap_df.columns) if not per_lap_df.empty else [],
            })
        finally:
            # Clean up temp file
            if tmp_path.exists():
                tmp_path.unlink()
                
    except Exception as exc:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(exc)}\n{traceback.format_exc()}"
        )


@app.get("/api/karma/components")
def get_karma_components():
    """Get information about karma components."""
    try:
        # Always return component specs (they're static)
        components = [
            {
                "name": spec.name,
                "description": spec.description,
                "features": list(spec.feature_weights.keys())
            }
            for spec in COMPONENT_SPECS
        ]
        return _to_plain({"components": components})
    except NameError as e:
        # COMPONENT_SPECS not imported
        raise HTTPException(
            status_code=500,
            detail=f"Component specs not available: {str(e)}. Check karma_stream import."
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading components: {str(exc)}"
        )


@app.get("/api/karma/vehicles")
def get_karma_vehicles():
    """Get list of vehicles with available karma data."""
    global _uploaded_per_lap_data
    
    try:
        # Use uploaded data if available, otherwise try to load from data sources
        if _uploaded_per_lap_data is not None and not _uploaded_per_lap_data.empty:
            per_lap_df = _uploaded_per_lap_data
        else:
            per_lap_df = _prepare_per_lap_features_for_karma(loader)
        
        if per_lap_df.empty or "vehicle_id" not in per_lap_df.columns:
            return {"vehicles": []}
        
        vehicles = sorted(per_lap_df["vehicle_id"].unique().tolist())
        return {"vehicles": vehicles}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/karma/{vehicle_id}")
def get_karma_scores(vehicle_id: str, max_lap: Optional[int] = Query(None, description="Maximum lap number to include")):
    """Get karma scores for a specific vehicle."""
    global _uploaded_per_lap_data
    
    try:
        # Use uploaded data if available, otherwise try to load from data sources
        if _uploaded_per_lap_data is not None and not _uploaded_per_lap_data.empty:
            per_lap_df = _uploaded_per_lap_data
        else:
            per_lap_df = _prepare_per_lap_features_for_karma(loader)
        
        if per_lap_df.empty:
            raise HTTPException(status_code=404, detail="No per-lap data available")
        
        # Filter by vehicle
        vehicle_df = per_lap_df[per_lap_df["vehicle_id"] == vehicle_id].copy()
        
        if vehicle_df.empty:
            raise HTTPException(status_code=404, detail=f"Vehicle {vehicle_id} not found")
        
        # Filter by max_lap if provided
        if max_lap is not None:
            vehicle_df = vehicle_df[vehicle_df["lap"] <= max_lap]
        
        if vehicle_df.empty:
            raise HTTPException(status_code=404, detail=f"No data for vehicle {vehicle_id}")
        
        # Debug: Log vehicle data info
        logger.info(f"Computing karma for vehicle {vehicle_id}: {len(vehicle_df)} rows, columns: {list(vehicle_df.columns)}")
        
        # Compute karma stream
        try:
            karma_df = compute_stream(vehicle_df)
            logger.info(f"Karma computation result: {len(karma_df)} rows, columns: {list(karma_df.columns) if not karma_df.empty else 'empty'}")
        except Exception as e:
            logger.exception(f"Error computing karma stream: {e}")
            raise HTTPException(status_code=500, detail=f"Karma computation failed: {str(e)}")
        
        if karma_df.empty:
            logger.warning(f"Karma computation returned empty result for vehicle {vehicle_id}")
            return {
                "vehicle_id": vehicle_id,
                "latest_scores": {},
                "time_series": [],
                "debug_info": {
                    "vehicle_rows": len(vehicle_df),
                    "available_columns": list(vehicle_df.columns),
                    "required_features": list(set().union(*(spec.feature_weights.keys() for spec in COMPONENT_SPECS)))
                }
            }
        
        # Get latest scores per component
        latest_scores = {}
        for component in karma_df["component"].unique():
            component_data = karma_df[karma_df["component"] == component]
            if not component_data.empty:
                latest = component_data.sort_values("lap").iloc[-1]
                latest_scores[component] = {
                    "score": float(latest["karma_score"]),
                    "lap": int(latest["lap"])
                }
        
        # Convert time series to records
        # Limit to reasonable size for frontend (sample if too large)
        max_time_series_rows = 10000  # Limit to 10k points for performance
        if len(karma_df) > max_time_series_rows:
            logger.info(f"Sampling karma data: {len(karma_df)} rows -> {max_time_series_rows} rows")
            # Sample evenly across laps
            karma_df_sampled = karma_df.groupby("lap").head(4)  # Max 4 components per lap
            if len(karma_df_sampled) > max_time_series_rows:
                # Further sample if still too large
                step = len(karma_df_sampled) // max_time_series_rows
                karma_df_sampled = karma_df_sampled.iloc[::max(1, step)]
            karma_data = karma_df_sampled.to_dict(orient="records")
        else:
            karma_data = karma_df.to_dict(orient="records")
        
        logger.info(f"Returning karma data: {len(latest_scores)} components, {len(karma_data)} time series points")
        
        return _to_plain({
            "vehicle_id": vehicle_id,
            "latest_scores": latest_scores,
            "time_series": karma_data
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

