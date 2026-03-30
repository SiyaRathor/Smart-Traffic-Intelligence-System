# ============================================================
# FASTAPI BACKEND — TRAFFIC STRESS DETECTION SYSTEM
# ============================================================
# HOW TO RUN THIS FILE:
#   1. Open terminal in this folder
#   2. pip install fastapi uvicorn joblib pandas numpy xgboost
#   3. uvicorn main:app --reload
#   4. Open browser → http://localhost:8000/docs  ← FREE Swagger UI
# ============================================================


# ── IMPORTS ──────────────────────────────────────────────────
# FastAPI is the framework. Think of it like Django but lighter and faster.
from fastapi import FastAPI, HTTPException

# Pydantic is for DATA VALIDATION.
# It checks that the data sent to your API is correct before processing.
# Example: if someone sends "abc" where a number is expected → automatic error
from pydantic import BaseModel, Field

# Standard libraries
import joblib          # Load your saved ML model (.pkl file)
import numpy as np
import pandas as pd
import heapq
from typing import Optional, List
import os


# ── CREATE THE APP ────────────────────────────────────────────
# This ONE line creates your entire API application.
# title, description, version → shown in Swagger UI docs
app = FastAPI(
    title="🚦 Traffic Stress Detection API",
    description="""
    Smart City Traffic Stress Prediction & Route Optimization System.
    
    ## What this API does:
    - **Predicts** traffic stress index for a given road/zone
    - **Classifies** stress as Low / Medium / High
    - **Decides** signal actions based on stress level
    - **Reroutes** traffic using Dijkstra's algorithm
    - **Returns** zone-level traffic dashboard data
    """,
    version="1.0.0"
)


# ── LOAD ML MODEL ─────────────────────────────────────────────
# We load the model ONCE when the server starts.
# This is important — you never want to reload a model on every request.
# That would be very slow.

MODEL_PATH = "models/xgb_model.pkl"
FEATURES_PATH = "models/feature_columns.pkl"

# We use a try-except because if the model file doesn't exist,
# we want a clear error message, not a crash.
try:
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("⚠️  Model not found. Run traffic_stress_project.py first to train and save the model.")
    model = None
    feature_columns = []


# ── PYDANTIC MODELS (Data Validation) ────────────────────────
# These are like "contracts" for your API.
# Anyone calling your API MUST send data in this exact format.
# FastAPI will automatically reject wrong data with a clear error.

# Field() lets you add:
#   description → shown in Swagger docs
#   ge (greater than or equal), le (less than or equal) → range validation
#   example → shown as sample value in Swagger docs

class TrafficInput(BaseModel):
    """
    Input data for predicting traffic stress.
    Every field here maps to a column in your dataset.
    """
    traffic_density: float = Field(
        ...,                          # ... means this field is REQUIRED
        ge=0, le=100,                 # must be between 0 and 100
        description="Number of vehicles per unit area",
        example=65.0
    )
    signal_wait_time: float = Field(
        ..., ge=0,
        description="Average wait time at signal in seconds",
        example=45.0
    )
    avg_speed: float = Field(
        ..., ge=0,
        description="Average vehicle speed in km/h",
        example=25.0
    )
    horn_events_per_min: float = Field(
        ..., ge=0,
        description="Number of horn events per minute",
        example=12.0
    )
    road_quality_score: float = Field(
        ..., ge=0, le=10,
        description="Road quality rating from 0 to 10",
        example=6.5
    )
    driver_experience_level: str = Field(
        ...,
        description="Driver experience: 'Beginner', 'Intermediate', 'Expert'",
        example="Intermediate"
    )
    weather_condition: str = Field(
        ...,
        description="Weather: 'Clear', 'Rain', 'Fog', 'Storm'",
        example="Clear"
    )
    zone_id: Optional[str] = Field(
        None,
        description="Zone identifier (optional, used for routing)",
        example="Z1"
    )


class RouteRequest(BaseModel):
    """Input for route optimization."""
    start_zone: str = Field(..., description="Starting zone ID", example="Z1")
    end_zone: str = Field(..., description="Destination zone ID", example="Z6")


class PredictionResponse(BaseModel):
    """What the API sends BACK after prediction."""
    predicted_stress_index: float
    stress_level: str          # "Low", "Medium", "High"
    is_congested: bool
    action: str                # What to do
    signal_action: str
    reroute_path: Optional[List[str]]   # Only present if High stress
    reroute_cost: Optional[float]


# ── HELPER FUNCTIONS ──────────────────────────────────────────

# Stress thresholds — ideally load these from your trained data
# In production, save these during training and load here
LOW_THRESH = 3.5
HIGH_THRESH = 6.5

def get_stress_level(stress_index: float) -> str:
    """Convert numeric stress to category."""
    if stress_index <= LOW_THRESH:
        return "Low"
    elif stress_index <= HIGH_THRESH:
        return "Medium"
    else:
        return "High"


# ── GRAPH & DIJKSTRA ──────────────────────────────────────────
# In production this would come from a database or map API.
# For now, this is your zone graph.

ZONE_GRAPH = {
    'Z1': {'Z2': 4.0, 'Z3': 2.5},
    'Z2': {'Z4': 5.0, 'Z5': 3.0},
    'Z3': {'Z5': 6.0},
    'Z4': {'Z6': 2.0},
    'Z5': {'Z6': 1.5},
    'Z6': {}
}

# Zone stress levels — in production, update this from live predictions
ZONE_STRESS = {
    'Z1': 'Low', 'Z2': 'High', 'Z3': 'Medium',
    'Z4': 'Low', 'Z5': 'High', 'Z6': 'Low'
}

def adjust_weight(base_weight: float, stress_level: str) -> float:
    multipliers = {"High": 3.0, "Medium": 1.5, "Low": 1.0}
    return base_weight * multipliers.get(stress_level, 1.0)

def build_adjusted_graph(graph: dict, stress_map: dict) -> dict:
    adjusted = {}
    for node in graph:
        adjusted[node] = {}
        for neighbor, weight in graph[node].items():
            stress = stress_map.get(neighbor, "Low")
            adjusted[node][neighbor] = adjust_weight(weight, stress)
    return adjusted

def dijkstra(graph: dict, start: str, end: str):
    if start not in graph or end not in graph:
        return [], float('inf')

    pq = [(0, start)]
    visited = set()
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    parent = {}

    while pq:
        cost, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        for neighbor, weight in graph[node].items():
            new_cost = cost + weight
            if new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                parent[neighbor] = node
                heapq.heappush(pq, (new_cost, neighbor))

    path, node = [], end
    while node in parent:
        path.append(node)
        node = parent[node]
    path.append(start)
    path.reverse()

    return (path if path[0] == start else []), distances.get(end, float('inf'))


def prepare_features(data: TrafficInput) -> pd.DataFrame:
    """
    Convert API input into the format the model expects.
    This mirrors your feature engineering from training.
    """
    row = {
        'traffic_density': data.traffic_density,
        'signal_wait_time': data.signal_wait_time,
        'avg_speed': data.avg_speed,
        'horn_events_per_min': data.horn_events_per_min,
        'road_quality_score': data.road_quality_score,
        # Engineered features
        'congestion_level': data.traffic_density * data.signal_wait_time,
        'frustration_index': data.horn_events_per_min * data.signal_wait_time,
        'speed_efficiency': data.avg_speed / (data.traffic_density + 1),
        'road_impact': data.road_quality_score * data.traffic_density,
    }

    df = pd.DataFrame([row])

    # One-hot encode categoricals — must match training encoding
    exp_dummies = pd.get_dummies(
        pd.Series([data.driver_experience_level], name='driver_experience_level')
    )
    weather_dummies = pd.get_dummies(
        pd.Series([data.weather_condition], name='weather_condition')
    )

    df = pd.concat([df, exp_dummies, weather_dummies], axis=1)

    # Align columns to match training — fill missing with 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]  # Reorder to exact training order
    return df


# ============================================================
# API ENDPOINTS
# ============================================================
# An "endpoint" is a URL that does something.
# @app.get("/url")  → for fetching data (READ)
# @app.post("/url") → for sending data (CREATE/PREDICT)
# Every function below = one endpoint in your API.


# ── ROOT ENDPOINT ─────────────────────────────────────────────
# This is just a welcome message.
# Visit http://localhost:8000/ to see it.
@app.get("/")
def root():
    """Welcome endpoint — checks if API is running."""
    return {
        "message": "🚦 Traffic Stress Detection API is running",
        "docs": "Visit /docs for Swagger UI",
        "version": "1.0.0"
    }


# ── HEALTH CHECK ──────────────────────────────────────────────
# Standard in production APIs.
# Used by deployment platforms (Docker, cloud) to check if app is alive.
@app.get("/health")
def health_check():
    """Check if model is loaded and API is healthy."""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "features_count": len(feature_columns)
    }


# ── PREDICT ENDPOINT ──────────────────────────────────────────
# This is the CORE of your API.
# Someone sends traffic data → you return stress prediction + decision.
#
# @app.post → because the user is SENDING data to us
# response_model → FastAPI validates our response matches PredictionResponse

@app.post("/predict", response_model=PredictionResponse)
def predict_stress(data: TrafficInput):
    """
    Predict traffic stress index and return intelligent decision.
    
    - Predicts stress level using XGBoost model
    - Classifies as Low / Medium / High
    - Returns signal action
    - If High stress: runs Dijkstra to find best reroute
    """

    # Guard: if model not loaded, return error
    # HTTPException is how FastAPI returns errors with proper HTTP status codes
    # 503 = Service Unavailable
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )

    try:
        # Step 1: Prepare features
        features_df = prepare_features(data)

        # Step 2: Predict
        predicted_stress = float(model.predict(features_df)[0])

        # Step 3: Classify
        stress_level = get_stress_level(predicted_stress)
        is_congested = stress_level == "High"

        # Step 4: Decision
        reroute_path = None
        reroute_cost = None

        if stress_level == "High":
            action = "REROUTE + EXTEND SIGNAL"
            signal_action = "Increase green signal time by 40%"

            # Run Dijkstra if zone provided
            if data.zone_id and data.zone_id in ZONE_GRAPH:
                adjusted = build_adjusted_graph(ZONE_GRAPH, ZONE_STRESS)
                # Find cheapest path to any other zone
                best_path, best_cost = [], float('inf')
                for target in ZONE_GRAPH:
                    if target != data.zone_id:
                        path, cost = dijkstra(adjusted, data.zone_id, target)
                        if path and cost < best_cost:
                            best_cost = cost
                            best_path = path
                reroute_path = best_path if best_path else None
                reroute_cost = round(best_cost, 2) if best_path else None

        elif stress_level == "Medium":
            action = "ADJUST SIGNALS"
            signal_action = "Increase green signal time by 15%"
        else:
            action = "NO ACTION"
            signal_action = "Normal signal timing"

        # Step 5: Return response
        return PredictionResponse(
            predicted_stress_index=round(predicted_stress, 4),
            stress_level=stress_level,
            is_congested=is_congested,
            action=action,
            signal_action=signal_action,
            reroute_path=reroute_path,
            reroute_cost=reroute_cost
        )

    # Catch any unexpected errors
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ── ROUTE OPTIMIZATION ENDPOINT ───────────────────────────────
@app.post("/route")
def optimize_route(request: RouteRequest):
    """
    Find the least-stressed route between two zones using Dijkstra.
    """
    if request.start_zone not in ZONE_GRAPH:
        raise HTTPException(status_code=404, detail=f"Zone '{request.start_zone}' not found")
    if request.end_zone not in ZONE_GRAPH:
        raise HTTPException(status_code=404, detail=f"Zone '{request.end_zone}' not found")

    adjusted = build_adjusted_graph(ZONE_GRAPH, ZONE_STRESS)
    path, cost = dijkstra(adjusted, request.start_zone, request.end_zone)

    if not path:
        raise HTTPException(status_code=404, detail="No route found between these zones")

    return {
        "start": request.start_zone,
        "end": request.end_zone,
        "optimal_path": path,
        "path_string": " → ".join(path),
        "total_cost": round(cost, 2),
        "zones_stress": {zone: ZONE_STRESS.get(zone, "Unknown") for zone in path}
    }


# ── ZONE STATUS ENDPOINT ──────────────────────────────────────
@app.get("/zones")
def get_all_zones():
    """Return current stress status of all zones — used by dashboard."""
    zones_data = []
    for zone, stress in ZONE_STRESS.items():
        zones_data.append({
            "zone_id": zone,
            "stress_level": stress,
            "is_congested": stress == "High",
            "neighbors": list(ZONE_GRAPH.get(zone, {}).keys())
        })
    return {
        "total_zones": len(zones_data),
        "congested_zones": sum(1 for z in zones_data if z["is_congested"]),
        "zones": zones_data
    }


@app.get("/zones/{zone_id}")
def get_zone(zone_id: str):
    """
    Get status of a specific zone.
    {zone_id} is a PATH PARAMETER — it comes from the URL.
    Example: GET /zones/Z1 → returns info about zone Z1
    """
    if zone_id not in ZONE_GRAPH:
        # 404 = Not Found
        raise HTTPException(status_code=404, detail=f"Zone '{zone_id}' not found")

    stress = ZONE_STRESS.get(zone_id, "Unknown")

    return {
        "zone_id": zone_id,
        "stress_level": stress,
        "is_congested": stress == "High",
        "neighbors": list(ZONE_GRAPH.get(zone_id, {}).keys()),
        "recommended_action": (
            "REROUTE" if stress == "High"
            else "ADJUST SIGNALS" if stress == "Medium"
            else "NO ACTION"
        )
    }


# ── BATCH PREDICT ENDPOINT ────────────────────────────────────
# Instead of predicting one at a time, predict for many zones at once.
@app.post("/predict/batch")
def batch_predict(data_list: List[TrafficInput]):
    """
    Predict stress for multiple zones at once.
    Useful for dashboard that needs to update all zones simultaneously.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(data_list) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 predictions per batch")

    results = []
    for i, data in enumerate(data_list):
        try:
            features_df = prepare_features(data)
            predicted_stress = float(model.predict(features_df)[0])
            stress_level = get_stress_level(predicted_stress)

            results.append({
                "index": i,
                "zone_id": data.zone_id,
                "predicted_stress_index": round(predicted_stress, 4),
                "stress_level": stress_level,
                "is_congested": stress_level == "High"
            })
        except Exception as e:
            results.append({"index": i, "error": str(e)})

    return {
        "total": len(results),
        "predictions": results
    }