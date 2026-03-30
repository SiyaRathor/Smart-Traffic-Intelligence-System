# 🚦 Smart City Traffic Stress Intelligence System

A full-stack ML system that predicts urban traffic stress, detects congestion, and intelligently reroutes traffic using machine learning and graph algorithms.

**Live Demo:** [Streamlit Dashboard](https://YOUR-APP.streamlit.app) | **API Docs:** [FastAPI Swagger](https://YOUR-API.railway.app/docs)

---

## 🏗️ System Architecture

```
Kaggle Dataset
     ↓
Data Pipeline (Pandas)
     ↓
Feature Engineering
     ↓
XGBoost ML Model ──→ Saved Model (.pkl)
     ↓                      ↓
Decision Engine      FastAPI Backend (REST API)
     ↓                      ↓
Dijkstra Routing     Streamlit Dashboard
```

---

## ✨ Features

- **ML Prediction** — XGBoost model predicts traffic stress index (R² > 0.90)
- **Stress Classification** — Data-driven Low / Medium / High thresholds using quantiles
- **Decision Engine** — Automatically recommends signal adjustments or rerouting
- **Route Optimization** — Dijkstra's algorithm with stress-weighted edges avoids congested zones
- **REST API** — 6 FastAPI endpoints with automatic Swagger documentation
- **Interactive Dashboard** — 4-page Streamlit dashboard with real-time charts

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data Pipeline | Python, Pandas, NumPy |
| ML Model | XGBoost, Scikit-learn |
| Experiment Tracking | Cross-validation, MLflow-ready |
| Backend API | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit, Plotly |
| Algorithm | Dijkstra's (heapq) |
| Deployment | Railway (API) + Streamlit Cloud |

---

## 📊 Model Performance

| Model | CV R² | Notes |
|---|---|---|
| Linear Regression | ~0.75 | Baseline |
| Random Forest | ~0.88 | Good |
| **XGBoost** | **~0.93** | **Best — deployed** |

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOURUSERNAME/Smart-Traffic-Intelligence-System.git
cd Smart-Traffic-Intelligence-System
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
pip install streamlit plotly requests
```

### 3. Train the model
```bash
python traffic_stress_project.py
```

### 4. Start the API
```bash
python -m uvicorn main:app --reload
```
API runs at `http://localhost:8000` | Swagger UI at `http://localhost:8000/docs`

### 5. Start the dashboard (new terminal)
```bash
streamlit run streamlit_app.py
```
Dashboard runs at `http://localhost:8501`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Model status |
| POST | `/predict` | Predict stress for a zone |
| POST | `/route` | Optimal route between zones |
| GET | `/zones` | All zone statuses |
| GET | `/zones/{zone_id}` | Single zone status |
| POST | `/predict/batch` | Batch predictions |

### Example Request
```bash
curl -X POST https://YOUR-API.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "traffic_density": 75.0,
    "signal_wait_time": 60.0,
    "avg_speed": 15.0,
    "horn_events_per_min": 20.0,
    "road_quality_score": 4.0,
    "driver_experience_level": "Beginner",
    "weather_condition": "Rain",
    "zone_id": "Z2"
  }'
```

### Example Response
```json
{
  "predicted_stress_index": 8.24,
  "stress_level": "High",
  "is_congested": true,
  "action": "REROUTE + EXTEND SIGNAL",
  "signal_action": "Increase green signal time by 40%",
  "reroute_path": ["Z2", "Z5", "Z6"],
  "reroute_cost": 4.5
}
```

---

## 📁 Project Structure

```
Smart_Traffic_Intelligence_System/
├── main.py                        # FastAPI backend
├── streamlit_app.py               # Streamlit dashboard
├── traffic_stress_project.py      # ML training pipeline
├── requirements.txt               # Backend dependencies
├── streamlit_requirements.txt     # Frontend dependencies
├── Procfile                       # Railway deployment config
├── runtime.txt                    # Python version
├── .streamlit/
│   └── config.toml                # Dashboard theme
├── models/
│   ├── xgb_model.pkl              # Trained XGBoost model
│   └── feature_columns.pkl        # Feature schema
└── outputs/
    ├── 01_stress_distribution.png
    ├── 02_correlation_heatmap.png
    ├── 03_features_vs_stress.png
    ├── 04_categorical_stress.png
    ├── 05_feature_importance.png
    └── 06_model_comparison.png
```

---

## 📈 Dashboard Pages

1. **📊 Dashboard** — KPI metrics, zone stress overview, distribution charts
2. **🔮 Predict Stress** — Real-time prediction with interactive sliders + gauge chart
3. **🗺️ Route Optimizer** — Dijkstra routing with network graph visualization
4. **🏙️ Zone Status** — Live zone monitoring with recommended actions

---

## 🧠 Key Technical Decisions

**Why XGBoost?** Best CV R² among tested models. Handles feature interactions well, faster than LSTM for tabular data, and natively supports feature importance.

**Why Dijkstra with stress weights?** Simple, interpretable, and efficient for small city graphs. Stress multipliers (Low=1x, Medium=1.5x, High=3x) naturally penalize congested routes without complex heuristics.

**Why FastAPI?** Automatic Swagger docs, Pydantic validation, async support, and significantly faster than Flask for ML serving.

---

## 👤 Author

Built by [Your Name] — [LinkedIn](https://linkedin.com) | [GitHub](https://github.com/YOURUSERNAME)