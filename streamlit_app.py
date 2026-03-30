# ============================================================
# STREAMLIT DASHBOARD — Smart City Traffic Stress System
# ============================================================
# HOW TO RUN:
#   1. pip install streamlit requests plotly pandas
#   2. Make sure your FastAPI is running:
#      python -m uvicorn main:app --reload
#   3. In a NEW terminal run:
#      streamlit run streamlit_app.py
#   4. Browser opens at http://localhost:8501
# ============================================================

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── PAGE CONFIG ───────────────────────────────────────────────
# Must be the FIRST streamlit command
st.set_page_config(
    page_title="Smart Traffic Intelligence",
    page_icon="🚦",
    layout="wide",                    # use full screen width
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
# Streamlit lets you inject raw CSS to style everything
st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

    /* Dark background */
    .stApp {
        background-color: #0a0e1a;
        color: #e2e8f0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1224;
        border-right: 1px solid #1e2d4a;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #0d1224 0%, #111827 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 16px;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        color: #38bdf8 !important;
    }

    /* Stress level badges */
    .badge-high {
        background: #7f1d1d;
        color: #fca5a5;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 13px;
        border: 1px solid #ef4444;
    }
    .badge-medium {
        background: #78350f;
        color: #fcd34d;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 13px;
        border: 1px solid #f59e0b;
    }
    .badge-low {
        background: #064e3b;
        color: #6ee7b7;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 13px;
        border: 1px solid #10b981;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #0d1224 0%, #0f172a 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
    }

    /* Action box */
    .action-reroute {
        background: #1a0a0a;
        border-left: 4px solid #ef4444;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-family: 'JetBrains Mono', monospace;
    }
    .action-adjust {
        border-left: 4px solid #f59e0b;
        background: #1a1400;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-family: 'JetBrains Mono', monospace;
    }
    .action-ok {
        border-left: 4px solid #10b981;
        background: #001a0f;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Divider */
    hr {
        border-color: #1e2d4a;
    }

    /* Route path */
    .route-path {
        font-family: 'JetBrains Mono', monospace;
        font-size: 18px;
        color: #38bdf8;
        letter-spacing: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ── API CONFIG ────────────────────────────────────────────────
API_BASE = "https://smart-traffic-intelligence-system-api.onrender.com"

# ── HELPER FUNCTIONS ──────────────────────────────────────────

def api_get(endpoint):
    """Call a GET endpoint on your FastAPI."""
    try:
        response = requests.get(f"{API_BASE}{endpoint}", timeout=5)
        if response.status_code == 200:
            return response.json(), None
        return None, f"API Error {response.status_code}"
    except requests.exceptions.ConnectionError:
        return None, "❌ Cannot connect to API. Is FastAPI running? (python -m uvicorn main:app --reload)"
    except Exception as e:
        return None, str(e)

def api_post(endpoint, data):
    """Call a POST endpoint on your FastAPI."""
    try:
        response = requests.post(f"{API_BASE}{endpoint}", json=data, timeout=5)
        if response.status_code == 200:
            return response.json(), None
        return None, response.json().get("detail", "Unknown error")
    except requests.exceptions.ConnectionError:
        return None, "❌ Cannot connect to API. Is FastAPI running?"
    except Exception as e:
        return None, str(e)

def stress_badge(level):
    """Return colored HTML badge for stress level."""
    classes = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}
    return f'<span class="{classes.get(level, "badge-low")}">{level}</span>'

def stress_color(level):
    return {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}.get(level, "#94a3b8")

def stress_emoji(level):
    return {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(level, "⚪")


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 Smart Traffic\nIntelligence System")
    st.markdown("---")

    # API health check
    health, err = api_get("/health")
    if health:
        status = health.get("status", "unknown")
        model_loaded = health.get("model_loaded", False)
        st.success(f"API: {status.upper()}")
        if model_loaded:
            st.success("ML Model: Loaded ✅")
        else:
            st.warning("ML Model: Not loaded ⚠️")
    else:
        st.error(err)

    st.markdown("---")
    st.markdown("### Navigation")

    # Page selection
    page = st.radio(
        "Go to:",
        ["📊 Dashboard", "🔮 Predict Stress", "🗺️ Route Optimizer", "🏙️ Zone Status"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(f"**Last updated:** {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Refresh"):
        st.rerun()


# ============================================================
# PAGE 1: DASHBOARD OVERVIEW
# ============================================================
if page == "📊 Dashboard":

    st.markdown("# 📊 Traffic Stress Dashboard")
    st.markdown("Real-time overview of your Smart City Traffic Intelligence System")
    st.markdown("---")

    # Fetch zone data
    zones_data, err = api_get("/zones")

    if err:
        st.error(err)
    else:
        zones = zones_data.get("zones", [])
        total = zones_data.get("total_zones", 0)
        congested = zones_data.get("congested_zones", 0)
        safe = total - congested

        # ── KPI METRICS ROW ──────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="🏙️ Total Zones",
                value=total,
                help="Total monitored zones in the city"
            )
        with col2:
            st.metric(
                label="🔴 Congested Zones",
                value=congested,
                delta=f"{congested} need action",
                delta_color="inverse"
            )
        with col3:
            st.metric(
                label="🟢 Clear Zones",
                value=safe,
                delta="Normal flow",
                delta_color="normal"
            )
        with col4:
            congestion_pct = round((congested / total) * 100) if total > 0 else 0
            st.metric(
                label="📈 Congestion Rate",
                value=f"{congestion_pct}%",
                help="Percentage of zones with High stress"
            )

        st.markdown("---")

        # ── ZONE STRESS CHART ─────────────────────────────────
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("### Zone Stress Overview")

            zone_ids = [z["zone_id"] for z in zones]
            stress_levels = [z["stress_level"] for z in zones]
            colors = [stress_color(s) for s in stress_levels]

            # Assign numeric stress for bar height
            stress_num = {"Low": 1, "Medium": 2, "High": 3}
            stress_values = [stress_num[s] for s in stress_levels]

            fig = go.Figure(data=[
                go.Bar(
                    x=zone_ids,
                    y=stress_values,
                    marker_color=colors,
                    text=stress_levels,
                    textposition='outside',
                    hovertemplate="<b>%{x}</b><br>Stress: %{text}<extra></extra>"
                )
            ])

            fig.update_layout(
                plot_bgcolor='#0a0e1a',
                paper_bgcolor='#0a0e1a',
                font=dict(color='#e2e8f0'),
                yaxis=dict(
                    tickvals=[1, 2, 3],
                    ticktext=["Low", "Medium", "High"],
                    gridcolor='#1e2d4a'
                ),
                xaxis=dict(gridcolor='#1e2d4a'),
                showlegend=False,
                height=350,
                margin=dict(t=20, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("### Stress Distribution")

            level_counts = {"Low": 0, "Medium": 0, "High": 0}
            for z in zones:
                level_counts[z["stress_level"]] += 1

            fig_pie = go.Figure(data=[go.Pie(
                labels=list(level_counts.keys()),
                values=list(level_counts.values()),
                hole=0.5,
                marker=dict(colors=["#10b981", "#f59e0b", "#ef4444"]),
                textfont=dict(color='white', size=14)
            )])

            fig_pie.update_layout(
                plot_bgcolor='#0a0e1a',
                paper_bgcolor='#0a0e1a',
                font=dict(color='#e2e8f0'),
                showlegend=True,
                legend=dict(font=dict(color='#e2e8f0')),
                height=350,
                margin=dict(t=20, b=20)
            )

            st.plotly_chart(fig_pie, use_container_width=True)

        # ── ZONE TABLE ────────────────────────────────────────
        st.markdown("### Zone Details")

        for zone in zones:
            col_a, col_b, col_c, col_d = st.columns([1, 2, 2, 3])
            with col_a:
                st.markdown(f"**{zone['zone_id']}**")
            with col_b:
                st.markdown(
                    stress_badge(zone['stress_level']),
                    unsafe_allow_html=True
                )
            with col_c:
                congested_text = "🚨 Congested" if zone['is_congested'] else "✅ Clear"
                st.markdown(congested_text)
            with col_d:
                neighbors = ", ".join(zone.get("neighbors", []))
                st.markdown(f"Connects to: `{neighbors}`" if neighbors else "No neighbors")


# ============================================================
# PAGE 2: PREDICT STRESS
# ============================================================
elif page == "🔮 Predict Stress":

    st.markdown("# 🔮 Predict Traffic Stress")
    st.markdown("Enter road conditions to get an instant ML prediction")
    st.markdown("---")

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("### Input Parameters")

        traffic_density = st.slider(
            "🚗 Traffic Density",
            min_value=0.0, max_value=100.0, value=65.0, step=0.5,
            help="Number of vehicles per unit area"
        )
        signal_wait_time = st.slider(
            "⏱️ Signal Wait Time (seconds)",
            min_value=0.0, max_value=180.0, value=45.0, step=1.0
        )
        avg_speed = st.slider(
            "💨 Average Speed (km/h)",
            min_value=0.0, max_value=120.0, value=25.0, step=0.5
        )
        horn_events = st.slider(
            "📯 Horn Events per Minute",
            min_value=0.0, max_value=50.0, value=12.0, step=0.5
        )
        road_quality = st.slider(
            "🛣️ Road Quality Score (0-10)",
            min_value=0.0, max_value=10.0, value=6.5, step=0.1
        )

        col_x, col_y = st.columns(2)
        with col_x:
            driver_exp = st.selectbox(
                "👤 Driver Experience",
                ["Beginner", "Intermediate", "Expert"]
            )
        with col_y:
            weather = st.selectbox(
                "🌤️ Weather",
                ["Clear", "Rain", "Fog", "Storm"]
            )

        zone_id = st.selectbox(
            "📍 Zone (for routing)",
            ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]
        )

        predict_btn = st.button("🔮 Predict Stress", type="primary", use_container_width=True)

    with col_result:
        st.markdown("### Prediction Result")

        if predict_btn:
            payload = {
                "traffic_density": traffic_density,
                "signal_wait_time": signal_wait_time,
                "avg_speed": avg_speed,
                "horn_events_per_min": horn_events,
                "road_quality_score": road_quality,
                "driver_experience_level": driver_exp,
                "weather_condition": weather,
                "zone_id": zone_id
            }

            with st.spinner("Running ML model..."):
                result, err = api_post("/predict", payload)

            if err:
                st.error(f"Error: {err}")
            else:
                level = result["stress_level"]
                stress_val = result["predicted_stress_index"]
                action = result["action"]

                # Big stress index gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=stress_val,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Stress Index", 'font': {'color': '#e2e8f0', 'size': 16}},
                    number={'font': {'color': stress_color(level), 'size': 48}},
                    gauge={
                        'axis': {'range': [0, 10], 'tickcolor': '#e2e8f0'},
                        'bar': {'color': stress_color(level)},
                        'bgcolor': '#0d1224',
                        'bordercolor': '#1e3a5f',
                        'steps': [
                            {'range': [0, 3.5], 'color': '#064e3b'},
                            {'range': [3.5, 6.5], 'color': '#78350f'},
                            {'range': [6.5, 10], 'color': '#7f1d1d'}
                        ],
                        'threshold': {
                            'line': {'color': stress_color(level), 'width': 4},
                            'thickness': 0.75,
                            'value': stress_val
                        }
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor='#0a0e1a',
                    font=dict(color='#e2e8f0'),
                    height=280,
                    margin=dict(t=30, b=10)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Stress badge
                st.markdown(
                    f"**Stress Level:** {stress_badge(level)}",
                    unsafe_allow_html=True
                )

                # Action box
                action_class = {
                    "High": "action-reroute",
                    "Medium": "action-adjust",
                    "Low": "action-ok"
                }.get(level, "action-ok")

                st.markdown(f"""
                <div class="{action_class}">
                    <strong>Action:</strong> {action}<br>
                    <strong>Signal:</strong> {result['signal_action']}
                </div>
                """, unsafe_allow_html=True)

                # Reroute path
                if result.get("reroute_path"):
                    st.markdown("**🗺️ Suggested Reroute:**")
                    path_str = " → ".join(result["reroute_path"])
                    st.markdown(
                        f'<div class="route-path">🚗 {path_str}</div>',
                        unsafe_allow_html=True
                    )
                    st.caption(f"Route cost: {result['reroute_cost']}")

                # Congestion status
                if result["is_congested"]:
                    st.error("🚨 CONGESTION DETECTED — Immediate action required")
                else:
                    st.success("✅ Traffic flowing normally")

        else:
            # Placeholder before prediction
            st.info("👈 Set parameters and click **Predict Stress** to get results")

            # Show example values
            st.markdown("**Example scenarios:**")
            st.markdown("""
            | Scenario | Density | Speed | Expected |
            |---|---|---|---|
            | Rush Hour | 85 | 10 | 🔴 High |
            | Normal Day | 50 | 40 | 🟡 Medium |
            | Late Night | 10 | 80 | 🟢 Low |
            """)


# ============================================================
# PAGE 3: ROUTE OPTIMIZER
# ============================================================
elif page == "🗺️ Route Optimizer":

    st.markdown("# 🗺️ Route Optimizer")
    st.markdown("Find the least-stressed route between zones using Dijkstra's Algorithm")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Select Route")
        start_zone = st.selectbox("🟢 Start Zone", ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"], index=0)
        end_zone = st.selectbox("🔴 End Zone", ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"], index=5)
        route_btn = st.button("🔍 Find Optimal Route", type="primary", use_container_width=True)

    with col2:
        st.markdown("### How it works")
        st.markdown("""
        The route optimizer uses **Dijkstra's Algorithm** with stress-weighted edges:

        | Stress Level | Weight Multiplier |
        |---|---|
        | 🟢 Low | 1.0x (normal) |
        | 🟡 Medium | 1.5x (slower) |
        | 🔴 High | 3.0x (avoid) |

        High stress zones are penalized heavily so the algorithm naturally avoids congested areas.
        """)

    if route_btn:
        if start_zone == end_zone:
            st.warning("Start and end zone cannot be the same!")
        else:
            with st.spinner("Running Dijkstra's Algorithm..."):
                result, err = api_post("/route", {
                    "start_zone": start_zone,
                    "end_zone": end_zone
                })

            if err:
                st.error(f"Error: {err}")
            else:
                st.markdown("---")
                st.markdown("### Optimal Route Found")

                path = result["optimal_path"]
                cost = result["total_cost"]
                zones_stress = result["zones_stress"]

                # Route display
                path_display = " → ".join([
                    f"{stress_emoji(zones_stress.get(z, 'Low'))} {z}"
                    for z in path
                ])

                st.markdown(
                    f'<div class="route-path">{path_display}</div>',
                    unsafe_allow_html=True
                )

                st.markdown(f"**Total Route Cost:** `{cost}`")
                st.markdown(f"**Zones Traversed:** `{len(path)}`")

                # Zone stress along route
                st.markdown("### Zone Stress Along Route")

                cols = st.columns(len(path))
                for i, (col, zone) in enumerate(zip(cols, path)):
                    with col:
                        stress = zones_stress.get(zone, "Unknown")
                        st.markdown(f"**{zone}**")
                        st.markdown(
                            stress_badge(stress),
                            unsafe_allow_html=True
                        )
                        if i < len(path) - 1:
                            pass

                # Route visualization as network graph
                st.markdown("### Route Map")

                # Node positions (fixed layout for Z1-Z6)
                pos = {
                    'Z1': (0, 1), 'Z2': (1, 2), 'Z3': (1, 0),
                    'Z4': (2, 2), 'Z5': (2, 0), 'Z6': (3, 1)
                }

                edge_x, edge_y = [], []
                edges = [('Z1','Z2'), ('Z1','Z3'), ('Z2','Z4'),
                         ('Z2','Z5'), ('Z3','Z5'), ('Z4','Z6'), ('Z5','Z6')]

                for e0, e1 in edges:
                    x0, y0 = pos[e0]
                    x1, y1 = pos[e1]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]

                # Highlight path edges
                path_edge_x, path_edge_y = [], []
                for i in range(len(path) - 1):
                    x0, y0 = pos[path[i]]
                    x1, y1 = pos[path[i+1]]
                    path_edge_x += [x0, x1, None]
                    path_edge_y += [y0, y1, None]

                all_zones_list = list(pos.keys())
                node_colors = [stress_color(zones_stress.get(z, "Low")) for z in all_zones_list]
                node_x = [pos[z][0] for z in all_zones_list]
                node_y = [pos[z][1] for z in all_zones_list]

                fig_net = go.Figure()

                # All edges (gray)
                fig_net.add_trace(go.Scatter(
                    x=edge_x, y=edge_y, mode='lines',
                    line=dict(color='#1e3a5f', width=2),
                    hoverinfo='none', showlegend=False
                ))

                # Path edges (highlighted)
                fig_net.add_trace(go.Scatter(
                    x=path_edge_x, y=path_edge_y, mode='lines',
                    line=dict(color='#38bdf8', width=4),
                    hoverinfo='none', showlegend=False
                ))

                # Nodes
                fig_net.add_trace(go.Scatter(
                    x=node_x, y=node_y, mode='markers+text',
                    marker=dict(size=40, color=node_colors, line=dict(width=2, color='#e2e8f0')),
                    text=all_zones_list,
                    textposition='middle center',
                    textfont=dict(color='white', size=13, family='monospace'),
                    hovertemplate="<b>%{text}</b><extra></extra>",
                    showlegend=False
                ))

                fig_net.update_layout(
                    plot_bgcolor='#0a0e1a',
                    paper_bgcolor='#0a0e1a',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=350,
                    margin=dict(t=20, b=20, l=20, r=20)
                )

                st.plotly_chart(fig_net, use_container_width=True)
                st.caption("🔵 Blue path = optimal route | Node color = stress level")


# ============================================================
# PAGE 4: ZONE STATUS
# ============================================================
elif page == "🏙️ Zone Status":

    st.markdown("# 🏙️ Zone Status Monitor")
    st.markdown("Live status of all city zones")
    st.markdown("---")

    zones_data, err = api_get("/zones")

    if err:
        st.error(err)
    else:
        zones = zones_data.get("zones", [])

        for zone in zones:
            with st.expander(
                f"{stress_emoji(zone['stress_level'])} Zone {zone['zone_id']} — {zone['stress_level']} Stress",
                expanded=zone["is_congested"]   # auto-expand congested zones
            ):
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown("**Stress Level**")
                    st.markdown(stress_badge(zone['stress_level']), unsafe_allow_html=True)

                with c2:
                    st.markdown("**Traffic Status**")
                    if zone["is_congested"]:
                        st.error("🚨 CONGESTED")
                    else:
                        st.success("✅ CLEAR")

                with c3:
                    st.markdown("**Connected Zones**")
                    neighbors = zone.get("neighbors", [])
                    if neighbors:
                        st.code(" | ".join(neighbors))
                    else:
                        st.caption("No outgoing connections")

                # Fetch individual zone detail
                zone_detail, _ = api_get(f"/zones/{zone['zone_id']}")
                if zone_detail:
                    action = zone_detail.get("recommended_action", "")
                    action_class = {
                        "REROUTE": "action-reroute",
                        "ADJUST SIGNALS": "action-adjust",
                        "NO ACTION": "action-ok"
                    }.get(action, "action-ok")

                    st.markdown(
                        f'<div class="{action_class}"><strong>Recommended Action:</strong> {action}</div>',
                        unsafe_allow_html=True
                    )