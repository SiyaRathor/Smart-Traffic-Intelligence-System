import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import heapq
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ============================================================
# CONFIG
# ============================================================
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
df = pd.read_csv('smart_city_traffic_stress_dataset.csv')

print("=" * 60)
print("STEP 1: DATA OVERVIEW")
print("=" * 60)
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())

# ============================================================
# STEP 2: EDA — EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: EDA")
print("=" * 60)

sns.set_theme(style="darkgrid", palette="muted")

# --- 2a. Target Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Stress Index Distribution", fontsize=16, fontweight='bold')

sns.histplot(df['stress_index'], bins=30, kde=True, ax=axes[0], color='steelblue')
axes[0].set_title("Histogram + KDE")
axes[0].set_xlabel("Stress Index")

sns.boxplot(x=df['stress_index'], ax=axes[1], color='lightcoral')
axes[1].set_title("Boxplot — Outlier Check")

plt.tight_layout()
plt.savefig("outputs/01_stress_distribution.png", dpi=150)
plt.show()

# --- 2b. Correlation Heatmap ---
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

plt.figure(figsize=(12, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0,
    linewidths=0.5, square=True
)
plt.title("Feature Correlation Heatmap", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/02_correlation_heatmap.png", dpi=150)
plt.show()

# Print top correlations with target
print("\nTop correlations with stress_index:")
print(corr['stress_index'].sort_values(ascending=False).to_string())

# --- 2c. Key Feature vs Stress ---
key_features = ['traffic_density', 'signal_wait_time', 'avg_speed', 'horn_events_per_min']
# Only keep features that exist in dataset
key_features = [f for f in key_features if f in df.columns]

if key_features:
    fig, axes = plt.subplots(1, len(key_features), figsize=(5 * len(key_features), 5))
    if len(key_features) == 1:
        axes = [axes]
    fig.suptitle("Key Features vs Stress Index", fontsize=15, fontweight='bold')

    for ax, feat in zip(axes, key_features):
        ax.scatter(df[feat], df['stress_index'], alpha=0.3, color='steelblue', s=15)
        ax.set_xlabel(feat)
        ax.set_ylabel("stress_index")
        ax.set_title(feat)

    plt.tight_layout()
    plt.savefig("outputs/03_features_vs_stress.png", dpi=150)
    plt.show()

# --- 2d. Categorical Analysis ---
cat_cols = df.select_dtypes(include='object').columns.tolist()

if cat_cols:
    fig, axes = plt.subplots(1, len(cat_cols), figsize=(6 * len(cat_cols), 5))
    if len(cat_cols) == 1:
        axes = [axes]
    fig.suptitle("Stress Index by Category", fontsize=15, fontweight='bold')

    for ax, col in zip(axes, cat_cols):
        order = df.groupby(col)['stress_index'].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x=col, y='stress_index', order=order, ax=ax, palette='Set2')
        ax.set_title(f"Stress by {col}")
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig("outputs/04_categorical_stress.png", dpi=150)
    plt.show()

# --- 2e. Outlier Detection ---
z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())
outlier_counts = (z_scores > 3).sum()
print("\nOutlier counts per feature (|z| > 3):")
print(outlier_counts[outlier_counts > 0].to_string())

# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 60)

if 'traffic_density' in df.columns and 'signal_wait_time' in df.columns:
    df['congestion_level'] = df['traffic_density'] * df['signal_wait_time']

if 'horn_events_per_min' in df.columns and 'signal_wait_time' in df.columns:
    df['frustration_index'] = df['horn_events_per_min'] * df['signal_wait_time']

if 'avg_speed' in df.columns and 'traffic_density' in df.columns:
    df['speed_efficiency'] = df['avg_speed'] / (df['traffic_density'] + 1)

if 'road_quality_score' in df.columns and 'traffic_density' in df.columns:
    df['road_impact'] = df['road_quality_score'] * df['traffic_density']

# Encode categoricals
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ============================================================
# STEP 4: STRESS THRESHOLDS — DATA-DRIVEN (not arbitrary)
# ============================================================
low_thresh = df['stress_index'].quantile(0.33)
high_thresh = df['stress_index'].quantile(0.66)

print(f"\nData-driven stress thresholds:")
print(f"  Low   → stress_index ≤ {low_thresh:.2f}")
print(f"  Medium → {low_thresh:.2f} < stress_index ≤ {high_thresh:.2f}")
print(f"  High  → stress_index > {high_thresh:.2f}")

def get_stress_level(x):
    if x <= low_thresh:
        return "Low"
    elif x <= high_thresh:
        return "Medium"
    else:
        return "High"

# ============================================================
# STEP 5: MODEL TRAINING
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: MODEL TRAINING")
print("=" * 60)

X = df_encoded.drop('stress_index', axis=1)
y = df_encoded['stress_index']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

model_LR = LinearRegression()
cv_lr = cross_val_score(model_LR, X_train, y_train, cv=kf, scoring='r2')
print(f"Linear Regression CV R²: {cv_lr.mean():.4f} ± {cv_lr.std():.4f}")

model_RF = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
cv_rf = cross_val_score(model_RF, X_train, y_train, cv=kf, scoring='r2')
print(f"Random Forest     CV R²: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")

model_XGB = XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
)
cv_xgb = cross_val_score(model_XGB, X_train, y_train, cv=kf, scoring='r2')
print(f"XGBoost           CV R²: {cv_xgb.mean():.4f} ± {cv_xgb.std():.4f}")

# Final training
model_XGB.fit(X_train, y_train)
y_pred = model_XGB.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nFinal XGBoost Test → MSE: {mse:.4f} | R²: {r2:.4f}")

# Save model
joblib.dump(model_XGB, "models/xgb_model.pkl")
joblib.dump(list(X.columns), "models/feature_columns.pkl")
print("Model saved to models/xgb_model.pkl")

# ============================================================
# STEP 6: BUILD REAL ZONE GRAPH FROM DATASET
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: ZONE GRAPH FROM DATASET")
print("=" * 60)

# Detect zone/location column dynamically
zone_col = None
for candidate in ['zone', 'location', 'intersection', 'area', 'road_id', 'segment_id', 'node']:
    if candidate in df.columns:
        zone_col = candidate
        break

if zone_col:
    print(f"Using column '{zone_col}' as zone identifier.")

    # Compute per-zone average stress
    zone_stress = df.groupby(zone_col)['stress_index'].mean()
    zones = zone_stress.index.tolist()

    print(f"Found {len(zones)} zones: {zones[:10]} {'...' if len(zones) > 10 else ''}")

    # Build graph: connect adjacent zones (sequential adjacency as proxy)
    # In a real deployment this would come from a road network map
    graph = {}
    for i, zone in enumerate(zones):
        graph[zone] = {}
        # Connect to next and previous zone as neighbors
        if i > 0:
            dist = round(np.random.uniform(1, 10), 1)  # base distance in km
            graph[zone][zones[i - 1]] = dist
        if i < len(zones) - 1:
            dist = round(np.random.uniform(1, 10), 1)
            graph[zone][zones[i + 1]] = dist

    node_stress_map = zone_stress.apply(get_stress_level).to_dict()

else:
    # Fallback: derive zones from dataset rows by clustering stress into buckets
    print("No zone column found. Deriving synthetic zones from dataset rows.")

    df['zone_id'] = pd.qcut(df.index, q=6, labels=['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6'])
    zone_col = 'zone_id'

    zone_stress = df.groupby(zone_col)['stress_index'].mean()
    zones = zone_stress.index.tolist()

    graph = {
        'Z1': {'Z2': 4.0, 'Z3': 2.5},
        'Z2': {'Z4': 5.0, 'Z5': 3.0},
        'Z3': {'Z5': 6.0},
        'Z4': {'Z6': 2.0},
        'Z5': {'Z6': 1.5},
        'Z6': {}
    }

    node_stress_map = zone_stress.apply(get_stress_level).to_dict()

print("\nZone → Stress Level:")
for z, s in node_stress_map.items():
    print(f"  {z}: {s} (avg stress: {zone_stress[z]:.2f})")

# ============================================================
# STEP 7: DIJKSTRA WITH REAL STRESS WEIGHTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: DIJKSTRA — STRESS-WEIGHTED ROUTING")
print("=" * 60)

def adjust_weight(base_weight, stress_level):
    """Higher stress = higher travel cost = avoid this zone."""
    multipliers = {"High": 3.0, "Medium": 1.5, "Low": 1.0}
    return base_weight * multipliers.get(stress_level, 1.0)

def build_adjusted_graph(graph, node_stress_map):
    adjusted = {}
    for node in graph:
        adjusted[node] = {}
        for neighbor, base_weight in graph[node].items():
            stress = node_stress_map.get(neighbor, "Low")
            adjusted[node][neighbor] = adjust_weight(base_weight, stress)
    return adjusted

def dijkstra(graph, start, end):
    pq = [(0, start)]
    visited = set()
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    parent = {}

    while pq:
        current_cost, current_node = heapq.heappop(pq)
        if current_node in visited:
            continue
        visited.add(current_node)

        for neighbor, weight in graph[current_node].items():
            new_cost = current_cost + weight
            if new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                parent[neighbor] = current_node
                heapq.heappush(pq, (new_cost, neighbor))

    path = []
    node = end
    while node in parent:
        path.append(node)
        node = parent[node]
    path.append(start)
    path.reverse()

    return path if path[0] == start else [], distances.get(end, float('inf'))

adjusted_graph = build_adjusted_graph(graph, node_stress_map)

# ============================================================
# STEP 8: DECISION ENGINE — CONNECTED TO DIJKSTRA
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: DECISION ENGINE")
print("=" * 60)

def traffic_decision(zone, stress_level, graph, node_stress_map, all_zones):
    """
    Real decision engine:
    - Low/Medium stress → no reroute needed
    - High stress → run Dijkstra to find least-stressed alternate path
    """
    decision = {
        "zone": zone,
        "stress_level": stress_level,
        "action": None,
        "reroute_path": None,
        "reroute_cost": None,
        "signal_action": None
    }

    if stress_level == "High":
        decision["action"] = "REROUTE + EXTEND SIGNAL"
        decision["signal_action"] = "Increase green signal time by 40%"

        # Find least stressed destination zone from all zones
        adjusted = build_adjusted_graph(graph, node_stress_map)
        best_path, best_cost = [], float('inf')

        for target in all_zones:
            if target != zone and target in adjusted:
                path, cost = dijkstra(adjusted, zone, target)
                if path and cost < best_cost:
                    best_cost = cost
                    best_path = path

        decision["reroute_path"] = best_path
        decision["reroute_cost"] = round(best_cost, 2)

    elif stress_level == "Medium":
        decision["action"] = "ADJUST SIGNALS"
        decision["signal_action"] = "Increase green signal time by 15%"

    else:
        decision["action"] = "NO ACTION"
        decision["signal_action"] = "Normal signal timing"

    return decision

# Run decisions for each zone
print("\nDecision Engine Output per Zone:")
print("-" * 60)

all_zones = list(graph.keys())

for zone in all_zones:
    stress = node_stress_map.get(zone, "Low")
    decision = traffic_decision(zone, stress, graph, node_stress_map, all_zones)

    print(f"\nZone: {zone} | Stress: {stress}")
    print(f"  Action       : {decision['action']}")
    print(f"  Signal       : {decision['signal_action']}")
    if decision['reroute_path']:
        print(f"  Reroute Path : {' → '.join(str(z) for z in decision['reroute_path'])}")
        print(f"  Reroute Cost : {decision['reroute_cost']}")

# ============================================================
# STEP 9: APPLY PREDICTIONS ON TEST SET
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: PREDICTIONS ON TEST SET")
print("=" * 60)

results = X_test.copy()
results['predicted_stress'] = y_pred
results['stress_level'] = results['predicted_stress'].apply(get_stress_level)
results['is_congested'] = results['stress_level'].apply(lambda x: 1 if x == "High" else 0)

print(results[['predicted_stress', 'stress_level', 'is_congested']].head(10))
print(f"\nCongestion rate: {results['is_congested'].mean() * 100:.1f}% of predictions are High stress")

# ============================================================
# STEP 10: FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: FEATURE IMPORTANCE")
print("=" * 60)

importances = model_XGB.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(importance_df.head(10).to_string())

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis')
plt.title("Top 10 Feature Importances — XGBoost", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/05_feature_importance.png", dpi=150)
plt.show()

# ============================================================
# STEP 11: MODEL COMPARISON CHART
# ============================================================
model_names = ['Linear Regression', 'Random Forest', 'XGBoost']
cv_means = [cv_lr.mean(), cv_rf.mean(), cv_xgb.mean()]
cv_stds = [cv_lr.std(), cv_rf.std(), cv_xgb.std()]

plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, cv_means, yerr=cv_stds, capsize=5,
               color=['#4C72B0', '#55A868', '#C44E52'], alpha=0.85)
plt.ylabel("CV R² Score")
plt.title("Model Comparison — 5-Fold Cross Validation R²", fontsize=13, fontweight='bold')
plt.ylim(0, 1.05)
for bar, val in zip(bars, cv_means):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{val:.3f}", ha='center', fontsize=11)
plt.tight_layout()
plt.savefig("outputs/06_model_comparison.png", dpi=150)
plt.show()

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)
print(f"Dataset shape      : {df.shape}")
print(f"Features used      : {X.shape[1]}")
print(f"Best model         : XGBoost")
print(f"Test R²            : {r2:.4f}")
print(f"Test MSE           : {mse:.4f}")
print(f"Stress thresholds  : Low ≤ {low_thresh:.2f} | Medium ≤ {high_thresh:.2f} | High > {high_thresh:.2f}")
print(f"Zones in graph     : {len(all_zones)}")
print(f"Model saved to     : models/xgb_model.pkl")
print(f"Plots saved to     : outputs/")
print("=" * 60)