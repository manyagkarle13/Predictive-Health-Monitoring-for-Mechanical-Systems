"""
Train Model Script
Predictive Health Monitoring System
Remaining Useful Life (RUL) Estimation
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


# ==========================================================
# 1. LOAD DATA
# ==========================================================

datasets = ["FD001", "FD002", "FD003", "FD004"]

for dataset in datasets:
    print(f"\nTraining model for {dataset}...")

    file_path = f"../data/train_{dataset}.txt"
    df = pd.read_csv(file_path, sep=" ", header=None)
    df = df.iloc[:, :-2]

    columns = ["engine_id", "cycle"]
    for i in range(1, 4):
        columns.append(f"op_setting_{i}")
    for i in range(1, 22):
        columns.append(f"sensor_{i}")

    df.columns = columns

    max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycle.columns = ["engine_id", "max_cycle"]
    df = df.merge(max_cycle, on="engine_id", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df = df.drop(columns=["max_cycle"])

    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    df = df.drop(columns=constant_columns)

    engine_ids = df["engine_id"].unique()
    train_size = int(0.8 * len(engine_ids))
    train_engines = engine_ids[:train_size]
    test_engines = engine_ids[train_size:]

    train_df = df[df["engine_id"].isin(train_engines)]
    test_df = df[df["engine_id"].isin(test_engines)]

    X_train = train_df.drop(columns=["engine_id", "RUL"])
    y_train = train_df["RUL"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    joblib.dump(model, f"model/rul_model_{dataset}.pkl")
    print(f"{dataset} model saved.")
df = pd.read_csv(file_path, sep=" ", header=None)
df = df.iloc[:, :-2]


# ==========================================================
# 2. ADD COLUMN NAMES
# ==========================================================

columns = ["engine_id", "cycle"]

for i in range(1, 4):
    columns.append(f"op_setting_{i}")

for i in range(1, 22):
    columns.append(f"sensor_{i}")

df.columns = columns


# ==========================================================
# 3. CALCULATE RUL
# ==========================================================

max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
max_cycle.columns = ["engine_id", "max_cycle"]

df = df.merge(max_cycle, on="engine_id", how="left")
df["RUL"] = df["max_cycle"] - df["cycle"]
df = df.drop(columns=["max_cycle"])


# ==========================================================
# 4. REMOVE CONSTANT COLUMNS
# ==========================================================

constant_columns = [col for col in df.columns if df[col].nunique() == 1]
df = df.drop(columns=constant_columns)


# ==========================================================
# 5. TRAIN / TEST SPLIT (ENGINE-WISE)
# ==========================================================

engine_ids = df["engine_id"].unique()
train_size = int(0.8 * len(engine_ids))

train_engines = engine_ids[:train_size]
test_engines = engine_ids[train_size:]

train_df = df[df["engine_id"].isin(train_engines)]
test_df = df[df["engine_id"].isin(test_engines)]


# ==========================================================
# 6. PREPARE FEATURES & TARGET
# ==========================================================

X_train = train_df.drop(columns=["engine_id", "RUL"])
y_train = train_df["RUL"]

X_test = test_df.drop(columns=["engine_id", "RUL"])
y_test = test_df["RUL"]


# ==========================================================
# 7. TRAIN RANDOM FOREST MODEL
# ==========================================================

print("Training RandomForest model...")

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

print("Model trained successfully.")


# ==========================================================
# 8. EVALUATE MODEL
# ==========================================================

y_pred = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"RandomForest RMSE: {rmse:.2f}")


# ==========================================================
# 9. XGBOOST COMPARISON
# ==========================================================

print("\nTraining XGBoost model...")

xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

print(f"XGBoost RMSE: {xgb_rmse:.2f}")


# ==========================================================
# 10. SAVE BEST MODEL
# ==========================================================

# Choose better model based on RMSE
if rmse <= xgb_rmse:
    joblib.dump(rf_model, "model/rul_model.pkl")
    joblib.dump(X_train.columns.tolist(), "model/feature_columns.pkl")
    print("RandomForest model saved.")
else:
    joblib.dump(xgb_model, "model/rul_model.pkl")
    print("XGBoost model saved.")