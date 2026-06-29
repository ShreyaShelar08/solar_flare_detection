import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# SECTION 1 — Load Data
# ─────────────────────────────────────────
df_sxr = pd.read_csv(r"C:\Users\Personal\Desktop\Shreya Shelar\Projects\ISRO\SoLEXS_combined_cleaned.csv", parse_dates=["datetime_utc"])
df_nowcast = pd.read_csv("nowcast_output.csv", parse_dates=["datetime_utc"])

df_sxr = df_sxr.sort_values("datetime_utc").reset_index(drop=True)
df_nowcast = df_nowcast.sort_values("datetime_utc").reset_index(drop=True)

# Merge SXR + nowcast on timestamp
df = pd.merge_asof(
    df_sxr,
    df_nowcast[["datetime_utc", "nowcast_label", "nowcast_proba"]],
    on="datetime_utc",
    tolerance=pd.Timedelta("2s"),
    direction="nearest"
)

df = df.dropna().reset_index(drop=True)
print("Merged shape:", df.shape)
print("Columns:", df.columns.tolist())

# ─────────────────────────────────────────
# SECTION 2 — Create is_true_flare FIRST
# ─────────────────────────────────────────
COUNT_THRESH = 22.0
ZSCORE_THRESH = 3.0

df["is_true_flare"] = (
    (df["zscore"] > ZSCORE_THRESH) &
    (df["counts_clean"] > COUNT_THRESH)
).astype(int)

print(f"True flare peaks (per second): {df['is_true_flare'].sum()}")

# ─────────────────────────────────────────
# SECTION 3 — Downsample FIRST, then label
# ─────────────────────────────────────────
df["minute"] = df["datetime_utc"].dt.floor("min")

df_min = df.groupby("minute").agg(
    counts_clean  =("counts_clean", "mean"),
    rolling_mean  =("rolling_mean", "mean"),
    rolling_std   =("rolling_std", "mean"),
    zscore        =("zscore", "max"),
    nowcast_label =("nowcast_label", "max"),
    nowcast_proba =("nowcast_proba", "max"),
    is_true_flare =("is_true_flare", "max")
).reset_index()

df_min = df_min.sort_values("minute").reset_index(drop=True)

print(f"\nPer-minute shape: {df_min.shape}")
print(f"Flare minutes: {df_min['is_true_flare'].sum()}")

# ─────────────────────────────────────────
# SECTION 4 — Label at per-minute level
# ─────────────────────────────────────────
FORECAST_WINDOW_MIN = 15

flare_flags = df_min["is_true_flare"].values
n = len(df_min)
target = np.zeros(n, dtype=int)

for i in range(n):
    end = min(i + FORECAST_WINDOW_MIN, n)
    if flare_flags[i:end].sum() > 0:
        target[i] = 1

df_min["target"] = target

print("\nPer-minute target distribution:")
print(pd.Series(target).value_counts())
print(f"Positive ratio: {target.mean():.4f}")

# ─────────────────────────────────────────
# DIAGNOSTIC — Feature discrimination check
# Run once, then remove
# ─────────────────────────────────────────
print("\n=== Feature Discrimination Check ===")
print("Mean counts_clean — non-flare:", df_min[df_min["target"]==0]["counts_clean"].mean())
print("Mean counts_clean — flare    :", df_min[df_min["target"]==1]["counts_clean"].mean())

print("\nMean nowcast_proba — non-flare:", df_min[df_min["target"]==0]["nowcast_proba"].mean())
print("Mean nowcast_proba — flare    :", df_min[df_min["target"]==1]["nowcast_proba"].mean())

print("\nMean nowcast_label — non-flare:", df_min[df_min["target"]==0]["nowcast_label"].mean())
print("Mean nowcast_label — flare    :", df_min[df_min["target"]==1]["nowcast_label"].mean())

import sys
sys.exit()  # stops here so you don't waste time training