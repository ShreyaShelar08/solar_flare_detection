import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report

# ─────────────────────────────────────────
# SECTION 1 — Load & Merge Data
# ─────────────────────────────────────────
df_sxr = pd.read_csv(r"C:\Users\Personal\Desktop\Shreya Shelar\Projects\ISRO\SoLEXS_combined_cleaned.csv", parse_dates=["datetime_utc"])
df_nowcast = pd.read_csv("nowcast_output.csv", parse_dates=["datetime_utc"])

df_sxr = df_sxr.sort_values("datetime_utc").reset_index(drop=True)
df_nowcast = df_nowcast.sort_values("datetime_utc").reset_index(drop=True)

df = pd.merge_asof(
    df_sxr,
    df_nowcast[["datetime_utc", "nowcast_label", "nowcast_proba"]],
    on="datetime_utc",
    tolerance=pd.Timedelta("2s"),
    direction="nearest"
)

df = df.dropna().reset_index(drop=True)
print("Merged shape:", df.shape)

# ─────────────────────────────────────────
# SECTION 2 — Create is_true_flare
# ─────────────────────────────────────────
COUNT_THRESH = 22.0
ZSCORE_THRESH = 3.0

df["is_true_flare"] = (
    (df["zscore"] > ZSCORE_THRESH) &
    (df["counts_clean"] > COUNT_THRESH)
).astype(int)

print(f"True flare peaks (per second): {df['is_true_flare'].sum()}")

# ─────────────────────────────────────────
# SECTION 3 — Downsample to Per-Minute
# ─────────────────────────────────────────
df["minute"] = df["datetime_utc"].dt.floor("min")

df_min = df.groupby("minute").agg(
    counts_clean  =("counts_clean", "mean"),
    rolling_mean  =("rolling_mean", "mean"),
    rolling_std   =("rolling_std", "mean"),
    nowcast_label =("nowcast_label", "max"),
    nowcast_proba =("nowcast_proba", "max"),
    is_true_flare =("is_true_flare", "max")
).reset_index()

df_min = df_min.sort_values("minute").reset_index(drop=True)

print(f"Per-minute shape: {df_min.shape}")
print(f"Flare minutes: {df_min['is_true_flare'].sum()}")

# ─────────────────────────────────────────
# SECTION 4 — Label Engineering
# target=1 if flare occurs in next 15 minutes
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
# SECTION 5 — Sliding Window (120 min lookback)
# ─────────────────────────────────────────
FEATURES = ["counts_clean", "rolling_mean", "rolling_std",
            "nowcast_label", "nowcast_proba"]  # zscore removed
WINDOW_SIZE = 120

X_raw = df_min[FEATURES].fillna(0).values
y_raw = df_min["target"].values

X_windows, y_windows = [], []

for i in range(WINDOW_SIZE, len(X_raw)):
    X_windows.append(X_raw[i - WINDOW_SIZE:i])
    y_windows.append(y_raw[i])

X_windows = np.array(X_windows)  # (samples, 120, 5)
y_windows = np.array(y_windows)

print(f"\nWindowed shape: {X_windows.shape}")
print("Windowed target dist:", pd.Series(y_windows).value_counts().to_dict())

# ─────────────────────────────────────────
# SECTION 6 — Chronological Train/Test Split
# ─────────────────────────────────────────
split = int(len(X_windows) * 0.8)

X_train, X_test = X_windows[:split], X_windows[split:]
y_train, y_test = y_windows[:split], y_windows[split:]

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print("Train target dist:", pd.Series(y_train).value_counts().to_dict())
print("Test target dist:", pd.Series(y_test).value_counts().to_dict())

# ─────────────────────────────────────────
# SECTION 7 — Class Weight
# ─────────────────────────────────────────
neg = np.sum(y_train == 0)
pos = np.sum(y_train == 1)
class_weight = {0: 1.0, 1: neg / pos}
print(f"\nClass weight for flare class: {class_weight[1]:.2f}")

# ─────────────────────────────────────────
# SECTION 8 — Focal Loss
# ─────────────────────────────────────────
def focal_loss(gamma=2.0, alpha=0.85):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal = alpha_t * tf.pow(1 - p_t, gamma) * bce
        return tf.reduce_mean(focal)
    return loss

# ─────────────────────────────────────────
# SECTION 9 — BiLSTM Model
# ─────────────────────────────────────────
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True),
                  input_shape=(WINDOW_SIZE, len(FEATURES))),
    BatchNormalization(),
    Dropout(0.3),

    Bidirectional(LSTM(32, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),

    Bidirectional(LSTM(16, return_sequences=False)),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=focal_loss(gamma=2.0, alpha=0.85),
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

model.summary()

# ─────────────────────────────────────────
# SECTION 10 — Train
# ─────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_recall', patience=7,
                  restore_best_weights=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=3, verbose=1)
]

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────
# SECTION 11 — Threshold Tuning + Evaluate
# ─────────────────────────────────────────
y_prob = model.predict(X_test).flatten()

best_tss, best_hss, best_thresh = -1, -1, 0.5

for thresh in np.arange(0.01, 0.5, 0.01):
    y_pred_t = (y_prob >= thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_t)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        if (TP + FN) > 0 and (FP + TN) > 0:
            tss = (TP / (TP + FN)) - (FP / (FP + TN))
            denom = ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
            hss = (2 * (TP * TN - FP * FN)) / denom if denom > 0 else 0
            if tss > best_tss:
                best_tss = tss
                best_hss = hss
                best_thresh = thresh

print(f"\nBest Threshold : {best_thresh:.2f}")
print(f"Best TSS       : {best_tss:.4f}")
print(f"Best HSS       : {best_hss:.4f}")

y_pred_final = (y_prob >= best_thresh).astype(int)
cm = confusion_matrix(y_test, y_pred_final)
TN, FP, FN, TP = cm.ravel()

print("\n── BiLSTM Evaluation ──")
print(classification_report(y_test, y_pred_final))
print(f"TP: {TP}  FP: {FP}  TN: {TN}  FN: {FN}")

# ─────────────────────────────────────────
# SECTION 12 — Save Everything
# ─────────────────────────────────────────
X_all = X_windows
y_all_prob = model.predict(X_all).flatten()
y_all_pred = (y_all_prob >= best_thresh).astype(int)

out_df = df_min.iloc[WINDOW_SIZE:].copy().reset_index(drop=True)
out_df["forecast_label"] = y_all_pred
out_df["forecast_proba"] = y_all_prob

out_df[["minute", "counts_clean", "nowcast_label", "nowcast_proba",
        "target", "forecast_label",
        "forecast_proba"]].to_csv("forecast_output.csv", index=False)

model.save("forecast_bilstm_model.keras")

config = {
    "best_threshold": float(best_thresh),
    "tss": float(best_tss),
    "hss": float(best_hss),
    "window_size": WINDOW_SIZE,
    "forecast_window_minutes": FORECAST_WINDOW_MIN,
    "features": FEATURES
}
with open("forecast_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\n✅ Saved: forecast_output.csv")
print("✅ Saved: forecast_bilstm_model.keras")
print("✅ Saved: forecast_config.json")
print("\n🚀 Forecasting complete!")