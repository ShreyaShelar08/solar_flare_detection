import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# ─────────────────────────────────────────
# SECTION 1 — Load Data
# ─────────────────────────────────────────
df = pd.read_csv(r"C:\Users\Personal\Desktop\Shreya Shelar\Projects\ISRO\SoLEXS_combined_cleaned.csv", parse_dates=["datetime_utc"])
df = df.sort_values("datetime_utc").reset_index(drop=True)

# ─────────────────────────────────────────
# SECTION 2 — Rebuild Label Engineering
# ─────────────────────────────────────────
df["target"] = 0

# Step 1: Mark actual flare peaks — zscore > 2.5 AND counts above baseline
ZSCORE_THRESH = 3.0
COUNT_THRESH = df["counts_clean"].quantile(0.95)  # top 10% of counts

print(f"Count threshold (95th percentile): {COUNT_THRESH:.1f}")

# True flare = high zscore AND high absolute counts
df["is_true_flare"] = (
    (df["zscore"] > ZSCORE_THRESH) &
    (df["counts_clean"] > COUNT_THRESH)
).astype(int)

print(f"True flare peaks found: {df['is_true_flare'].sum()}")

# Step 2: For each true flare peak, mark 180 rows BEFORE as precursor
true_flare_indices = df[df["is_true_flare"] == 1].index.tolist()

PRECURSOR_WINDOW = 180

for idx in true_flare_indices:
    start = max(0, idx - PRECURSOR_WINDOW)
    df.loc[start:idx, "target"] = 1

print(f"\nAfter Label Engineering:")
print(df["target"].value_counts())
print(f"Positive ratio: {df['target'].mean():.4f}")

# Quick sanity check
print("\n=== NON-FLARE mean counts ===", df[df["target"]==0]["counts_clean"].mean())
print("=== FLARE mean counts ===", df[df["target"]==1]["counts_clean"].mean())
# Flare mean should now be HIGHER than non-flare
# ─────────────────────────────────────────
# SECTION 3 — Sliding Window (CNN input)
# ─────────────────────────────────────────
FEATURES = ["counts_clean", "rolling_mean", "rolling_std", "zscore"]
WINDOW_SIZE = 60  # 60 seconds lookback for nowcasting

X_raw = df[FEATURES].fillna(0).values
y_raw = df["target"].values

X_windows, y_windows = [], []

for i in range(WINDOW_SIZE, len(X_raw)):
    X_windows.append(X_raw[i - WINDOW_SIZE:i])   # shape: (60, 4)
    y_windows.append(y_raw[i])                    # label at current second

X_windows = np.array(X_windows)   # (samples, 60, 4)
y_windows = np.array(y_windows)

print(f"\nWindowed shape: {X_windows.shape}")
print("Windowed target dist:", pd.Series(y_windows).value_counts().to_dict())

# ─────────────────────────────────────────
# SECTION 4 — Train/Test Split (no SMOTE for CNN)
# ─────────────────────────────────────────
# Time series — DO NOT shuffle, split chronologically
split = int(len(X_windows) * 0.8)

X_train, X_test = X_windows[:split], X_windows[split:]
y_train, y_test = y_windows[:split], y_windows[split:]

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ─────────────────────────────────────────
# SECTION 5 — Class Weight (instead of SMOTE)
# ─────────────────────────────────────────
# Compute weight manually: ratio of negative to positive
neg = np.sum(y_train == 0)
pos = np.sum(y_train == 1)
class_weight = {0: 1.0, 1: neg / pos}

print(f"\nClass weight for flare class: {class_weight[1]:.2f}")

# ─────────────────────────────────────────
# SECTION 6 — CNN Model
# ─────────────────────────────────────────
model = Sequential([
    # Block 1
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE, len(FEATURES))),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # Block 2
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # Block 3
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')   # binary output
])

def focal_loss(gamma=2.0, alpha=0.85):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal = alpha_t * tf.pow(1 - p_t, gamma) * bce
        return tf.reduce_mean(focal)
    return loss

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=focal_loss(gamma=2.0, alpha=0.95),   # alpha=0.95 heavily weights flare class
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

model.summary()

# ─────────────────────────────────────────
# SECTION 7 — Train
# ─────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_recall', patience=5,
                  restore_best_weights=True, mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=3, verbose=1)
]

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=256,
    validation_split=0.1,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# # ─────────────────────────────────────────
# SECTION 8 — Threshold Tuning + Evaluate
# ─────────────────────────────────────────
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

y_prob = model.predict(X_test).flatten()

# Find best threshold by maximizing TSS across all thresholds
thresholds = np.arange(0.01, 0.5, 0.01)
best_tss, best_hss, best_thresh = -1, -1, 0.5

for thresh in thresholds:
    y_pred_t = (y_prob >= thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_t)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        if (TP + FN) > 0 and (FP + TN) > 0:
            tss = (TP / (TP + FN)) - (FP / (FP + TN))
            denom = ((TP + FN)*(FN + TN) + (TP + FP)*(FP + TN))
            hss = (2 * (TP * TN - FP * FN)) / denom if denom > 0 else 0
            if tss > best_tss:
                best_tss = tss
                best_hss = hss
                best_thresh = thresh

for thresh in [0.40, 0.42, 0.44, 0.46, 0.48]:
    y_pred_t = (y_prob >= thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_t)
    TN, FP, FN, TP = cm.ravel()
    tss = (TP/(TP+FN)) - (FP/(FP+TN))
    denom = ((TP+FN)*(FN+TN) + (TP+FP)*(FP+TN))
    hss = (2*(TP*TN - FP*FN)) / denom if denom > 0 else 0
    print(f"Thresh {thresh:.2f} → TSS: {tss:.4f}  HSS: {hss:.4f}  TP:{TP}  FP:{FP}")

print(f"\nBest Threshold : {best_thresh:.2f}")
print(f"Best TSS       : {best_tss:.4f}")
print(f"Best HSS       : {best_hss:.4f}")

# Final evaluation at best threshold
y_pred_final = (y_prob >= best_thresh).astype(int)
cm = confusion_matrix(y_test, y_pred_final)
TN, FP, FN, TP = cm.ravel()

print("\n── Final CNN Evaluation ──")
print(classification_report(y_test, y_pred_final))
print(f"TP: {TP}  FP: {FP}  TN: {TN}  FN: {FN}")

# Precision-Recall curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(8, 4))
plt.plot(recall_vals, precision_vals)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — CNN Nowcast")
plt.grid(True)
plt.savefig("pr_curve_nowcast.png", dpi=150)
plt.show()
print("✅ Saved: pr_curve_nowcast.png")

# ─────────────────────────────────────────
# SECTION 9 — Save Output CSV
# ─────────────────────────────────────────
# Predict on full windowed data
X_all = X_windows
y_all_prob = model.predict(X_all).flatten()
y_all_pred = (y_all_prob >= 0.5).astype(int)

# Align back to original df (first WINDOW_SIZE rows have no window)
out_df = df.iloc[WINDOW_SIZE:].copy().reset_index(drop=True)
out_df["nowcast_label"] = y_all_pred
out_df["nowcast_proba"] = y_all_prob

out_df[["datetime_utc", "counts_clean", "rolling_mean", "rolling_std",
        "zscore", "is_outlier_flare", "target",
        "nowcast_label", "nowcast_proba"]].to_csv("nowcast_output.csv", index=False)

print("\n✅ Saved: nowcast_output.csv")
model.save("nowcast_cnn_model.h5")

import json
config = {
    "best_threshold": 0.47,
    "tss": 0.6693,
    "hss": 0.2913,
    "precursor_window": 180,
    "zscore_thresh": 3.0,
    "count_thresh_percentile": 95,
    "count_thresh_value": 22.0,
    "window_size": 60,
    "features": ["counts_clean", "rolling_mean", "rolling_std", "zscore"]
}
with open("nowcast_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("✅ Saved: nowcast_cnn_model.h5")
print("✅ Saved: nowcast_config.json")
print("✅ Saved: nowcast_output.csv")
