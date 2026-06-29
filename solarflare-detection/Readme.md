# ☀️ SolarWatch — ISRO Aditya-L1 Solar Flare Detection

> **ISRO Build-A-Thon 2026 · Problem Statement PS15**  
> Real-time solar flare nowcasting and 15-minute forecasting using Aditya-L1 SXT (SDD detector) data.

---

## 🔭 Overview

SolarWatch is an end-to-end ML pipeline that processes soft X-ray (SXR) light curves from ISRO's Aditya-L1 satellite to:

- **Nowcast** solar flares in real time using a 1D CNN
- **Forecast** M/X-class flares 15 minutes ahead using a Bidirectional LSTM
- **Visualize** X-ray light curves with live alerts via a Streamlit dashboard

---

## 📊 Model Performance

| Model | Task | TSS | HSS | Recall |
|---|---|---|---|---|
| 1D CNN | Nowcasting (real-time) | **0.6693** | **0.2913** | **0.88** |
| BiLSTM | Forecasting (15-min lead) | **0.6044** | **0.4531** | **0.91** |

> TSS (True Skill Statistic) and HSS (Heidke Skill Score) are standard metrics for solar flare prediction evaluation.

---

## 🏗️ Pipeline Architecture

```
Raw SXR CSV (Aditya-L1 SDD)
        │
        ▼
┌─────────────────────┐
│  Data Cleaning      │  Remove NaNs, clip negatives, forward fill
│  Feature Engineering│  rolling_mean, rolling_std, zscore
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  CNN Nowcasting     │  60-second lookback window
│  (nowcast.py)       │  Focal loss · Threshold = 0.47
│                     │  TSS = 0.6693 · Recall = 0.88
└─────────┬───────────┘
          │  nowcast_output.csv
          ▼
┌─────────────────────┐
│  BiLSTM Forecasting │  120-minute lookback · per-minute data
│  (forecast.py)      │  Focal loss · Threshold = 0.49
│                     │  TSS = 0.6044 · HSS = 0.4531
└─────────┬───────────┘
          │  forecast_output.csv
          ▼
┌─────────────────────┐
│  Streamlit Dashboard│  Live inference · Visual alerts
│  (app.py)           │  Flare database · Download CSV
└─────────────────────┘
```

---

## 🧠 Model Details

### CNN Nowcasting (`nowcast.py`)

- **Input:** 60-second sliding window × 4 features
- **Features:** `counts_clean`, `rolling_mean`, `rolling_std`, `zscore`
- **Architecture:** 3× Conv1D blocks with BatchNorm + Dropout → Dense → Sigmoid
- **Loss:** Focal loss (γ=2.0, α=0.85)
- **Label:** zscore > 3.0 AND counts > 22.0 (95th percentile), 180s precursor window
- **Class weight:** ~27:1 (negative:positive)
- **Optimal threshold:** 0.47 (TSS-maximized)

### BiLSTM Forecasting (`forecast.py`)

- **Input:** 120-minute sliding window × 5 features (per-minute aggregated)
- **Features:** `counts_clean`, `rolling_mean`, `rolling_std`, `nowcast_label`, `nowcast_proba`
- **Architecture:** 3× Bidirectional LSTM layers with BatchNorm + Dropout → Dense → Sigmoid
- **Loss:** Focal loss (γ=2.0, α=0.85)
- **Label:** Any true flare peak within next 15 minutes → target=1
- **Optimal threshold:** 0.49 (TSS-maximized)

---

## 🖥️ Dashboard Features

| Feature | Description |
|---|---|
| 📤 Upload CSV | Upload any Aditya-L1 SXR CSV for live inference |
| 🔴 Alert Banner | Real-time alert when flare detected or forecast |
| 📈 Light Curve | Interactive SXR plot with flare markers + nowcast probability overlay |
| 🎯 Forecast Gauge | Visual probability gauge for 15-minute forecast |
| 📋 Flare Database | Automated table of all detected flares with timestamps |
| ⬇️ Download | Export flare database as CSV |

---

## 🔬 Label Engineering

**True flare peaks** are identified as:
```python
is_true_flare = (zscore > 3.0) AND (counts_clean > 22.0)
```
Where 22.0 is the 95th percentile of counts_clean across the dataset.

**Nowcast labels:** 180-second precursor window before each true flare peak → target=1

**Forecast labels:** Any true flare peak within next 15 minutes → target=1

---

## 📡 Data Source

- **Satellite:** ISRO Aditya-L1
- **Instrument:** Solar X-ray Monitor (SXM)
- **Detector:** SDD (Silicon Drift Detector) — Soft X-Ray channel
- **Cadence:** 1 second
- **Coverage:** ~7 days of continuous observation

---

