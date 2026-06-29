import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="SolarWatch — ISRO Aditya-L1",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# DARK THEME + ISRO ORANGE CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

.stApp {
    background: radial-gradient(ellipse at top, #12121c 0%, #0a0a0f 55%, #07070c 100%);
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #0a0a12 100%);
    border-right: 1px solid #1e1e2e;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 0.8rem;
}

/* Header */
.solar-header {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a0a00 100%);
    border: 1px solid #ff6b0022;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.solar-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, #ff6b0015 0%, transparent 70%);
    border-radius: 50%;
}

.solar-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.5px;
}

.solar-title span {
    color: #ff6b00;
}

.solar-subtitle {
    font-size: 0.9rem;
    color: #888899;
    margin-top: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.5px;
}

.hero-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    background: rgba(255, 107, 0, 0.12);
    border: 1px solid rgba(255, 107, 0, 0.24);
    color: #ffb06a;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.8rem;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.metric-card {
    background: linear-gradient(145deg, rgba(15, 15, 26, 0.96), rgba(10, 10, 15, 0.95));
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s ease, border-color 0.2s ease;
    animation: fadeInUp 0.7s ease both;
}

.metric-card:nth-child(2) { animation-delay: 0.08s; }
.metric-card:nth-child(3) { animation-delay: 0.16s; }
.metric-card:nth-child(4) { animation-delay: 0.24s; }

.metric-card:hover {
    transform: translateY(-2px);
    border-color: #2b2b3d;
}

.metric-card.orange { border-top: 2px solid #ff6b00; }
.metric-card.green  { border-top: 2px solid #00ff88; }
.metric-card.blue   { border-top: 2px solid #4488ff; }
.metric-card.red    { border-top: 2px solid #ff4444; }

.metric-label {
    font-size: 0.7rem;
    color: #666677;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'JetBrains Mono', monospace;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff;
    margin-top: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}

.metric-value.orange { color: #ff6b00; }
.metric-value.green  { color: #00ff88; }
.metric-value.blue   { color: #4488ff; }
.metric-value.red    { color: #ff4444; }

/* Alert banners */
.alert-flare {
    background: linear-gradient(135deg, #3a0000, #1a0000);
    border: 1px solid #ff4444;
    border-left: 4px solid #ff4444;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    animation: pulse-red 2s infinite;
    box-shadow: 0 10px 28px rgba(255, 68, 68, 0.15);
}

.alert-warning {
    background: linear-gradient(135deg, #2a1500, #1a0d00);
    border: 1px solid #ff6b00;
    border-left: 4px solid #ff6b00;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 10px 28px rgba(255, 107, 0, 0.12);
}

.alert-safe {
    background: linear-gradient(135deg, #001a0a, #000f06);
    border: 1px solid #00ff88;
    border-left: 4px solid #00ff88;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 10px 28px rgba(0, 255, 136, 0.12);
}

.alert-title {
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.alert-body {
    font-size: 0.85rem;
    color: #aaaacc;
    font-family: 'JetBrains Mono', monospace;
}

@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 #ff444433; }
    50% { box-shadow: 0 0 20px 4px #ff444422; }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Section headers */
.section-label {
    font-size: 0.7rem;
    color: #ff6b00;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.5rem;
}

.section-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 1rem;
}

/* Flare database table */
.flare-table {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 1rem;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: #0f0f1a;
    border: 1px dashed #ff6b0044;
    border-radius: 10px;
    padding: 1rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #ff6b00, #ff8c33);
    color: #000000;
    font-weight: 700;
    border: none;
    border-radius: 999px;
    padding: 0.6rem 1.4rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    transition: all 0.2s;
    box-shadow: 0 8px 22px rgba(255, 107, 0, 0.22);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #ff8c33, #ff6b00);
    transform: translateY(-1px);
    box-shadow: 0 12px 24px rgba(255, 107, 0, 0.28);
}

/* Score badges */
.score-badge {
    display: inline-block;
    background: #1a1a2e;
    border: 1px solid #ff6b0044;
    border-radius: 6px;
    padding: 0.3rem 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #ff6b00;
    margin: 0.2rem;
}

/* Probability gauge text */
.prob-display {
    text-align: center;
    padding: 1.5rem;
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
}

.prob-number {
    font-size: 3rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

.prob-label {
    font-size: 0.75rem;
    color: #666677;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Summary bar */
.summary-bar {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.8rem;
    margin-bottom: 1rem;
}

.summary-pill {
    background: linear-gradient(145deg, rgba(15, 15, 26, 0.96), rgba(10, 10, 15, 0.95));
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 0.8rem 0.95rem;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.16);
}

.summary-pill span {
    display: block;
    font-size: 0.68rem;
    color: #666677;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.25rem;
}

.summary-pill strong {
    color: #f2f2f7;
    font-size: 0.95rem;
}

/* Sidebar polish */
.sidebar-card {
    background: linear-gradient(135deg, rgba(255, 107, 0, 0.12), rgba(255, 107, 0, 0.05));
    border: 1px solid rgba(255, 107, 0, 0.2);
    border-radius: 14px;
    padding: 0.85rem 0.95rem;
    margin-bottom: 0.9rem;
}

.sidebar-kicker {
    font-size: 0.64rem;
    color: #ffb06a;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.3rem;
}

.sidebar-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #ffffff;
}

.sidebar-caption {
    font-size: 0.78rem;
    color: #7a7a90;
    margin-top: 0.2rem;
    font-family: 'JetBrains Mono', monospace;
}

.sidebar-section-title {
    font-size: 0.72rem;
    color: #ff6b00;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-family: 'JetBrains Mono', monospace;
    margin: 0.8rem 0 0.4rem;
}

.sidebar-meta {
    font-size: 0.78rem;
    color: #8a8aa4;
    line-height: 1.6;
    margin: 0.2rem 0 0.5rem;
}

/* Empty state + panels */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    background: linear-gradient(145deg, rgba(15, 15, 26, 0.96), rgba(10, 10, 15, 0.95));
    border: 1px solid rgba(255, 107, 0, 0.18);
    border-radius: 16px;
    margin-top: 1rem;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.24);
}

.section-card {
    background: linear-gradient(145deg, rgba(15, 15, 26, 0.96), rgba(10, 10, 15, 0.95));
    border: 1px solid #1e1e2e;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.18);
}

div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HELPERS — Feature Engineering
# ─────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    df["counts_clean"] = pd.to_numeric(df["counts"], errors="coerce")
    df["counts_clean"] = df["counts_clean"].clip(lower=0).ffill().fillna(0)
    WINDOW = 60
    df["rolling_mean"] = df["counts_clean"].rolling(WINDOW, min_periods=1).mean()
    df["rolling_std"]  = df["counts_clean"].rolling(WINDOW, min_periods=1).std().fillna(1)
    df["zscore"] = (df["counts_clean"] - df["rolling_mean"]) / (df["rolling_std"] + 1e-8)
    df["is_true_flare"] = (
        (df["zscore"] > 3.0) & (df["counts_clean"] > 22.0)
    ).astype(int)
    return df

def make_nowcast_windows(df, window_size=60, batch_size=2048):
    features = ["counts_clean", "rolling_mean", "rolling_std", "zscore"]
    X = df[features].fillna(0).astype(np.float32).values
    n = len(X)
    n_windows = max(0, n - window_size)
    if n_windows == 0:
        return None, 0, features

    def generator():
        for i in range(window_size, n):
            yield X[i - window_size:i]

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(
            shape=(window_size, len(features)), dtype=tf.float32
        )
    ).batch(batch_size)
    return ds, n_windows, features

def downsample_to_minute(df, nowcast_labels, nowcast_probas, window_size=60):
    df_out = df.iloc[window_size:].copy().reset_index(drop=True)
    df_out["nowcast_label"] = nowcast_labels
    df_out["nowcast_proba"] = nowcast_probas
    df_out["minute"] = df_out["datetime_utc"].dt.floor("min")
    df_min = df_out.groupby("minute").agg(
        counts_clean  =("counts_clean", "mean"),
        rolling_mean  =("rolling_mean", "mean"),
        rolling_std   =("rolling_std", "mean"),
        nowcast_label =("nowcast_label", "max"),
        nowcast_proba =("nowcast_proba", "max"),
        is_true_flare =("is_true_flare", "max")
    ).reset_index()
    return df_min.sort_values("minute").reset_index(drop=True)

def make_forecast_windows(df_min, window_size=120, batch_size=512):
    features = ["counts_clean", "rolling_mean", "rolling_std",
                "nowcast_label", "nowcast_proba"]
    X = df_min[features].fillna(0).astype(np.float32).values
    n = len(X)
    n_windows = max(0, n - window_size)
    if n_windows == 0:
        return None, [], features

    ds = tf.data.Dataset.from_generator(
        lambda: (X[i - window_size:i] for i in range(window_size, n)),
        output_signature=tf.TensorSpec(
            shape=(window_size, len(features)), dtype=tf.float32
        )
    ).batch(batch_size)
    indices = list(range(window_size, n))
    return ds, indices, features

# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        def focal_loss(gamma=2.0, alpha=0.85):
            def loss(y_true, y_pred):
                y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
                bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
                p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
                alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
                focal = alpha_t * tf.pow(1 - p_t, gamma) * bce
                return tf.reduce_mean(focal)
            return loss

        nowcast_model  = load_model("nowcast_cnn_model.h5",
                                    custom_objects={"loss": focal_loss()})
        forecast_model = load_model("forecast_bilstm_model.keras",
                                    custom_objects={"loss": focal_loss()})

        with open("nowcast_config.json") as f:
            nowcast_cfg = json.load(f)
        with open("forecast_config.json") as f:
            forecast_cfg = json.load(f)

        return nowcast_model, forecast_model, nowcast_cfg, forecast_cfg, None
    except Exception as e:
        return None, None, None, None, str(e)

# ─────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────
def plot_lightcurve(df, nowcast_labels=None, nowcast_probas=None, title="SXR Light Curve"):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35],
                        vertical_spacing=0.05)

    # Main light curve
    fig.add_trace(go.Scatter(
        x=df["datetime_utc"], y=df["counts_clean"],
        mode="lines", name="SXR Counts",
        line=dict(color="#4488ff", width=1),
        fill="tozeroy", fillcolor="rgba(68,136,255,0.08)"
    ), row=1, col=1)

    # Flare markers
    flare_df = df[df["is_true_flare"] == 1]
    if len(flare_df) > 0:
        fig.add_trace(go.Scatter(
            x=flare_df["datetime_utc"], y=flare_df["counts_clean"],
            mode="markers", name="Flare Peak",
            marker=dict(color="#ff6b00", size=8, symbol="star",
                       line=dict(color="#ffffff", width=1))
        ), row=1, col=1)

    # Nowcast probability
    if nowcast_probas is not None:
        times = df["datetime_utc"].iloc[60:].reset_index(drop=True)
        fig.add_trace(go.Scatter(
            x=times, y=nowcast_probas,
            mode="lines", name="Nowcast P(flare)",
            line=dict(color="#ff6b00", width=1.5),
            fill="tozeroy", fillcolor="rgba(255,107,0,0.12)"
        ), row=2, col=1)

        # Threshold line
        fig.add_hline(y=0.47, line_dash="dash",
                     line_color="rgba(255,68,68,0.53)", row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0f0f1a",
        font=dict(family="Space Grotesk", color="#e8e8f0"),
        title=dict(text=title, font=dict(size=14, color="#ffffff")),
        height=440,
        margin=dict(l=20, r=20, t=50, b=10),
        legend=dict(orientation="h", yanchor="top", y=1.08, xanchor="left", x=0.0,
                   bgcolor="rgba(0,0,0,0)"),
        showlegend=True,
        hovermode="x unified"
    )
    fig.update_xaxes(gridcolor="#1e1e2e", showgrid=True)
    fig.update_yaxes(gridcolor="#1e1e2e", showgrid=True)

    return fig

def plot_forecast_timeline(df_min, forecast_probas, forecast_indices, threshold):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_min["minute"], y=df_min["counts_clean"],
        mode="lines", name="SXR (per min)",
        line=dict(color="#4488ff", width=1.5),
        yaxis="y1"
    ))

    prob_times = df_min["minute"].iloc[forecast_indices]
    fig.add_trace(go.Scatter(
        x=prob_times, y=forecast_probas,
        mode="lines", name="Forecast P(flare in 15 min)",
        line=dict(color="#ff6b00", width=2),
        fill="tozeroy", fillcolor="rgba(255,107,0,0.1)",
        yaxis="y2"
    ))

    # Alert zones
    alert_mask = forecast_probas >= threshold
    if alert_mask.any():
        alert_times = prob_times.values[alert_mask]
        alert_probs = forecast_probas[alert_mask]
        fig.add_trace(go.Scatter(
            x=alert_times, y=alert_probs,
            mode="markers", name="⚠ Alert",
            marker=dict(color="#ff4444", size=6, symbol="triangle-up"),
            yaxis="y2"
        ))

    fig.add_hline(y=threshold, line_dash="dash",
                 line_color="rgba(255,68,68,0.53)",
                 annotation_text=f"Alert threshold ({threshold:.2f})",
                 annotation_font_color="#ff4444",
                 yref="y2")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0f0f1a",
        font=dict(family="Space Grotesk", color="#e8e8f0"),
        height=400,
        margin=dict(l=20, r=20, t=50, b=10),
        yaxis=dict(title="Counts", gridcolor="#1e1e2e"),
        yaxis2=dict(title="Forecast Probability", overlaying="y",
                   side="right", range=[0, 1], gridcolor="#1e1e2e"),
        legend=dict(orientation="h", yanchor="top", y=1.08, xanchor="left", x=0.0,
                   bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified"
    )

    return fig

def plot_gauge(probability, threshold):
    color = "#ff4444" if probability >= threshold else \
            "#ff6b00" if probability >= threshold * 0.7 else "#00ff88"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number=dict(suffix="%", font=dict(size=36, color=color,
                                          family="JetBrains Mono")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#444455",
                     tickfont=dict(color="#666677")),
            bar=dict(color=color, thickness=0.25),
            bgcolor="#0f0f1a",
            bordercolor="#1e1e2e",
            steps=[
                dict(range=[0, threshold*70], color="#0f1a0f"),
                dict(range=[threshold*70, threshold*100], color="#1a1000"),
                dict(range=[threshold*100, 100], color="#1a0000"),
            ],
            threshold=dict(line=dict(color="#ff4444", width=3),
                          thickness=0.8, value=threshold * 100)
        ),
        title=dict(text="15-min Forecast Probability",
                  font=dict(color="#888899", size=12,
                            family="Space Grotesk"))
    ))

    fig.update_layout(
        paper_bgcolor="#0a0a0f",
        font=dict(color="#e8e8f0"),
        height=260,
        margin=dict(l=20, r=20, t=40, b=10)
    )
    return fig

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sidebar-card'>
        <div class='sidebar-kicker'>Mission Control</div>
        <div class='sidebar-title'>SolarWatch</div>
        <div class='sidebar-caption'>ISRO · Aditya-L1 · SDD Detector</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section-title'>Upload Data</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "CSV with datetime_utc + counts columns",
        type=["csv"],
        help="Upload Aditya-L1 SDD detector CSV"
    )

    st.markdown("<div class='sidebar-section-title'>Model Performance</div>", unsafe_allow_html=True)

    nowcast_model, forecast_model, nowcast_cfg, forecast_cfg, model_error = load_models()

    if nowcast_cfg:
        st.markdown(f"""
        <div style='font-size:0.7rem; color:#888899; margin-bottom:0.3rem;'>NOWCAST CNN</div>
        <span class='score-badge'>TSS {nowcast_cfg['tss']:.3f}</span>
        <span class='score-badge'>HSS {nowcast_cfg['hss']:.3f}</span>
        <div style='font-size:0.7rem; color:#888899; margin: 0.8rem 0 0.3rem;'>FORECAST BiLSTM</div>
        <span class='score-badge'>TSS {forecast_cfg['tss']:.3f}</span>
        <span class='score-badge'>HSS {forecast_cfg['hss']:.3f}</span>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='sidebar-meta'>
        CNN Nowcast · 60s window<br>
        BiLSTM Forecast · 120min window<br>
        Lead time · 15 minutes<br>
        Detector · SDD (SXR)
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class='solar-header'>
    <div class='hero-chip'>ISRO · ADITYA-L1</div>
    <div class='solar-title'>Solar<span>Watch</span></div>
    <div class='solar-subtitle'>
        Operational flare nowcasting and forecasting dashboard for SXR monitoring
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='summary-bar'>
    <div class='summary-pill'><span>System</span><strong>Live monitoring</strong></div>
    <div class='summary-pill'><span>Models</span><strong>CNN + BiLSTM</strong></div>
    <div class='summary-pill'><span>Lead time</span><strong>15 minutes</strong></div>
    <div class='summary-pill'><span>Windowing</span><strong>60s / 120min</strong></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# MODEL ERROR CHECK
# ─────────────────────────────────────────
if model_error:
    st.error(f"⚠ Models not found: {model_error}")
    st.info("Place `nowcast_cnn_model.h5`, `forecast_bilstm_model.keras`, "
            "`nowcast_config.json`, and `forecast_config.json` in the same folder as app.py")
    st.stop()

# ─────────────────────────────────────────
# NO FILE UPLOADED — SHOW DEMO STATE
# ─────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div class='empty-state'>
        <div style='font-size:3rem;'>☀️</div>
        <div style='font-size:1.2rem; font-weight:600;
                    color:#ffffff; margin-top:1rem;'>
            Upload SXR data to begin analysis
        </div>
        <div style='font-size:0.85rem; color:#7a7a90; margin-top:0.5rem;
                    font-family:JetBrains Mono;'>
            CSV must contain: datetime_utc and counts columns
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show static model scorecard
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Pipeline Overview</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='metric-card blue'>
            <div class='metric-label'>Step 1</div>
            <div class='metric-value' style='font-size:1.1rem; color:#4488ff;'>
                CNN Nowcasting
            </div>
            <div style='font-size:0.8rem; color:#666677; margin-top:0.5rem;'>
                60s lookback · Real-time flare detection
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card orange'>
            <div class='metric-label'>Step 2</div>
            <div class='metric-value' style='font-size:1.1rem; color:#ff6b00;'>
                BiLSTM Forecast
            </div>
            <div style='font-size:0.8rem; color:#666677; margin-top:0.5rem;'>
                120min lookback · 15-min lead time
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card green'>
            <div class='metric-label'>Step 3</div>
            <div class='metric-value' style='font-size:1.1rem; color:#00ff88;'>
                Visual Alerts
            </div>
            <div style='font-size:0.8rem; color:#666677; margin-top:0.5rem;'>
                Real-time alerts + flare database
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────
# FILE UPLOADED — RUN PIPELINE
# ─────────────────────────────────────────
with st.spinner("Loading data..."):
    try:
        df_raw = pd.read_csv(uploaded_file, parse_dates=["datetime_utc"])
        df_raw["datetime_utc"] = pd.to_datetime(
            df_raw["datetime_utc"], utc=True, errors="coerce"
        )
        df_raw = df_raw.dropna(subset=["datetime_utc"]).reset_index(drop=True)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

with st.spinner("Engineering features..."):
    df = engineer_features(df_raw)

# ── NOWCASTING ──
with st.spinner("Running CNN nowcasting..."):
    X_now, n_now, _ = make_nowcast_windows(df, window_size=60)
    if n_now == 0:
        st.error("Not enough data — need at least 60 rows.")
        st.stop()
    nowcast_probas = nowcast_model.predict(X_now, verbose=0).flatten()
    nowcast_labels = (nowcast_probas >= nowcast_cfg["best_threshold"]).astype(int)

# ── DOWNSAMPLING ──
df_min = downsample_to_minute(df, nowcast_labels, nowcast_probas, window_size=60)

# ── FORECASTING ──
with st.spinner("Running BiLSTM forecasting..."):
    X_fore, fore_indices, _ = make_forecast_windows(df_min, window_size=120)
    if len(fore_indices) == 0:
        st.warning("Not enough data for forecasting — need at least 120 minutes.")
        forecast_probas = np.array([])
    else:
        forecast_probas = forecast_model.predict(X_fore, verbose=0).flatten()

fore_thresh = forecast_cfg["best_threshold"]

# ─────────────────────────────────────────
# CURRENT STATUS
# ─────────────────────────────────────────
latest_nowcast_prob  = float(nowcast_probas[-1]) if len(nowcast_probas) > 0 else 0
latest_nowcast_label = int(nowcast_labels[-1])   if len(nowcast_labels) > 0 else 0
latest_forecast_prob = float(forecast_probas[-1]) if len(forecast_probas) > 0 else 0
latest_forecast_alert = latest_forecast_prob >= fore_thresh

total_flares    = int(df["is_true_flare"].sum())
total_minutes   = len(df_min)
flare_minutes   = int((nowcast_labels == 1).sum())

# ── ALERT BANNER ──
if latest_nowcast_label == 1:
    st.markdown(f"""
    <div class='alert-flare'>
        <div class='alert-title' style='color:#ff4444;'>
            🔴 ACTIVE FLARE DETECTED
        </div>
        <div class='alert-body'>
            CNN nowcast probability: {latest_nowcast_prob:.1%} &nbsp;|&nbsp;
            Threshold: {nowcast_cfg['best_threshold']:.2f} &nbsp;|&nbsp;
            Status: NOWCASTING POSITIVE
        </div>
    </div>
    """, unsafe_allow_html=True)
elif latest_forecast_alert:
    st.markdown(f"""
    <div class='alert-warning'>
        <div class='alert-title' style='color:#ff6b00;'>
            ⚠ FLARE FORECAST — 15 MINUTE WARNING
        </div>
        <div class='alert-body'>
            BiLSTM forecast probability: {latest_forecast_prob:.1%} &nbsp;|&nbsp;
            Threshold: {fore_thresh:.2f} &nbsp;|&nbsp;
            Lead time: 15 minutes
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class='alert-safe'>
        <div class='alert-title' style='color:#00ff88;'>
            ✓ SOLAR ACTIVITY NOMINAL
        </div>
        <div class='alert-body'>
            Nowcast: {latest_nowcast_prob:.1%} &nbsp;|&nbsp;
            Forecast: {latest_forecast_prob:.1%} &nbsp;|&nbsp;
            No flare activity detected
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── METRIC CARDS ──
st.markdown(f"""
<div class='metric-grid'>
    <div class='metric-card orange'>
        <div class='metric-label'>Nowcast Probability</div>
        <div class='metric-value orange'>{latest_nowcast_prob:.1%}</div>
    </div>
    <div class='metric-card {"red" if latest_forecast_alert else "blue"}'>
        <div class='metric-label'>Forecast Probability</div>
        <div class='metric-value {"red" if latest_forecast_alert else "blue"}'>{latest_forecast_prob:.1%}</div>
    </div>
    <div class='metric-card green'>
        <div class='metric-label'>Flare Peaks Detected</div>
        <div class='metric-value green'>{total_flares}</div>
    </div>
    <div class='metric-card blue'>
        <div class='metric-label'>Data Coverage</div>
        <div class='metric-value blue'>{total_minutes}m</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────
st.markdown("<div class='section-label'>X-Ray Light Curve</div>", unsafe_allow_html=True)
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
fig_lc = plot_lightcurve(df, nowcast_labels, nowcast_probas)
st.plotly_chart(fig_lc, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
st.markdown("</div>", unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("<div class='section-label'>15-Minute Forecast Timeline</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    if len(forecast_probas) > 0:
        fig_fore = plot_forecast_timeline(
            df_min, forecast_probas, fore_indices, fore_thresh
        )
        st.plotly_chart(fig_fore, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
    else:
        st.info("Need 120+ minutes of data for forecasting.")
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='section-label'>Forecast Gauge</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    if len(forecast_probas) > 0:
        fig_gauge = plot_gauge(latest_forecast_prob, fore_thresh)
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# FLARE DATABASE TABLE
# ─────────────────────────────────────────
st.markdown("<div class='section-label'>Flare Database</div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Nowcasted Solar Flares</div>",
            unsafe_allow_html=True)
st.markdown("<div class='section-card'>", unsafe_allow_html=True)

flare_events = df[df["is_true_flare"] == 1][
    ["datetime_utc", "counts_clean", "zscore"]
].copy()

if len(flare_events) > 0:
    # Add nowcast probability aligned by index
    df_tagged = df.iloc[60:].copy().reset_index(drop=True)
    df_tagged["nowcast_proba"] = nowcast_probas
    df_tagged["nowcast_label"] = nowcast_labels

    flare_db = df_tagged[df_tagged["is_true_flare"] == 1][
        ["datetime_utc", "counts_clean", "zscore", "nowcast_proba"]
    ].copy()

    flare_db.columns = ["Timestamp (UTC)", "Peak Counts", "Z-Score", "Nowcast P(flare)"]
    flare_db["Peak Counts"] = flare_db["Peak Counts"].round(1)
    flare_db["Z-Score"] = flare_db["Z-Score"].round(3)
    flare_db["Nowcast P(flare)"] = flare_db["Nowcast P(flare)"].apply(lambda x: f"{x:.1%}")
    flare_db["Timestamp (UTC)"] = flare_db["Timestamp (UTC)"].astype(str)

    st.dataframe(
        flare_db.reset_index(drop=True),
        use_container_width=True,
        height=min(400, 40 + len(flare_db) * 35)
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Download button
    csv_out = flare_db.to_csv(index=False)
    st.download_button(
        "⬇ Download Flare Database",
        csv_out,
        file_name=f"flare_database_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
else:
    st.markdown("</div>", unsafe_allow_html=True)
    st.info("No flare peaks detected in uploaded data.")

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<hr style='border-color:#1e1e2e; margin: 2rem 0 1rem;'>
<div style='text-align:center; font-size:0.75rem; color:#333344;
            font-family:JetBrains Mono;'>
    ISRO Build-A-Thon 2026 · PS15 · Aditya-L1 Solar Flare Detection ·
    CNN Nowcast + BiLSTM Forecast
</div>
""", unsafe_allow_html=True)