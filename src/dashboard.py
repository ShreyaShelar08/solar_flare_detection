import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from collections import deque

# Page configuration
st.set_page_config(
    page_title="Aditya-L1 Solar Flare Nowcasting System",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp {
        background-color: #0d0e15;
        color: #f0f2f6;
    }
    
    /* Header card styling */
    .header-card {
        background: linear-gradient(135deg, #1e1f29 0%, #14151b 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #2e303f;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .header-title {
        color: #ff6b00;
        text-align: center;
        margin: 0;
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(255, 107, 0, 0.3);
    }
    .header-subtitle {
        color: #8a8d9f;
        text-align: center;
        margin: 8px 0 0 0;
        font-size: 1.1rem;
        letter-spacing: 1px;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background-color: #181922;
        border: 1px solid #282a36;
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    .metric-title {
        color: #8a8d9f;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Blink animation for critical alert */
    @keyframes blinker {
        50% { opacity: 0.3; }
    }
    .blink {
        animation: blinker 1.5s linear infinite;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get prediction from FastAPI backend
def fetch_prediction():
    try:
        r = requests.get("http://127.0.0.1:8000/api/prediction", timeout=0.5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# Helper functions to manage dynamic thresholds
def fetch_config():
    try:
        r = requests.get("http://127.0.0.1:8000/api/config", timeout=0.5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"th_level2": 0.20, "th_level3": 0.52, "th_level4": 0.75}

def update_config(th_level2, th_level3, th_level4):
    try:
        payload = {"th_level2": th_level2, "th_level3": th_level3, "th_level4": th_level4}
        r = requests.post("http://127.0.0.1:8000/api/config", json=payload, timeout=0.5)
        return r.status_code == 200
    except Exception:
        return False

# Session state initialization for historical telemetry plotting
if 'solexs_history' not in st.session_state:
    st.session_state.solexs_history = deque(maxlen=300) 
if 'helios_history' not in st.session_state:
    st.session_state.helios_history = deque(maxlen=300)
if 'time_history' not in st.session_state:
    st.session_state.time_history = deque(maxlen=300)
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []

# Sidebar Calibration Panel
st.sidebar.markdown("## ⚙️ Calibration Panel")
st.sidebar.markdown("Fine-tune warning thresholds and evaluate the cost-utility trade-off.")

# Fetch current config from backend or session state
if 'th_level2' not in st.session_state or 'th_level3' not in st.session_state or 'th_level4' not in st.session_state:
    cfg = fetch_config()
    st.session_state.th_level2 = cfg["th_level2"]
    st.session_state.th_level3 = cfg["th_level3"]
    st.session_state.th_level4 = cfg["th_level4"]

# Interactive Warning Level Thresholds
st.sidebar.subheader("Probability Boundaries")
th_l2 = st.sidebar.slider(
    "Level 2 (Active Region)",
    min_value=0.05, max_value=0.40, 
    value=float(st.session_state.th_level2), 
    step=0.01,
    help="Default: 0.20. Probability where the active region warning triggers."
)
th_l3 = st.sidebar.slider(
    "Level 3 (Flare Precursor)",
    min_value=0.40, max_value=0.70, 
    value=float(st.session_state.th_level3), 
    step=0.01,
    help="Default: 0.52. Core optimized TSS decision boundary. Transition to precursor warning."
)
th_l4 = st.sidebar.slider(
    "Level 4 (Flare Imminent)",
    min_value=0.70, max_value=0.95, 
    value=float(st.session_state.th_level4), 
    step=0.01,
    help="Default: 0.75. Imminent flare trigger threshold."
)

# Cost-Utility Calculator
st.sidebar.markdown("---")
st.sidebar.subheader("Cost-Utility Analyzer")
st.sidebar.markdown(
    "Assess the trade-off between missing a flare (potential instrument damage) "
    "and triggering false alarms (unnecessary shutter closures)."
)
c_missed = st.sidebar.slider(
    "Cost of Missed Flare (C_Missed)", 
    min_value=1.0, max_value=100.0, 
    value=50.0, step=1.0,
    help="The severity weight of missing a real flare event."
)
c_false = st.sidebar.slider(
    "Cost of False Alarm (C_False)", 
    min_value=1.0, max_value=20.0, 
    value=5.0, step=1.0,
    help="The cost/inconvenience of triggering false warnings."
)

# Calculate recommended threshold
# Formula: Suggest lower threshold if C_Missed / C_False is high (sensitivity preferred)
ratio = c_missed / c_false
suggested_th = 0.52 - 0.15 * np.log10(ratio)
suggested_th = float(np.clip(suggested_th, 0.30, 0.70))

st.sidebar.info(f"💡 Recommended Level 3 Threshold: **{suggested_th:.2f}**")

use_suggested = st.sidebar.button("Apply Suggested Threshold")
if use_suggested:
    # Set the value in session state and rerun to update slider
    st.session_state.th_level3 = suggested_th
    st.rerun()

# Save Config Button
if st.sidebar.button("Apply Calibration Settings", type="primary"):
    # Validate hierarchy
    if not (0.0 < th_l2 < th_l3 < th_l4 <= 1.0):
        st.sidebar.error("Invalid hierarchy! Ensure Level 2 < Level 3 < Level 4.")
    else:
        success = update_config(th_l2, th_l3, th_l4)
        if success:
            st.session_state.th_level2 = th_l2
            st.session_state.th_level3 = th_l3
            st.session_state.th_level4 = th_l4
            st.sidebar.success("Settings applied to Backend!")
            time.sleep(0.5)
            st.rerun()
        else:
            st.sidebar.error("Failed to update backend configuration.")

# Header Banner
st.markdown("""
<div class="header-card">
    <h1 class="header-title">☀️ ADITYA-L1 SOLAR FLARE NOWCASTING SYSTEM</h1>
    <div class="header-subtitle">Real-Time In-Situ Solar Flare Early Warning Detection Pipeline (1D-CNN Model)</div>
</div>
""", unsafe_allow_html=True)

# Main container for reactive updates
placeholder = st.empty()

# Run the live refresh loop
while True:
    data = fetch_prediction()
    
    # Store latest data points if we got them
    if data and data.get("timestamp"):
        ts_str = data["timestamp"]
        try:
            ts_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            time_label = ts_dt.strftime("%H:%M:%S")
        except Exception:
            time_label = ts_str.split()[-1] if len(ts_str.split()) > 1 else ts_str
            
        st.session_state.solexs_history.append(data["solexs_counts"])
        st.session_state.helios_history.append(data["helios_counts_20_40"])
        st.session_state.time_history.append(time_label)
        
        # Add to alert history if state changes
        if not st.session_state.alert_history or st.session_state.alert_history[-1]["level"] != data["warning_level"]:
            st.session_state.alert_history.append({
                "time": time_label,
                "level": data["warning_level"],
                "alert_state": data["alert_state"],
                "details": data["status"]
            })
            if len(st.session_state.alert_history) > 10:
                st.session_state.alert_history.pop(0)

    # Render dashboard contents inside the placeholder
    with placeholder.container():
        
        # 1. Alert Status Banner
        if not data:
            st.markdown("""
            <div style="background-color: #2b181c; color: #ff5f74; padding: 25px; border-radius: 12px; border-left: 10px solid #d9383a; text-align: center; font-size: 20px; font-weight: bold;">
                📡 CONNECTING... FastAPI backend not responding. Ensure uvicorn server is running on port 8000.
            </div>
            """, unsafe_allow_html=True)
        else:
            w_level = data["warning_level"]
            status_text = data["status"]
            
            if w_level == 1:
                # Level 1: Quiet (Normal) - Green
                st.markdown(f"""
                <div style="background-color: #0b2d18; color: #bbf7d0; padding: 25px; border-radius: 12px; border-left: 10px solid #10b981; font-size: 20px; font-weight: bold; margin-bottom: 25px;">
                    🟢 {status_text}
                </div>
                """, unsafe_allow_html=True)
            elif w_level == 2:
                # Level 2: Active Region (Moderate) - Yellow
                st.markdown(f"""
                <div style="background-color: #3e3b07; color: #fef08a; padding: 25px; border-radius: 12px; border-left: 10px solid #eab308; font-size: 20px; font-weight: bold; margin-bottom: 25px;">
                    🟡 {status_text}
                </div>
                """, unsafe_allow_html=True)
            elif w_level == 3:
                # Level 3: Flare Precursor (High) - Orange
                st.markdown(f"""
                <div style="background-color: #3f2106; color: #fed7aa; padding: 25px; border-radius: 12px; border-left: 10px solid #f97316; font-size: 20px; font-weight: bold; margin-bottom: 25px;">
                    🟠 {status_text}
                </div>
                """, unsafe_allow_html=True)
            elif w_level == 4:
                # Level 4: Flare Imminent (Critical) - Red
                st.markdown(f"""
                <div class="blink" style="background-color: #450a0a; color: #fca5a5; padding: 25px; border-radius: 12px; border-left: 10px solid #ef4444; font-size: 22px; font-weight: bold; margin-bottom: 25px; box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);">
                    🚨 {status_text}
                </div>
                """, unsafe_allow_html=True)

        # 2. Main 2-Column Layout
        col_telemetry, col_engines = st.columns([2, 1])
        
        with col_telemetry:
            st.markdown("### 📊 REAL-TIME INSTRUMENT TELEMETRY")
            
            if len(st.session_state.solexs_history) > 0:
                plot_df = pd.DataFrame({
                    "Timestamp": list(st.session_state.time_history),
                    "SoLEXS Soft X-Rays": list(st.session_state.solexs_history),
                    "HEL1OS Hard X-Rays": list(st.session_state.helios_history)
                }).set_index("Timestamp")
                
                # Plot SoLEXS SXR (Thermal)
                st.markdown("**SoLEXS SDD2 Cleaned Counts (Soft X-Ray Flux - Thermal)**")
                st.line_chart(plot_df["SoLEXS Soft X-Rays"], color="#ffd700", height=200)
                
                # Plot HEL1OS HXR (Non-Thermal)
                st.markdown("**HEL1OS CZT1 20-40 keV Cleaned Counts/Sec (Hard X-Ray Rate - Non-Thermal)**")
                st.line_chart(plot_df["HEL1OS Hard X-Rays"], color="#ff4500", height=200)
            else:
                st.info("Awaiting telemetry stream to plot charts...")
                
            # Alert History Log
            st.markdown("### 📜 ALERT LOG HISTORY")
            if st.session_state.alert_history:
                log_df = pd.DataFrame(st.session_state.alert_history[::-1])
                st.table(log_df)
            else:
                st.text("No alerts recorded yet.")

        with col_engines:
            st.markdown("### 🧠 REAL-TIME DECISION ENGINE")
            
            if data:
                # Criticality Check
                is_crit = data["is_critical"]
                crit_text = "CRITICAL STATE ACTIVE" if is_crit else "SAFE / NON-CRITICAL"
                crit_color = "#ef4444" if is_crit else "#10b981"
                
                st.markdown(f"""
                <div class="metric-card" style="border: 2px solid {crit_color};">
                    <div class="metric-title">Criticality Status</div>
                    <div class="metric-value" style="color: {crit_color}; font-size: 2.2rem;">{crit_text}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Warning Level Display
                level_idx = data["warning_level"]
                level_name = data["warning_level_name"]
                
                if level_idx == 4:
                    lvl_color = "#ef4444" # red
                elif level_idx == 3:
                    lvl_color = "#f97316" # orange
                elif level_idx == 2:
                    lvl_color = "#eab308" # yellow
                else:
                    lvl_color = "#10b981" # green
                    
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Warning Level</div>
                    <div class="metric-value" style="color: {lvl_color}; font-size: 2.2rem;">LEVEL {level_idx}</div>
                    <div style="font-size: 1.1rem; color: #8a8d9f; font-weight: bold; margin-top: 5px;">{level_name}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Nowcasting Probability
                nowcast_val = data["nowcast_probability"]
                prob_percent = int(nowcast_val * 100)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Nowcast Probability (0-15m horizon)</div>
                    <div class="metric-value" style="color: {lvl_color};">{prob_percent}%</div>
                    <div style="background-color: #2a2c35; border-radius: 5px; height: 12px; width: 100%; margin-top: 15px;">
                        <div style="background-color: {lvl_color}; height: 12px; border-radius: 5px; width: {prob_percent}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Live Metrics Table
                st.markdown("#### Latest Values")
                st.markdown(f"""
                <div class="metric-card">
                    <table style="width: 100%; border-collapse: collapse; font-family: monospace;">
                        <tr style="border-bottom: 1px solid #282a36;">
                            <td style="padding: 10px 0; color: #8a8d9f;">SoLEXS SXR Counts</td>
                            <td style="padding: 10px 0; text-align: right; font-weight: bold; color: #ffd700;">{data["solexs_counts"]:.1f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px 0; color: #8a8d9f;">HEL1OS HXR Count Rate</td>
                            <td style="padding: 10px 0; text-align: right; font-weight: bold; color: #ff4500;">{data["helios_counts_20_40"]:.1f} c/s</td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="font-size: 0.8rem; color: #8a8d9f; text-align: center; margin-top: 20px;">
                    Last packet timestamp: {data["timestamp"]}<br>
                    Engine Status: ACTIVE &bull; Resampling: 1s
                </div>
                """, unsafe_allow_html=True)
                
    time.sleep(1.0)
