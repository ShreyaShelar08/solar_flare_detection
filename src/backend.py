import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

# Add current directory (src/) to module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nowcasting_model import Nowcast1DCNN

app = FastAPI(title="Aditya-L1 Solar Flare Nowcasting Early Warning System API")

# Global state buffers
# Nowcasting needs last 60 seconds of HEL1OS data (cadence: 1s, features: 12)
# Features: czt1_20_40, czt1_40_60, czt1_60_80, czt1_80_150, czt1_18_160,
#           czt2_20_40, czt2_40_60, czt2_60_80, czt2_80_150, czt2_18_160,
#           czt1_hardness_ratio, czt2_hardness_ratio
helios_buffer = deque(maxlen=60)

# Threshold configurations (Fine-tunable in real-time)
threshold_config = {
    "th_level2": 0.20,
    "th_level3": 0.52,
    "th_level4": 0.75
}

class ThresholdUpdateRequest(BaseModel):
    th_level2: float
    th_level3: float
    th_level4: float


# Latest prediction results and alert state
latest_prediction = {
    "timestamp": None,
    "solexs_counts": 0.0,
    "helios_counts_20_40": 0.0,
    "nowcast_probability": 0.0,
    "alert_state": "NORMAL", 
    "warning_level": 1,
    "warning_level_name": "Level 1: Quiet Sun (Normal)",
    "is_critical": False,
    "status": "Initializing buffers..."
}

# Model paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOWCAST_MODEL_PATH = os.path.join(BASE_DIR, "models", "nowcast_1dcnn.pt")

# Load models on startup
nowcast_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.on_event("startup")
def load_models():
    global nowcast_model
    
    # Load Nowcasting Model (12 features)
    if os.path.exists(NOWCAST_MODEL_PATH):
        try:
            nowcast_model = Nowcast1DCNN(num_features=12)
            nowcast_model.load_state_dict(torch.load(NOWCAST_MODEL_PATH, map_location=device))
            nowcast_model.to(device)
            nowcast_model.eval()
            print("Loaded Optimized Nowcasting 1D-CNN model successfully.")
        except Exception as e:
            print(f"Error loading Nowcasting model: {e}. Fallback to heuristics.")
            nowcast_model = None
    else:
        print("Nowcasting model file not found. System will run in HEURISTIC fallback mode for Nowcasting.")

class TelemetryPacket(BaseModel):
    timestamp: str
    solexs_counts: float
    czt1_20_40: float
    czt1_40_60: float
    czt1_60_80: float
    czt1_80_150: float
    czt1_18_160: float
    czt2_20_40: float
    czt2_40_60: float
    czt2_60_80: float
    czt2_80_150: float
    czt2_18_160: float

def run_nowcasting_inference():
    """
    Runs Nowcasting inference on the last 60 seconds of HEL1OS telemetry (12 features).
    """
    if len(helios_buffer) < 60:
        return 0.0 # Not enough data
        
    # Convert buffer to numpy array
    data = np.array(helios_buffer) # shape: (60, 12)
    
    if nowcast_model is not None:
        try:
            # Reshape for Conv1D: (batch_size, features, sequence_length) -> (1, 12, 60)
            X = np.transpose(data, (1, 0)) # shape: (12, 60)
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logit = nowcast_model(X_tensor)
                prob = torch.sigmoid(logit).cpu().item()
            return prob
        except Exception as e:
            print(f"Error in Nowcasting ML inference: {e}")
            # Fall through to heuristic fallback
            
    # Heuristic Fallback for Nowcasting:
    # Based on Hardness Ratio and CZT1/CZT2 (20-40 keV) counts
    # If CZT counts spike above baseline (e.g. 50 counts) or Hardness ratio is high
    czt20_40_1 = data[:, 0]
    czt20_40_2 = data[:, 5]
    
    # Calculate average counts and hardness ratios across both detectors
    avg_20_40 = (czt20_40_1 + czt20_40_2) / 2.0
    hr1 = data[:, 10]
    hr2 = data[:, 11]
    avg_hr = (hr1 + hr2) / 2.0
    
    # Calculate relative increase in HXR counts over the last 10 seconds compared to baseline
    baseline = np.mean(avg_20_40[:30]) if len(avg_20_40) >= 30 else 5.0
    latest = np.mean(avg_20_40[-5:])
    
    if baseline < 1.0: baseline = 1.0
    ratio = latest / baseline
    
    # Sigmoid function mapper
    prob = 1.0 / (1.0 + np.exp(-0.5 * (ratio - 3.0)))
    
    # If HXR is very quiet, make sure probability is low
    if latest < 10.0:
        prob *= 0.1
        
    return float(prob)

@app.post("/api/telemetry")
def append_telemetry(packet: TelemetryPacket):
    global latest_prediction
    
    try:
        dt = datetime.strptime(packet.timestamp, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            dt = datetime.fromisoformat(packet.timestamp.replace('Z', '+00:00'))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid timestamp format. Use YYYY-MM-DD HH:MM:SS or ISO format.")
            
    # 1. Update HEL1OS Buffer (12 features)
    # Calculate Hardness Ratios for both detectors
    hr1 = (packet.czt1_40_60 + 1e-5) / (packet.packet_czt1_20_40 if hasattr(packet, 'packet_czt1_20_40') else packet.czt1_20_40 + 1e-5)
    # Correcting check to prevent typos
    hr1 = (packet.czt1_40_60 + 1e-5) / (packet.czt1_20_40 + 1e-5)
    hr2 = (packet.czt2_40_60 + 1e-5) / (packet.czt2_20_40 + 1e-5)
    
    helios_row = [
        packet.czt1_20_40,
        packet.czt1_40_60,
        packet.czt1_60_80,
        packet.czt1_80_150,
        packet.czt1_18_160,
        packet.czt2_20_40,
        packet.czt2_40_60,
        packet.czt2_60_80,
        packet.czt2_80_150,
        packet.czt2_18_160,
        hr1,
        hr2
    ]
    helios_buffer.append(helios_row)
    
    # 2. Run Inferences
    nowcast_prob = run_nowcasting_inference()
    
    # 3. Warning Levels & Decision Logic (Aligned with Dynamic/Optimized Thresholds)
    is_critical = False
    th2 = threshold_config["th_level2"]
    th3 = threshold_config["th_level3"]
    th4 = threshold_config["th_level4"]
    
    if nowcast_prob < th2:
        level_idx = 1
        level_name = "Level 1: Quiet Sun (Normal)"
        status_msg = "Normal: Solar baseline activity is stable. No imminent flare threat detected."
        alert_state = "NORMAL"
    elif nowcast_prob < th3:
        level_idx = 2
        level_name = "Level 2: Active Region (Moderate)"
        status_msg = f"Moderate: Elevated counts detected. Monitoring active region (Threshold: {th2:.2f}-{th3:.2f})."
        alert_state = "MODERATE"
    elif nowcast_prob < th4:
        level_idx = 3
        level_name = "Level 3: Flare Precursor (High)"
        status_msg = f"High Warning: Non-thermal electron acceleration detected. Precursor phase active (Threshold: {th3:.2f}-{th4:.2f})."
        alert_state = "HIGH WARNING"
        is_critical = True
    else:
        level_idx = 4
        level_name = "Level 4: Flare Imminent (Critical)"
        status_msg = f"Critical Warning: Impulsive phase peak reached. Flare onset IMMINENT (Threshold: >= {th4:.2f})."
        alert_state = "CRITICAL"
        is_critical = True
        
    # Update global prediction dict
    latest_prediction = {
        "timestamp": packet.timestamp,
        "solexs_counts": packet.solexs_counts,
        "helios_counts_20_40": packet.czt1_20_40,
        "nowcast_probability": nowcast_prob,
        "alert_state": alert_state,
        "warning_level": level_idx,
        "warning_level_name": level_name,
        "is_critical": is_critical,
        "status": status_msg
    }
    
    return {"status": "success", "alert_state": alert_state, "warning_level": level_idx}

@app.get("/api/prediction")
def get_prediction():
    if latest_prediction["timestamp"] is None:
        latest_prediction["status"] = f"Waiting for live telemetry. Buffer: HEL1OS={len(helios_buffer)}/60"
    return latest_prediction

@app.get("/api/config")
def get_config():
    return threshold_config

@app.post("/api/config")
def update_config(req: ThresholdUpdateRequest):
    global threshold_config
    if not (0.0 < req.th_level2 < req.th_level3 < req.th_level4 <= 1.0):
        raise HTTPException(
            status_code=400, 
            detail="Invalid threshold hierarchy. Ensure: 0.0 < th_level2 < th_level3 < th_level4 <= 1.0"
        )
    threshold_config["th_level2"] = req.th_level2
    threshold_config["th_level3"] = req.th_level3
    threshold_config["th_level4"] = req.th_level4
    return {"status": "success", "updated_thresholds": threshold_config}
