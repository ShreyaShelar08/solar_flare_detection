# Aditya-L1 Solar Flare Nowcasting Early Warning System

An end-to-end physics-informed machine learning pipeline for real-time solar flare nowcasting on the Lagrange Point 1 (L1) solar observer **Aditya-L1**. 

The system processes 1-second cadence telemetry from the **HEL1OS** (Hard X-ray Spectrometer) and **SoLEXS** (Solar Low Energy X-ray Spectrometer) instruments. It utilizes a **1D-CNN with Residual Skip Connections (ResNet-1D)** to predict imminent solar flares 10 to 15 minutes before their thermal impulsive peak.

---

## 📁 Repository Structure

* `src/data_processing.py`: Handles HEL1OS pivoting (fusing CZT1 and CZT2 bands), calculates spectral hardness ratios, resamples SoLEXS telemetry, and generates sliding windows.
* `src/nowcasting_model.py`: Implements the ResNet-1D PyTorch model class, Binary Focal Loss, training loop, and threshold optimization routines.
* `src/backend.py`: A FastAPI endpoint structure that hosts the model, streams telemetry, processes inputs, and maintains dynamic threshold configurations.
* `src/dashboard.py`: A premium scientific Streamlit dashboard plotting real-time telemetry, warning states, alert history, and presenting the Operator Calibration panel.
* `src/simulation.py`: Streams real-time playback telemetry of historical solar flare events on June 20, 2026, to verify the end-to-end pipeline.
* `models/nowcast_1dcnn.pt`: The optimized PyTorch 1D-CNN model weights.

---

## 🛠️ Setup and Installation

### 1. Create a Virtual Environment
Inside the project root directory, run:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
Install all required libraries:
```powershell
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline (Real-Time Demo)

To run the full real-time pipeline, open **three separate terminals** in VS Code (with the virtual environment activated in each):

### Terminal 1: Start the FastAPI Backend
Exposes ingestion and prediction endpoints on port 8000.
```powershell
python -m uvicorn src.backend:app --host 127.0.0.1 --port 8000
```

### Terminal 2: Start the Streamlit Dashboard
Launches the visualization and calibration control room UI.
```powershell
streamlit run src/dashboard.py
```

### Terminal 3: Start the Telemetry Simulator
Plays back telemetry of the June 20, 2026 solar storm event at a speedup factor of 20.
```powershell
python src/simulation.py 20
```

*Open your web browser and navigate to `http://localhost:8501` to monitor the live telemetry feed.*

---

## ⚙️ Warning Levels & Live Calibration

The system groups solar activity into four distinct states:
1. **Level 1: Quiet Sun (Normal)** ($P < 0.20$): Baseline solar emissions.
2. **Level 2: Active Region (Moderate)** ($0.20 \le P < 0.52$): Increased thermal output.
3. **Level 3: Flare Precursor (High)** ($0.52 \le P < 0.75$): Non-thermal electron acceleration detected.
4. **Level 4: Flare Imminent (Critical)** ($P \ge 0.75$): High-energy impulsive peak onset.

### Dynamic Calibration Sidebar (Dashboard)
Operators can fine-tune these warning boundaries dynamically without interrupting the live stream:
* **Threshold Sliders:** Slide to adjust the probability triggers for Levels 2, 3, and 4.
* **Cost-Utility Analyzer:** Weigh the cost of missed flares ($C_{\text{Missed}}$ - i.e., instrument damage) against false alarms ($C_{\text{False}}$ - i.e., unnecessary shutter closures). The dashboard will automatically compute and suggest the optimal Level 3 decision boundary ($T_{\text{opt}}$):
  $$T_{\text{opt}} = 0.52 - 0.15 \times \log_{10}\left(\frac{C_{\text{Missed}}}{C_{\text{False}}}\right)$$
* **Apply Settings:** Saves the config to the FastAPI server memory in real-time, immediately adapting the warning logic.

---

## 📈 Model Performance Summary
* **Optimal Boundary:** `0.52`
* **True Skill Statistic (TSS):** `0.7165` (Excellent skill above chance)
* **Heidke Skill Score (HSS):** `0.3715`
* **Sensitivity (Recall):** `83.52%` (Flags 83.5% of imminent flare starts)
* **Specificity:** `88.13%`
* **Precision:** `28.15%` (Controlled false alarm rate in highly imbalanced data)
