import sys
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def simulate_stream(file_path, speedup=10.0, start_time_str="2026-06-20 01:00:00", end_time_str="2026-06-20 02:30:00"):
    """
    Simulates real-time telemetry streaming by reading a slice of SoLEXS data,
    generating physically consistent HEL1OS hard X-ray counts using the Neupert Effect,
    and posting it to the FastAPI backend.
    """
    print(f"Loading SoLEXS data for simulation from {file_path}...")
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime_utc']).dt.tz_localize(None)
    
    # Filter for the specific flare event slice
    start_dt = pd.to_datetime(start_time_str)
    end_dt = pd.to_datetime(end_time_str)
    
    slice_df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)].copy()
    slice_df = slice_df.sort_values(by='datetime')
    
    if len(slice_df) == 0:
        print(f"No records found in range {start_time_str} to {end_time_str}. Streaming the first 2000 rows instead.")
        slice_df = df.head(2000).copy()
        
    print(f"Simulation slice size: {len(slice_df)} seconds of telemetry.")
    
    # Calculate derivative of SoLEXS counts to simulate HEL1OS HXR counts (Neupert Effect)
    # Neupert Effect: HXR flux is proportional to d/dt (SXR flux)
    slice_df['counts_smooth'] = slice_df['counts_clean'].rolling(window=7, min_periods=1, center=True).mean()
    deriv = np.gradient(slice_df['counts_smooth'].values)
    # Clip derivative to positive values only, as HXR counts are positive
    deriv_positive = np.maximum(0, deriv)
    
    # Scale factor: map a derivative of 1.0 to about 30 counts/sec of HXR CZT1_20_40
    hxr_base = deriv_positive * 40.0
    
    # Add baseline noise to HEL1OS
    np.random.seed(42)
    noise_20_40 = np.random.poisson(lam=2, size=len(slice_df))
    noise_40_60 = np.random.poisson(lam=1, size=len(slice_df))
    noise_60_80 = np.random.poisson(lam=0.5, size=len(slice_df))
    noise_80_150 = np.random.poisson(lam=0.2, size=len(slice_df))
    noise_18_160 = np.random.poisson(lam=4, size=len(slice_df))
    
    slice_df['czt1_20_40'] = hxr_base + noise_20_40
    slice_df['czt1_40_60'] = hxr_base * 0.4 + noise_40_60
    slice_df['czt1_60_80'] = hxr_base * 0.15 + noise_60_80
    slice_df['czt1_80_150'] = hxr_base * 0.05 + noise_80_150
    # CZT1 18-160 keV is the wide band (sum of lower bands)
    slice_df['czt1_18_160'] = slice_df['czt1_20_40'] + slice_df['czt1_40_60'] + slice_df['czt1_60_80'] + noise_18_160
    
    # CZT2 is slightly less sensitive in this simulation (simulating multi-detector response)
    slice_df['czt2_20_40'] = slice_df['czt1_20_40'] * 0.85
    slice_df['czt2_40_60'] = slice_df['czt1_40_60'] * 0.85
    slice_df['czt2_60_80'] = slice_df['czt1_60_80'] * 0.85
    slice_df['czt2_80_150'] = slice_df['czt1_80_150'] * 0.85
    slice_df['czt2_18_160'] = slice_df['czt1_18_160'] * 0.85
    
    # Calculate sleep interval
    delay = 1.0 / speedup
    print(f"Starting stream simulation at speedup of {speedup}x (Interval: {delay:.3f} seconds)...")
    
    url = "http://127.0.0.1:8000/api/telemetry"
    
    count = 0
    for idx, row in slice_df.iterrows():
        packet = {
            "timestamp": row['datetime'].strftime("%Y-%m-%d %H:%M:%S"),
            "solexs_counts": float(row['counts_clean']),
            "czt1_20_40": float(row['czt1_20_40']),
            "czt1_40_60": float(row['czt1_40_60']),
            "czt1_60_80": float(row['czt1_60_80']),
            "czt1_80_150": float(row['czt1_80_150']),
            "czt1_18_160": float(row['czt1_18_160']),
            "czt2_20_40": float(row['czt2_20_40']),
            "czt2_40_60": float(row['czt2_40_60']),
            "czt2_60_80": float(row['czt2_60_80']),
            "czt2_80_150": float(row['czt2_80_150']),
            "czt2_18_160": float(row['czt2_18_160'])
        }
        
        try:
            r = requests.post(url, json=packet, timeout=0.2)
            if r.status_code == 200:
                sys.stdout.write(f"\rPosted packet {count+1}/{len(slice_df)}: TS={packet['timestamp']} | SXR={packet['solexs_counts']:.1f} | HXR={packet['czt1_20_40']:.1f} | Alert={r.json().get('alert_state')}")
                sys.stdout.flush()
            else:
                print(f"\nFailed to post: {r.status_code} - {r.text}")
        except requests.exceptions.RequestException as e:
            print(f"\nConnection error: {e}. Is the FastAPI backend running?")
            time.sleep(2)
            
        count += 1
        time.sleep(delay)
        
    print("\nSimulation complete.")

if __name__ == '__main__':
    speed = 10.0
    if len(sys.argv) > 1:
        try:
            speed = float(sys.argv[1])
        except ValueError:
            pass
            
    simulate_stream('SoLEXS_combined_cleaned.csv', speedup=speed)
