import os
import pandas as pd
import numpy as np

def clean_and_pivot_helios(file_path):
    """
    Loads the HEL1OS dataset, rounds timestamps to the nearest second,
    and pivots energy bands/detectors into columns.
    """
    print(f"Loading HEL1OS data from {file_path}...")
    # Use only necessary columns to save memory
    usecols = ['datetime_utc', 'detector', 'energy_band', 'counts_per_sec_clean', 'is_outlier_flare']
    df = pd.read_csv(file_path, usecols=usecols)
    
    # Parse timestamps and round to 1-second resolution
    df['datetime'] = pd.to_datetime(df['datetime_utc']).dt.round('1s')
    
    # Map energy bands to cleaner column names
    band_mapping = {
        'CZT1_LC_BAND_20.00KEV_TO_40.00KEV': 'czt1_20_40',
        'CZT1_LC_BAND_40.00KEV_TO_60.00KEV': 'czt1_40_60',
        'CZT1_LC_BAND_60.00KEV_TO_80.00KEV': 'czt1_60_80',
        'CZT1_LC_BAND_80.00KEV_TO_150.00KEV': 'czt1_80_150',
        'CZT1_LC_BAND_18.00KEV_TO_160.00KEV': 'czt1_18_160',
        'CZT2_LC_BAND_20.00KEV_TO_40.00KEV': 'czt2_20_40',
        'CZT2_LC_BAND_40.00KEV_TO_60.00KEV': 'czt2_40_60',
        'CZT2_LC_BAND_60.00KEV_TO_80.00KEV': 'czt2_60_80',
        'CZT2_LC_BAND_80.00KEV_TO_150.00KEV': 'czt2_80_150',
        'CZT2_LC_BAND_18.00KEV_TO_160.00KEV': 'czt2_18_160'
    }
    df['feature_name'] = df.apply(lambda r: band_mapping.get(r['energy_band'], f"{r['detector']}_{r['energy_band']}"), axis=1)
    
    # Pivot the table: index is datetime, columns are feature names, values are counts_per_sec_clean
    print("Pivoting HEL1OS dataset...")
    df_pivot = df.pivot_table(
        index='datetime',
        columns='feature_name',
        values='counts_per_sec_clean',
        aggfunc='mean'
    ).fillna(0)
    
    # Map and group is_outlier_flare column to join pivoted data
    df['is_outlier_flare_int'] = df['is_outlier_flare'].map({'FALSE': 0, 'TRUE': 1, False: 0, True: 1, 'False': 0, 'True': 1}).fillna(0).astype(int)
    df_outliers = df.groupby('datetime')['is_outlier_flare_int'].max()
    df_pivot['is_outlier_flare'] = df_outliers
    
    # Add Spectral Hardness Ratios
    # Hardness Ratio = CZT (40-60 keV) / CZT (20-40 keV)
    df_pivot['czt1_hardness_ratio'] = (df_pivot['czt1_40_60'] + 1e-5) / (df_pivot['czt1_20_40'] + 1e-5)
    df_pivot['czt2_hardness_ratio'] = (df_pivot['czt2_40_60'] + 1e-5) / (df_pivot['czt2_20_40'] + 1e-5)
    
    return df_pivot

def load_and_preprocess_solexs(file_path):
    """
    Loads the SoLEXS dataset, resamples to a strict 1-second cadence,
    interpolates missing values, and computes Neupert features (derivatives and integrals).
    """
    print(f"Loading SoLEXS data from {file_path}...")
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime_utc']).dt.round('1s')
    
    # Drop duplicates in timestamp
    df = df.drop_duplicates(subset=['datetime'])
    df = df.set_index('datetime').sort_index()
    
    # Resample to full 1-second grid to handle any missing seconds
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1s')
    df = df.reindex(full_range)
    df.index.name = 'datetime'
    
    # Interpolate missing counts_clean values
    df['counts_clean'] = df['counts_clean'].interpolate(method='linear').fillna(0)
    
    # Smooth counts_clean first to reduce noise in the derivative
    df['counts_smooth'] = df['counts_clean'].rolling(window=5, min_periods=1, center=True).mean()
    
    # Compute Neupert Features
    # 1. SXR Derivative (rate of thermal heating, tracks non-thermal particle acceleration)
    df['sxr_derivative'] = np.gradient(df['counts_smooth'].values)
    
    # 2. SXR rolling integral (thermal energy accumulation, e.g., over a 10-minute window)
    df['sxr_integral_10m'] = df['counts_clean'].rolling(window=600, min_periods=1).sum()
    
    return df

def generate_nowcasting_labels(df_solexs, horizon_secs=900, quiet_threshold=50, flare_threshold=100):
    """
    Generates Nowcasting labels.
    A positive label (1) represents a 'flare imminent within next horizon_secs (e.g. 15 mins)'
    where the current SXR flux is relatively quiet but will exceed the flare_threshold soon.
    """
    # Max SXR counts in the future window (shifted backwards)
    future_max = df_solexs['counts_clean'].shift(-horizon_secs).rolling(window=horizon_secs, min_periods=1).max()
    
    # Flare imminent condition: SXR will exceed flare_threshold in next 15 mins, and current SXR is quiet
    df_solexs['label_nowcast'] = ((future_max >= flare_threshold) & (df_solexs['counts_clean'] < quiet_threshold)).astype(int)
    return df_solexs

def prepare_nowcasting_windows(df_helios, window_size=60, step_size=10):
    """
    Prepares sliding windows from HEL1OS data for training the Nowcasting model.
    The label is the current outlier flare state at the end of the 60-second window.
    """
    feature_cols = [
        'czt1_20_40', 'czt1_40_60', 'czt1_60_80', 'czt1_80_150', 'czt1_18_160',
        'czt2_20_40', 'czt2_40_60', 'czt2_60_80', 'czt2_80_150', 'czt2_18_160',
        'czt1_hardness_ratio', 'czt2_hardness_ratio'
    ]
    
    X_list = []
    y_list = []
    
    data_feat = df_helios[feature_cols].values
    data_labels = df_helios['is_outlier_flare'].values
    
    # Slide a window over the data
    n_samples = len(df_helios)
    for start in range(0, n_samples - window_size, step_size):
        end = start + window_size
        X_list.append(data_feat[start:end])
        y_list.append(data_labels[end - 1]) # Label at the last second of the window
        
    return np.array(X_list), np.array(y_list)
