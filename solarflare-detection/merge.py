import pandas as pd

# Load both cleaned feature CSVs
df_sxr = pd.read_csv(r"C:\Users\Personal\Desktop\Shreya Shelar\Projects\ISRO\SoLEXS_combined_cleaned.csv", parse_dates=["datetime_utc"])
df_hxr = pd.read_csv(r"C:\Users\Personal\Desktop\Shreya Shelar\Projects\ISRO\HEL1OS_cleaned_Data.csv", parse_dates=["datetime_utc"])

# Rename columns to avoid conflicts
df_sxr = df_sxr.rename(columns={
    "counts_clean": "sxr_counts",
    "rolling_mean": "sxr_rolling_mean",
    "rolling_std":  "sxr_rolling_std",
    "zscore":       "sxr_zscore",
    "is_outlier_flare": "sxr_is_outlier"
})

df_hxr = df_hxr.rename(columns={
    "counts_per_sec_clean": "hxr_counts",
    "rolling_mean":         "hxr_rolling_mean",
    "rolling_std":          "hxr_rolling_std",
    "zscore":               "hxr_zscore",
    "is_outlier_flare":     "hxr_is_outlier"
})

# Keep only needed columns
sxr_cols = ["datetime_utc", "sxr_counts", "sxr_rolling_mean", "sxr_rolling_std", "sxr_zscore", "sxr_is_outlier"]
hxr_cols = ["datetime_utc", "hxr_counts", "hxr_rolling_mean", "hxr_rolling_std", "hxr_zscore", "hxr_is_outlier"]

df_sxr = df_sxr[sxr_cols]
df_hxr = df_hxr[hxr_cols]

# Merge on timestamp
df_merged = pd.merge_asof(
    df_sxr.sort_values("datetime_utc"),
    df_hxr.sort_values("datetime_utc"),
    on="datetime_utc",
    tolerance=pd.Timedelta("1s"),
    direction="nearest"
)

# Add nowcast output as additional feature
df_nowcast = pd.read_csv("nowcast_output.csv", parse_dates=["datetime_utc"])
df_nowcast = df_nowcast[["datetime_utc", "nowcast_label", "nowcast_proba"]]

df_merged = pd.merge_asof(
    df_merged.sort_values("datetime_utc"),
    df_nowcast.sort_values("datetime_utc"),
    on="datetime_utc",
    tolerance=pd.Timedelta("1s"),
    direction="nearest"
)

df_merged = df_merged.dropna().reset_index(drop=True)

df_merged.to_csv("merged_features.csv", index=False)

print("Merged shape:", df_merged.shape)
print("Columns:", df_merged.columns.tolist())
print(df_merged.head(3))