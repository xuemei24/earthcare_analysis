#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import sys
from scipy.spatial import cKDTree

# --- Utilities ---
def get_AOD(aod_old, wave_old, wave_new, angstrom):
    return ((wave_new / wave_old) ** (-angstrom)) * aod_old

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def latlon_to_xyz(lat, lon):
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.vstack((x, y, z)).T

# --- Configuration ---
months = ['january','february','march','april','may','june','july','august','september','october','november','december']
month2s = [str(nmonth).zfill(2) for nmonth in range(1,13)]
fmonths = ['January','February','March','April','May','June','July','August','September','October','November','December']

all_detailed_matches = [] # List to hold all points for the entire year

for fmonth, month2, month in zip(fmonths, month2s, months):
    year = '2024' if month == 'december' else '2025'
    time_tol = pd.Timedelta("30min")
    
    print(f'Processing {month} {year}...')

    # --- Load AERONET ---
    aeronet_path = '/scratch/nld6854/earthcare/aeronet/'
    try:
        df_aeronet_raw = pd.read_table(f"{aeronet_path}{year}{month2}_all_sites_aod15_allpoints.txt", delimiter=',', header=[7])
        df_aeronet_raw = df_aeronet_raw.replace(-999, np.nan)
        
        # Calculate AOD 355
        aod355 = get_AOD(df_aeronet_raw['AOD_340nm'], 340, 355, df_aeronet_raw['340-440_Angstrom_Exponent'])
        
        # DateTime handling
        dt_str = df_aeronet_raw["Date(dd:mm:yyyy)"] + " " + df_aeronet_raw["Time(hh:mm:ss)"]
        df_aer_f = pd.DataFrame({
            "time": pd.to_datetime(dt_str, format="%d:%m:%Y %H:%M:%S"),
            "lat": df_aeronet_raw["Site_Latitude(Degrees)"],
            "lon": df_aeronet_raw["Site_Longitude(Degrees)"],
            "aeronet_aod": aod355
        }).dropna(subset=['lat', 'lon', 'time', 'aeronet_aod'])
    except Exception as e:
        print(f"Skipping AERONET for {month}: {e}")
        continue

    # --- Load ATLID ---
    file_dir = '/scratch/nld6854/earthcare/cams_data/'
    atlid_file = f"{file_dir}{month}_{year}/{year}_{month}_cams_atlid_co-located_aod_snr_gr_2.txt"
    
    if not os.path.exists(atlid_file):
        print(f"File missing: {atlid_file}")
        continue
        
    df_atlid = pd.read_csv(atlid_file, delimiter=",")
    
    # Process ATLID time and mask
    all_time = np.datetime64("1970-01-01T00:00:00") + df_atlid["time"].values.astype("timedelta64[ns]")
    a_aod = df_atlid['aod355nm_atlid'].values
    all_lat = df_atlid['# latitude'].values
    all_lon = df_atlid['longitude'].values
    
    mask_atl = np.isfinite(all_lat) & np.isfinite(all_lon) & np.isfinite(a_aod)
    
    all_lat_f = all_lat[mask_atl]
    all_lon_f = all_lon[mask_atl]
    a_aod_f = a_aod[mask_atl]
    all_time_f = all_time[mask_atl]

    # --- Spatial Query (KDTree) ---
    atlid_xyz = latlon_to_xyz(np.radians(all_lat_f), np.radians(all_lon_f))
    aer_xyz = latlon_to_xyz(np.radians(df_aer_f["lat"].values), np.radians(df_aer_f["lon"].values))
    
    tree = cKDTree(atlid_xyz)
    max_rad = 100 / 6371.0
    neighbors = tree.query_ball_point(aer_xyz, r=max_rad)

    # --- Detailed Co-location Loop ---
    for i in range(len(df_aer_f)):
        inds = neighbors[i]
        if len(inds) == 0: continue

        # Filter by time
        t_aer = df_aer_f["time"].iloc[i].to_datetime64()
        time_diff = np.abs(all_time_f[inds] - t_aer) / np.timedelta64(1, "m")
        time_mask = time_diff <= 30
        
        if np.any(time_mask):
            valid_inds = np.array(inds)[time_mask]
            
            # Generate a unique ID for this overpass event
            overpass_id = f"{year}_{month2}_{i}"
            
            # Distance for all matched points
            dists = haversine(df_aer_f["lat"].iloc[i], df_aer_f["lon"].iloc[i], 
                              all_lat_f[valid_inds], all_lon_f[valid_inds])
            
            # Add every matched ATLID point individually
            for j, idx in enumerate(valid_inds):
                all_detailed_matches.append({
                    "overpass_id": overpass_id,
                    "month": month,
                    "aeronet_time": t_aer,
                    "aeronet_lat": df_aer_f["lat"].iloc[i],    # Added AERONET Lat
                    "aeronet_lon": df_aer_f["lon"].iloc[i],    # Added AERONET Lon
                    "aeronet_aod": df_aer_f["aeronet_aod"].iloc[i],
                    "atlid_time": all_time_f[idx],
                    "atlid_lat": all_lat_f[idx],               # Added ATLID Lat
                    "atlid_lon": all_lon_f[idx],               # Added ATLID Lon
                    "atlid_aod": a_aod_f[idx],
                    "dist_km": dists[j],
                    "time_diff_min": time_diff[time_mask][j]
                })

# --- Convert to DataFrame and Save ---
df_final = pd.DataFrame(all_detailed_matches)

# Optional: Add a column for the distance error
# This helps you see if the 'insane' points happen when the satellite is far away
df_final.to_csv("/scratch/nld6854/earthcare/cams_data/122024-112025_detailed_yearly_atlid_aeronet_detailed_matches.csv", index=False)

# --- Analysis: Address the "Insane Low" weighting problem ---
# 1. Group by Overpass ID to get 1:1 averages (Weight is now per event, not per point)
df_event_based = df_final.groupby("overpass_id").agg({
    "month": "first",
    "aeronet_aod": "mean",
    "atlid_aod": "mean", # This averages the 20 ATLID points into 1 value for that event
    "dist_km": "min"
})

# 2. Filter out the "insanely low" values if they are physical outliers
# (e.g., ATLID < 0.01 while AERONET is high)
clean_events = df_event_based[~((df_event_based['atlid_aod'] < 0.02) & (df_event_based['aeronet_aod'] < 0.02))]

print(f"Total Match-up Events: {len(df_event_based)}")
print(f"Corrected Yearly ATLID AOD: {clean_events['atlid_aod'].mean():.4f}")
print(f"Corrected Yearly AERONET AOD: {clean_events['aeronet_aod'].mean():.4f}")
