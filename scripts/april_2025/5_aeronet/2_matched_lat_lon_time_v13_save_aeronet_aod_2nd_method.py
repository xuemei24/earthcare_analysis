#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import display, HTML
#from IPython import display#, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

import pandas as pd
import numpy as np
import xarray as xr
from importlib import reload
import matplotlib.pyplot as plt
import glob
import seaborn as sns
sns.set_style('ticks')
sns.set_context('poster')
import os
from matplotlib.colors import LogNorm, Normalize
import matplotlib
#print(dir(ectools))
import getopt
import shutil
import zipfile
from scipy.ndimage import gaussian_filter
from geopy.distance import geodesic
from datetime import datetime
import sys
from netCDF4 import Dataset
from pylab import *
from scipy.interpolate import interp1d
from matplotlib.colors import ListedColormap, BoundaryNorm

script_path = '/home/nld6854/earthcare_scripts/scripts/april_2025'
sys.path.append(script_path)

from scipy.spatial import cKDTree
from multiprocessing import Pool
from pathlib import Path


def get_AOD(aod_old, wave_old, wave_new, angstrom):
    print('aod_old>0',aod_old[aod_old>0], 'wave_old',wave_old, 'wave_new',wave_new, 'angstrom>0',angstrom[angstrom>0])
    return ((wave_new / wave_old) ** (-angstrom)) * aod_old

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians,
                                 [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

month = 'november'
month2 = '11'
fmonth = 'November'
print('month=',month)

months = ['january','february','march','april','may','june','july','august','september','october','november','december']
month2s = [str(nmonth) if nmonth>9 else '0'+str(nmonth) for nmonth in range(1,13)]
fmonths = ['January','February','March','April','May','June','July','August','September','October','November','December']

for fmonth,month2,month in zip(fmonths,month2s,months):
    year = '2024' if month == 'december' else '2025'
    time_tol = pd.Timedelta("30min")
    
    #####AERONET#####
    aeronet_path = '/scratch/nld6854/earthcare/aeronet/'
    df = pd.read_table(aeronet_path+year+month2+'_all_sites_aod15_allpoints.txt', delimiter=',', header=[7])
    df = df.replace(-999,np.nan)
    
    sites = df['AERONET_Site']
    print(sites)
    print(len(sites))
    aod340 = df['AOD_340nm']
    angstrom = df['340-440_Angstrom_Exponent']
    aod355 = get_AOD(aod340,340,355,angstrom)
    df['AOD_355nm']=aod355
    
    datetime = df["Date(dd:mm:yyyy)"] + " " + df["Time(hh:mm:ss)"]
    datetime2 = pd.to_datetime(datetime, format="%d:%m:%Y %H:%M:%S")
    df["DateTime"] = pd.to_datetime(
        df["Date(dd:mm:yyyy)"] + " " + df["Time(hh:mm:ss)"],
        format="%d:%m:%Y %H:%M:%S")
    
    df = df.rename(columns={
        "Site_Latitude(Degrees)": "latitude",
        "Site_Longitude(Degrees)": "longitude"})
    
    df_aeronet = pd.DataFrame({
        "time": df["DateTime"],
        "lat": df["latitude"],
        "lon": df["longitude"],
        "aeronet_aod": aod355})
    print(df)
    print(df_aeronet['time'])
    
    
    
    def find_nearest_time(t):
        diffs = (df_aeronet["time"] - t).abs()
        idx = diffs.idxmin()
        if diffs.loc[idx] <= time_tol:
            return df_aeronet.loc[idx, "time"]
        else:
            return pd.NaT  # no valid match
    
    
    ######ATLID#####
    file_dir = '/scratch/nld6854/earthcare/cams_data/'
    day_or_night = ''
    df = pd.read_csv(file_dir+month+'_'+year+'/'+year+"_"+month+"_cams_atlid_co-located_"+day_or_night+"aod_snr_gr_2.txt", delimiter=",")
    a_aod = df['aod355nm_atlid'].values
    all_lat = df['# latitude'].values
    all_lon = df['longitude'].values
    #all_time = df['time'].values
    all_time = np.datetime64("1970-01-01T00:00:00") + \
                 df["time"].values.astype("timedelta64[ns]")
    
    # AERONET mask
    mask_aer = (np.isfinite(df_aeronet["lat"].values) &
        np.isfinite(df_aeronet["lon"].values) &
        df_aeronet["time"].notna().values)
        
    # ATLID mask
    mask_atl = (np.isfinite(all_lat) &
        np.isfinite(all_lon) &
        np.isfinite(a_aod) &
        np.isfinite(all_time.astype("datetime64[ns]").astype("int64")))
    
    
    # Filter ATLID
    all_lat_f = all_lat[mask_atl]
    all_lon_f = all_lon[mask_atl]
    a_aod_f   = a_aod[mask_atl]
    all_time_f = all_time[mask_atl]
    
    # Filter AERONET
    df_aer_f = df_aeronet.loc[mask_aer].reset_index(drop=True)
    
    # convert lat/lon to radians
    all_lat_rad = np.radians(all_lat_f)
    all_lon_rad = np.radians(all_lon_f)
    
    aer_lat_rad = np.radians(df_aer_f["lat"].values)
    aer_lon_rad = np.radians(df_aer_f["lon"].values)
    
    # Convert to 3D Cartesian coordinates on unit sphere
    def latlon_to_xyz(lat, lon):
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        return np.vstack((x, y, z)).T
    atlid_xyz = latlon_to_xyz(all_lat_rad, all_lon_rad)
    aeronet_xyz = latlon_to_xyz(aer_lat_rad, aer_lon_rad)
    
    # build KDTree for ATLID points
    tree = cKDTree(atlid_xyz)
    
    # query all AERONET points at once for neighbors within 100 km
    R = 6371.0  # Earth radius in km
    max_rad = 100 / R  # convert 100 km to radians on unit sphere
    
    neighbors = tree.query_ball_point(aeronet_xyz, r=max_rad)
    print(len(neighbors))
    
    colocated_atlid_aod = []
    colocated_atlid_std = []
    colocated_atlid_count = []
    colocated_min_dist = []
    
    aer_lats = df_aer_f["lat"].values
    aer_lons = df_aer_f["lon"].values
    
    for i in range(len(df_aer_f)):
        inds = neighbors[i]
    
        if len(inds) == 0:
            colocated_atlid_aod.append(np.nan)
            colocated_atlid_std.append(np.nan)
            colocated_atlid_count.append(0)
            colocated_min_dist.append(np.nan)
            continue

        co_loc_time = all_time_f[inds]
        co_loc_aods = a_aod_f[inds]
        co_loc_lats = all_lat_f[inds]
        co_loc_lons = all_lon_f[inds]
 
        t_aer = df_aer_f["time"].iloc[i].to_datetime64()
        time_diff = np.abs(co_loc_time - t_aer) / np.timedelta64(1, "m")
        time_mask = time_diff <= 30
 
        if np.any(time_mask):
            final_aods = co_loc_aods[time_mask]
            final_lats = co_loc_lats[time_mask]
            final_lons = co_loc_lons[time_mask]
 
            colocated_atlid_aod.append(np.mean(final_aods))
            colocated_atlid_std.append(np.std(final_aods))
            colocated_atlid_count.append(len(final_aods))
            
            dists = haversine(aer_lats[i], aer_lons[i], final_lats, final_lons)
            colocated_min_dist.append(np.min(dists))
        else:
            colocated_atlid_aod.append(np.nan)
            colocated_atlid_std.append(np.nan)
            colocated_atlid_count.append(0)
            colocated_min_dist.append(np.nan)
 
    df_aeronet["atlid_aod_mean"] = np.nan
    df_aeronet["atlid_aod_std"] = np.nan
    df_aeronet["atlid_count"] = 0
    df_aeronet["atlid_min_dist_km"] = np.nan
   
    df_aeronet.loc[mask_aer, "atlid_aod_mean"] = colocated_atlid_aod
    df_aeronet.loc[mask_aer, "atlid_aod_std"] = colocated_atlid_std
    df_aeronet.loc[mask_aer, "atlid_count"] = colocated_atlid_count
    df_aeronet.loc[mask_aer, "atlid_min_dist_km"] = colocated_min_dist
    df_aeronet.to_csv("/scratch/nld6854/earthcare/cams_data/"+month+"_"+year+"/"+year+"_"+month+"_atlid_aeronet_co-located_100km_10atlid_per_aeronet_2nd_method.csv", index=False)
