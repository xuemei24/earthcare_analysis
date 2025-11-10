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
#from tools.common import colormaps
from matplotlib.colors import LogNorm, Normalize
import matplotlib
#print(dir(ectools))
#ecio=ectools.ecio
#ecplot=ectools.ecplot
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

from ectools import ecio
from ectools import ecplot as ecplt
from ectools import colormaps
from plotting_tools import read_h5,ATC_category_colors,projections

from scipy.spatial import cKDTree
from multiprocessing import Pool
from pathlib import Path


def get_AOD(aod_old, wave_old, wave_new, angstrom):
    print('aod_old>0',aod_old[aod_old>0], 'wave_old',wave_old, 'wave_new',wave_new, 'angstrom>0',angstrom[angstrom>0])
    return ((wave_new / wave_old) ** (-angstrom)) * aod_old

def latlon_to_xyz(lat, lon):
    lat = np.radians(lat)
    lon = np.radians(lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.vstack((x, y, z)).T

month = 'april'
month2 = '04'
time_tol = pd.Timedelta("30min")

#####Constants#####
EARTH_RADIUS = 6371000  # meters

#####AERONET#####
aeronet_path = '/net/pc190625/nobackup_1/users/wangxu/aeronet/'
df = pd.read_table(aeronet_path+'2025'+month2+'_all_sites_aod15_allpoints.txt', delimiter=',', header=[7])
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

def find_nearest_time(t):
    diffs = (df_aeronet["time"] - t).abs()
    idx = diffs.idxmin()
    if diffs.loc[idx] <= time_tol:
        return df_aeronet.loc[idx, "time"]
    else:
        return pd.NaT  # no valid match


######ATLID#####
srcdir = '/net/pc190625/nobackup_1/users/wangxu/earthcare_data/'+month+'_2025/EBD/'
cmap = ecplt.colormaps.chiljet2
ATC = ecio.load_ATC('/scratch/nld6854/earthcare/earthcare_data/march_2025/TC_/ECA_EXBA_ATL_TC__2A_20250321T122819Z_20250913T131504Z_04614F.h5', prodmod_code="ECA_EXBA")

cmap_tc,bounds,categories_formatted,norm_tc = ATC_category_colors.ecplt_cmap(ATC,'classification_low_resolution')

def process_file(filen,df_aeronet):
    tc_file = filen.replace('EBD','TC_')
    if not Path(tc_file).exists():
        return None  # skip missing files

    orbit_sequence = filen[-9:-3]
    atlid_aod, atlid_lats, atlid_lons, atlid_times = read_h5.get_aod(filen, np.array([10,11,12,13,14,15,25,26,27]))
    df_atlid = pd.DataFrame({
        "time": atlid_times,
        "lat": atlid_lats,
        "lon": atlid_lons,
        "atlid_aod": atlid_aod})

    df_aeronet = df_aeronet.dropna(subset=["time"]).sort_values("time")
    df_atlid = df_atlid.dropna(subset=["time"]).sort_values("time")

    df_aeronet = df_aeronet.sort_values("time")
    df_atlid = df_atlid.sort_values("time")
   
    df_atlid = pd.merge_asof(df_atlid,df_aeronet[["time"]].rename(columns={"time":"nearest_time"}),
        left_on="time",right_on="nearest_time",direction="nearest",tolerance=time_tol)

    df_atlid = df_atlid.dropna(subset=["nearest_time"])  # drop if no time match
    df_aeronet_time = df_aeronet[df_aeronet["time"].isin(df_atlid["nearest_time"])]
   
    matched = []
    for t in df_atlid["nearest_time"].unique():
        group_atlid = df_atlid[df_atlid["nearest_time"] == t]
        group_aeronet = df_aeronet_time[df_aeronet_time["time"] == t]
        if group_aeronet.empty:
            continue

        # convert to xyz (vectorized)
        atlid_xyz = latlon_to_xyz(group_atlid["lat"].values, group_atlid["lon"].values)
        aeronet_xyz = latlon_to_xyz(group_aeronet["lat"].values, group_aeronet["lon"].values)

        # one KDTree build per group (fast)
        tree = cKDTree(aeronet_xyz)
        distances_rad, indices = tree.query(atlid_xyz, k=10)
        distances_m = distances_rad * EARTH_RADIUS

        # filter within cutoff (vectorized mask)
        mask = distances_m <= 100000  # e.g. 100 km
        if not np.any(mask):
            continue

        # pre-collect rows
        df_tmp = pd.DataFrame({
            "time": t,
            "atlid_lat": group_atlid["lat"].values[mask],
            "atlid_lon": group_atlid["lon"].values[mask],
            "atlid_aod": group_atlid["atlid_aod"].values[mask],
            "aeronet_lat": group_aeronet["lat"].values[indices[mask]],
            "aeronet_lon": group_aeronet["lon"].values[indices[mask]],
            "aeronet_aod": group_aeronet["aeronet_aod"].values[indices[mask]],
            "distance_m": distances_m[mask]
        })

        matched.append(df_tmp)

    if matched:
        return pd.concat(matched, ignore_index=True)
    return None

# run in parallel
ebd_files = sorted(glob.glob(srcdir+'*h5'))
#for f in ebd_files:
#    process_file(f)
def process_file_wrapper(filen):
    return process_file(filen, df_aeronet)

with Pool(processes=8) as pool:
    results = pool.map(process_file_wrapper, ebd_files)

df_all = pd.concat([r for r in results if r is not None], ignore_index=True)
df_all.to_csv("/net/pc190625/nobackup_1/users/wangxu/cams_data/2025_"+month+"_atlid_aeronet_co-located_100km.csv", index=False)
