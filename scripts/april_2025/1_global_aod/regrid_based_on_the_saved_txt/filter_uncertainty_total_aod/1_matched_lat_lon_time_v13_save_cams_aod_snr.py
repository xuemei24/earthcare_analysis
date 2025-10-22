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
import numpy as np
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
from plotting_tools import read_h5,ATC_category_colors

month = 'september'

# CAMS Data (xxx)
cams_dir = '/scratch/nld6854/earthcare/cams_data/'+month+'_2025/'
fcams = cams_dir+'total_aerosol_optical_depth_355nm_'+month+'_2025.nc'

ds = xr.open_dataset(fcams)
print(ds)
print(ds['forecast_reference_time'])
print(ds['forecast_period'])
valid_time = ds['forecast_period']+ds['forecast_reference_time']
print(valid_time)
print(valid_time.shape)
ds = ds.assign_coords(valid_time=valid_time)
ds = ds.stack(forecast_time=["forecast_reference_time", "forecast_period"])
ds = ds.reset_index("forecast_time")  # removes the MultiIndex
#ds = ds.assign_coords(forecast_time=ds['valid_time'].values)


ds = ds.assign_coords(forecast_time=ds['valid_time'])
ds = ds.swap_dims({"forecast_time": "valid_time"})
if 'time' in ds.coords or 'time' in ds.dims:
    ds = ds.drop_vars('time', errors='ignore')
ds = ds.rename({'valid_time': 'time'})
ds = ds.sortby("time")
print(ds)
print(ds['time'])

cams_aod_r = ds['aod355'].values[:,:,::3]
print('cams_aod_r',cams_aod_r.shape)
cams_lat = ds['latitude'].values
cams_lon = ds['longitude'].values
ilon = np.where(cams_lon>180)
cams_lon[ilon] = cams_lon[ilon]-360.
ds = ds.assign_coords(longitude=(('longitude',), cams_lon))

#this file should be merged
dslwc = xr.open_dataset('/scratch/nld6854/earthcare/cams_data/'+month+'_2025/specific_cloud_liquid_water_content_'+month+'_2025.nc')
lwc = dslwc['clwc'].values[:,:]
def collapse_lwc(cloud,data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            mask = (cloud[i,j]>=0.0001).any(axis=0)
            data[i,j,:,:] = np.where(mask, np.nan, data[i,j,:,:])
    return data

lwc_cams = collapse_lwc(lwc,lwc[:,:,0,:,:])
print('lwc_cams.shape',lwc_cams.shape)

valid_timelwc = dslwc['forecast_period']+dslwc['forecast_reference_time'][:]
print(valid_timelwc.shape)

# Flatten both
valid_time_flat = valid_timelwc.values.ravel()
lwc_flat = lwc_cams.reshape(-1, *lwc_cams.shape[2:])
print('valid_time_flat.shape,lwc_flat.shape',valid_time_flat.shape,lwc_flat.shape)

# Sort
sorted_indices = np.argsort(valid_time_flat)
lwc_sorted = lwc_flat[sorted_indices]
lwc_final = np.transpose(lwc_sorted,(1,2,0))


print(lwc_final.shape)
aod_cams = np.where(lwc_final>=0.0001,np.nan,cams_aod_r)
#aod_cams = np.expand_dims(aod_cams,axis=1)
print(aod_cams.shape)

new_dims = ("latitude","longitude","timeN")
timeN = ds["time"][::3].rename({"time": "timeN"})

new_var = xr.DataArray(
    aod_cams,
    dims=("latitude", "longitude", "timeN"),
    coords={
        "latitude": ds["latitude"],
        "longitude": ds["longitude"],
        "timeN": timeN
    }
)

#new_var = xr.DataArray(aod_cams,dims=new_dims,coords={#'forecast_periodN':ds['forecast_period'][::3].expand_dims(dim='forecast_periodN'),
#                          #'forecast_reference_time':ds['forecast_reference_time'],
#                          'latitude':ds['latitude'],'longitude':ds['longitude'],'timeN':ds['time'][::3].values})
print(aod_cams.shape)
ds_clear = ds.copy(deep=True)
ds_clear = xr.Dataset({'aod355':new_var})
print(ds_clear)
print('longitude',ds_clear['longitude'])


orography = xr.open_dataset('/scratch/nld6854/earthcare/earthcare_scripts/scripts/cams_orography.nc')
orography = orography.assign_coords(longitude=(('longitude',), cams_lon))





srcdir = '/scratch/nld6854/earthcare/earthcare_data/'+month+'_2025/EBD/'

cmap = ecplt.colormaps.chiljet2
ATC = ecio.load_ATC('/scratch/nld6854/earthcare/earthcare_data/march_2025/TC_/ECA_EXAE_ATL_TC__2A_20250321T133730Z_20250321T152847Z_04615D.h5', prodmod_code="ECA_EXAE")

cmap_tc,bounds,categories_formatted,norm_tc = ATC_category_colors.ecplt_cmap(ATC,'classification_low_resolution')
#category_colors = ecplt.ATC_category_colors
#cmap_tc = ListedColormap(sns.color_palette(category_colors[:len(u)]).as_hex())

'''
fname = "selected_files_Greenland.txt"
dirs = "/usr/people/wangxu/Desktop/earthcare_scripts/scripts/april_2025/3_co-located_orbits/"
fname = "selected_files_Africa_"+month+"_2025.txt"
#fname = "selected_files_Southern_Ocean.txt"
#fname = "selected_files_Australia.txt"
print(fname[15:-4])
f = open(dirs+fname, "r")
ebd_files = [line.strip() for line in f]
f.close()
'''
from multiprocessing import Pool
from pathlib import Path
import numpy as np

def process_file(filen):
    tc_file = filen.replace('EBD','TC_')
    if not Path(tc_file).exists():
        return None  # skip missing files

    orbit_sequence = filen[-9:-3]
    atlid_aod, atlid_lats, atlid_lons, atlid_times = read_h5.get_aod_snr(filen, np.array([10,11,12,13,14,15,25,26,27]))

    cams_interp = ds_clear.interp(latitude=('points', atlid_lats),
                                  longitude=('points', atlid_lons),
                                  timeN=('points', atlid_times),
                                  method='nearest')
    cams_aod = cams_interp['aod355'].values

    return (atlid_lats, atlid_lons, atlid_aod, cams_aod)

# run in parallel
ebd_files = sorted(glob.glob(srcdir+'*h5'))
with Pool(processes=8) as pool:  # adjust number of processes
    results = pool.map(process_file, ebd_files)

# combine results
all_lat, all_lon, a_aod, c_aod = [], [], [], []
for res in results:
    if res is None:
        continue
    lat, lon, aod, caod = res
    all_lat.extend(lat)
    all_lon.extend(lon)
    a_aod.extend(aod)
    c_aod.extend(caod)

# save once
np.savetxt(cams_dir+'2025_'+month+'_cams_atlid_co-located_aod_snr_gr_2.txt',
           np.array([all_lat, all_lon, a_aod, c_aod]).T,
           header='latitude,longitude,aod355nm_atlid,aod355nm_cams',
           delimiter=',')
