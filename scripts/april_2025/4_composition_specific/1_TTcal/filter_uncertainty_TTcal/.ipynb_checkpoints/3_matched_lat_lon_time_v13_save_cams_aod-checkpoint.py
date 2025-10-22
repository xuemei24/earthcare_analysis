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
from plotting_tools import read_h5,ATC_category_colors,projections

month = 'may'
which_aerosol = 'dust'
which_aerosol = 'ssa'
fname = 'sea_salt' if which_aerosol == 'ssa' else 'dust'
tcs   = [11] if which_aerosol == 'ssa' else [10,13,14,15,27] #12=continental pollution?

cams_dir = '/scratch/nld6854/earthcare/cams_data/'+month+'_2025/TTcal/'
fcams = cams_dir+'TTcal_aod355nm_per_composition_'+month+"_2025.nc"

ds = xr.open_dataset(fcams)
print(ds)
print(ds['forecast_period'])
ds['forecast_reference_time'] = (('forecast_reference_time', 
     ds['forecast_reference_time'].values.astype('datetime64[ns]')))
print(ds['forecast_reference_time'])
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
print(ds['time'].values)

if which_aerosol == 'ssa':
    cams_aod = ds['ssdryaod355'].values
elif which_aerosol == 'dust':
    cams_aod = ds['dudryaod355'].values+ds['bcdryaod355'].values+ds['omdryaod355'].values
print('cams_aod_',which_aerosol,cams_aod.shape)
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
aod_cams  = np.where(lwc_final>=0.0001,np.nan,cams_aod)
#aod_cams = np.expand_dims(aod_cams,axis=1)
print(aod_cams.shape)

new_dims = ("latitude","longitude","timeN")
timeN = ds["time"].rename({"time": "timeN"})

new_var = xr.DataArray(
    aod_cams,
    dims=("latitude", "longitude", "timeN"),
    coords={
        "latitude": ds["latitude"],
        "longitude": ds["longitude"],
        "timeN": timeN})

#new_var = xr.DataArray(aod_cams,dims=new_dims,coords={#'forecast_periodN':ds['forecast_period'][::3].expand_dims(dim='forecast_periodN'),
#                          #'forecast_reference_time':ds['forecast_reference_time'],
#                          'latitude':ds['latitude'],'longitude':ds['longitude'],'timeN':ds['time'][::3].values})
print(aod_cams.shape)
ds_clear = ds.copy(deep=True)
ds_clear = xr.Dataset({'aod355_'+which_aerosol:new_var})

print(ds_clear)
print('longitude',ds_clear['longitude'])

print(ds_clear.latitude.to_index().is_unique)  # True or False
print(ds_clear.longitude.to_index().is_unique)
print(ds_clear.timeN.to_index().is_unique)

orography = xr.open_dataset('/scratch/nld6854/earthcare/earthcare_scripts/scripts/cams_orography.nc')
orography = orography.assign_coords(longitude=(('longitude',), cams_lon))





srcdir = '/scratch/nld6854/earthcare/earthcare_data/'+month+'_2025/EBD/'

cmap = ecplt.colormaps.chiljet2
ATC = ecio.load_ATC('/scratch/nld6854/earthcare/earthcare_data/march_2025/TC_/ECA_EXAE_ATL_TC__2A_20250321T133730Z_20250321T152847Z_04615D.h5', prodmod_code="ECA_EXAE")

cmap_tc,bounds,categories_formatted,norm_tc = ATC_category_colors.ecplt_cmap(ATC,'classification_low_resolution')
#category_colors = ecplt.ATC_category_colors
#cmap_tc = ListedColormap(sns.color_palette(category_colors[:len(u)]).as_hex())

from multiprocessing import Pool
from pathlib import Path
import numpy as np

def process_file(filen):
    tc_file = filen.replace('EBD','TC_')
    if not Path(tc_file).exists():
        return None  # skip missing files

    orbit_sequence = filen[-9:-3]
    
    atlid_aod, atlid_lats, atlid_lons, atlid_times = read_h5.get_aod_snr(filen, np.array(tcs))

    cams_interp = ds_clear.interp(latitude=('points', atlid_lats),
                                  longitude=('points', atlid_lons),
                                  timeN=('points', atlid_times),
                                  method='nearest')
    cams_aod = cams_interp['aod355_'+which_aerosol].values

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
    lat, lon, a_s_aod, c_s_aod = res
    all_lat.extend(lat)
    all_lon.extend(lon)
    a_aod = np.append(a_aod,a_s_aod)
    c_aod = np.append(c_aod,c_s_aod)

# save once
np.savetxt(cams_dir+'TTcal_'+fname+'_aod355nm_per_composition_'+month+'_2025_cams_atlid_co-located_snr_gr_2.txt',
           np.array([all_lat, all_lon, a_aod, c_aod]).T,
           header='latitude,longitude,aod355nm_atlid,aod355nm_cams',
           delimiter=',')

