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
from pathlib import Path

from multiprocessing import Pool

month = 'august'
forecast_period = '3'
# CAMS Data (xxx)
fcams = '/scratch/nld6854/earthcare/cams_data/'+month+'_2025/aerosol_extinction_coe_355nm_'+month+'_2025_'+str(int(forecast_period))+'.nc'

if fcams[-4:-3] == '12':#'0':
    utcs = ['00','01','11','12','13','23']
elif fcams[-4:-3] == '3':
    utcs = ['02','03','04','14','15','16']
elif fcams[-4:-3] == '6':
    utcs = ['05','06','07','17','18','19']
elif fcams[-4:-3] == '9':
    utcs = ['08','09','10','20','21','22']

ds = xr.open_dataset(fcams)

cams_extcoe = ds['aerext355'].values[0]
cams_lat = ds['latitude'].values
cams_lon = ds['longitude'].values
ilon = np.where(cams_lon>180)
cams_lon[ilon] = cams_lon[ilon]-360.
ds = ds.assign_coords(longitude=(('longitude',), cams_lon))
# Convert the array to a list of strings (for human-readable format)
#cams_time = ds['forecast_reference_time'].values
#cams_time_list = [str(time) for time in cams_time]
#cams_time = cams_time_list
#print(cams_time_list)
ds['forecast_reference_time'] = ds['forecast_period'] + ds['forecast_reference_time']
print(ds['forecast_reference_time'])
ds['forecast_reference_time'] = ds['forecast_reference_time'][0]
 
print('forecast_reference_time',ds['forecast_reference_time'])

cams_altitude = xr.open_dataset('/scratch/nld6854/earthcare/earthcare_scripts/scripts/cams_altitude.nc')
cams_altitude = cams_altitude.assign_coords(longitude=(('longitude',), cams_lon))
cams_h = cams_altitude['altitude'].values[::-1,:,:]
cams_altitude['altitude'].values = cams_h
print('cams_h[:,0,0]',cams_h[:,0,0])
#cams_h = cams_h[::-1,:,:]
#print(cams_h.shape)

orography = xr.open_dataset('/scratch/nld6854/earthcare/earthcare_scripts/scripts/cams_orography.nc')
orography = orography.assign_coords(longitude=(('longitude',), cams_lon))
#orog = orography['orography'][:]

dslwc = xr.open_dataset('/scratch/nld6854/earthcare/cams_data/'+month+'_2025/specific_cloud_liquid_water_content_'+month+'_2025_'+str(int(forecast_period))+'.nc')
lwc = dslwc['clwc'].values[0,:]

'''
#def where_cloudy(cloud,data):
#    print('before',data.shape)
#    for i in range(cloud.shape[0]):
#        for j in range(cloud.shape[2]):
#            mask = (cloud[i,:,j,:]>=0.0001).any(axis=0)
#            data[i,:,j,mask] = np.nan
#    print('after',data.shape)
#    return data
def where_cloudy(cloud,data):
    print('before',data.shape)
    mask = (cloud>=0.0001).any(axis=1)
    mask_expanded = np.expand_dims(mask, axis=1) #Expand mask to match 'data' shape for height (axis=1)
    mask_final = np.repeat(mask_expanded, repeats=137, axis=1)
    print('mask_expanded.shape',mask_expanded.shape)
    data[:, :, :, :] = np.where(mask_final, np.nan, data)
    print('after',data.shape)
    return data

cams_extcoe_clear = where_cloudy(lwc,cams_extcoe)
'''
#the lines below keeps more aerosol grids than above
cams_extcoe_clear = np.where(lwc<0.0001,cams_extcoe,np.nan)
cams_extcoe_clear = np.expand_dims(cams_extcoe_clear,axis=0)
print(cams_extcoe_clear.shape)

ds_clear = ds.copy(deep=True)
ds_clear['aerext355'].values = cams_extcoe_clear[:,:,::-1,:,:]
 
srcdir = '/scratch/nld6854/earthcare/earthcare_data/'+month+'_2025/EBD/'

cmap = ecplt.colormaps.chiljet2
ATC = ecio.load_ATC('/scratch/nld6854/earthcare/earthcare_data/march_2025/TC_/ECA_EXAE_ATL_TC__2A_20250321T133730Z_20250321T152847Z_04615D.h5', prodmod_code="ECA_EXAE")

cmap_tc,bounds,categories_formatted,norm_tc = ATC_category_colors.ecplt_cmap(ATC,'classification_low_resolution')
#category_colors = ecplt.ATC_category_colors
#cmap_tc = ListedColormap(sns.color_palette(category_colors[:len(u)]).as_hex())

ebd_files = sorted(glob.glob(srcdir+'*h5'))
def process_file(filen):
    outpath = filen.replace("EBD","CAMS").replace("h5","nc")
    if os.path.exists(outpath):
        print(f"Skipping {filen}, already processed.")
        return

    try:
        if filen[-34:-32] not in utcs:
            return None

        tc_file = filen.replace('EBD','TC_')
        if not Path(tc_file).exists():
            return None

        orbit_sequence = filen[-9:-3]
        atlid_extcoe, atlid_lats, atlid_lons, atlid_times, atlid_h, tc_cld, tc_all, err = read_h5.get_ext_col(filen)

        # Interpolate CAMS data to ATLID positions
        cams_interp = ds_clear.interp(
            latitude=('points', atlid_lats),
            longitude=('points', atlid_lons),
            forecast_reference_time=('points', atlid_times),
            method='nearest'
        )

        cams_ext = cams_interp['aerext355'].values[0,:,:89]
        if len(cams_ext[cams_ext>0]) == 0:
            return None

        cams_hinterp = cams_altitude.interp(latitude=('points', atlid_lats),
                                            longitude=('points', atlid_lons), method='nearest')
        cams_alt = cams_hinterp['altitude'].values[:89,:]

        cams_orog_hinterp = orography.interp(latitude=('points', atlid_lats),
                                             longitude=('points', atlid_lons), method='nearest')
        cams_orog = cams_orog_hinterp['orography']

        cams_interp_varp = []
        for i in range(len(atlid_times)):
            f = interp1d(cams_alt[:,i], cams_ext[i], bounds_error=False, fill_value=np.nan)
            cams_interp_varp.append(f(atlid_h[i]))
        cams_interp_varp = np.array(cams_interp_varp)

        ds = xr.Dataset(
            {
                "cams_interp_varp": (("time", "level_number"), cams_interp_varp),
                "cams_orography": (("time"), cams_orog.values)
            },
            coords={
                "time": atlid_times,
                "level_number": np.arange(len(atlid_h[0]))
            }
        )

        fcams = filen.replace("EBD","CAMS").replace("h5","nc")
        ds.to_netcdf(fcams, format="NETCDF4")
    except Exception as e:
        print(f"❌ Error processing {filen}: {e}")


if __name__ == "__main__":
    ebd_files = ebd_files
    
    # Number of parallel workers (adjust based on your CPU cores)
    n_workers = 8
    
    with Pool(n_workers) as pool:
        pool.map(process_file, ebd_files)
