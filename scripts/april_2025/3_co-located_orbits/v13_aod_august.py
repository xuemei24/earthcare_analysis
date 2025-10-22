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

month = 'august'
#month = 'april'
# CAMS Data (xxx)
cams_dir = '/net/pc200254/nobackup/users/wangxu/cams_data/aerosol_mmr/august_fire/'
fcams = cams_dir+'total_aerosol_optical_depth_355nm_'+month+'_20250801-10.nc'

ds = xr.open_dataset(fcams)
print(ds)
print(ds['forecast_reference_time'])
print(ds['forecast_period'])
valid_time = ds['forecast_period'][:12]+ds['forecast_reference_time']
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
print(ds['aod355'].values.shape)
print('cams_aod_r',cams_aod_r.shape)
cams_lat = ds['latitude'].values
cams_lon = ds['longitude'].values
ilon = np.where(cams_lon>180)
cams_lon[ilon] = cams_lon[ilon]-360.
ds = ds.assign_coords(longitude=(('longitude',), cams_lon))

#this file should be merged
dslwc = xr.open_dataset(cams_dir+'specific_cloud_liquid_water_content_'+month+'_20250801-10.nc')
lwc = dslwc['clwc'].values[:,:]
print('lwc shape',lwc.shape)
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





srcdir = '/net/pc190625/nobackup_1/users/wangxu/earthcare_data/'+month+'_2025/EBD/'

cmap = ecplt.colormaps.chiljet2
ATC = ecio.load_ATC('/net/pc190625/nobackup_1/users/wangxu/earthcare_data/march_2025/TC_/ECA_EXAE_ATL_TC__2A_20250321T133730Z_20250321T152847Z_04615D.h5', prodmod_code="ECA_EXAE")

cmap_tc,bounds,categories_formatted,norm_tc = ATC_category_colors.ecplt_cmap(ATC,'classification_low_resolution')
#category_colors = ecplt.ATC_category_colors
#cmap_tc = ListedColormap(sns.color_palette(category_colors[:len(u)]).as_hex())

fname = "selected_files_Greenland.txt"
dirs = "/usr/people/wangxu/Desktop/earthcare_scripts/scripts/april_2025/3_co-located_orbits/"
fname = "selected_files_smoke_region_"+month+'_2025.txt'
#fname = "selected_files_Southern_Ocean.txt"
#fname = "selected_files_Australia.txt"
print(fname[15:-4])
f = open(dirs+fname, "r")
ebd_files = [line.strip() for line in f]
f.close()

#ebd_files = sorted(glob.glob(srcdir+'*h5'))
for i,filen in enumerate(ebd_files):
    print(i,filen)

    orbit_sequence=filen[-9:-3]
    atlid_aod,atlid_lats,atlid_lons,atlid_times = read_h5.get_aod(filen,np.array([10,11,12,13,14,15,25,26,27]))
    #atlid_extcoe = atlid_extcoe#[::nagg]
    #atlid_lats = atlid_lats#[::nagg]
    #atlid_lons = atlid_lons#[::nagg]
    #atlid_times = atlid_times#[::nagg]
    #atlid_h = atlid_h#[::nagg]

    cams_interp = ds_clear.interp(latitude=('points',atlid_lats), longitude=('points',atlid_lons), timeN=('points',atlid_times), method='nearest')
    print('cams_interp',cams_interp)
    cams_aod = cams_interp['aod355'].values
    print('cams_ext.shape',cams_aod.shape)
    print('cams_ext',cams_aod[cams_aod>0])

    nan_percentage = np.isnan(atlid_aod).sum() / atlid_aod.size * 100
    print(f"Percentage of NaNs in atlid_aod: {nan_percentage:.2f}%")
    nan_percentage = np.isnan(cams_aod).sum() / cams_aod.size * 100
    print(f"Percentage of NaNs in cams_aod: {nan_percentage:.2f}%")

    #AEBD = ecio.load_AEBD(filen,prodmod_code="ECA_EXAC")
    #aod = np.trapz(atlid_extcoe,x=atlid_h,axis=1)

    #start plotting
#    nrows=4
#    suffix = '_low_resolution'
#    hmax=20e3
    
#    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, gridspec_kw={'hspace':0.75})
#    print(AEBD['time'].shape,aod.shape)
#    print(AEBD)
#    #print(aod[aod>0])
#    ecplt.plot_EC_1Dxw(axes[0], AEBD, {'AEBD':{'xdata':AEBD['time'][:],'ydata':aod_cams}},#AEBD['time'], 'ydata':aod[0]}},
#                 "CAMS Aerosol optical depth", "AOD / -", timevar='time', include_ruler=False,line_color='red',label='CAMS')
#    ecplt.plot_EC_1Dxw(axes[0], AEBD, {'AEBD':{'xdata':AEBD['time'][:],'ydata':aod}},#AEBD['time'], 'ydata':aod[0]}},
#                 "ATLID Aerosol optical depth", "AOD / -", timevar='time', include_ruler=False,line_color='black',label='ATLID')
#
#    ecplt.plot_EC_2D(axes[1], AEBD, 'particle_extinction_coefficient_355nm_low_resolution', r"$\alpha_\mathrm{mie}$", cmap=ecplt.colormaps.calipso, plot_scale='log', plot_range=[1e-6,1e-3], units='m$^{-1}$', hmax=hmax, plot_where=AEBD.particle_extinction_coefficient_355nm > 1e-6)
# 
#    ecplt.plot_EC_2Dxw(axes[2], AEBD, 'particle_extinction_coefficient_355nm_low_resolution','particle_extinction_coefficient_355nm_low_resolution_error', r"-", cmap=ecplt.colormaps.calipso, plot_scale='linear', title='Extinction coefficient SNR', plot_range=[0,30],units='-', hmax=hmax, plot_where=AEBD.particle_extinction_coefficient_355nm > 1e-6)
# 
#    ecplt.plot_EC_target_classification(axes[3], ATC, 'classification_low_resolution', ecplt.ATC_category_colors, hmax=hmax)
# 
#    ecplt.add_subfigure_labels(axes)
#    
#    fig.savefig('slices_regions/'+fname[15:-4]+'_atlid_cams_aod_'+orbit_sequence+'.jpg')
# 

    fig, axs = plt.subplots(1, 1, figsize=(25,7*1), gridspec_kw={'hspace':0.67}, sharex=True)
    axs.plot(atlid_times,atlid_aod,'k-',label='ATLID')
    axs.plot(atlid_times,cams_aod,'r-',label='CAMS') 
    num_ticks = 6
    tick_indices = np.linspace(0, len(atlid_times)-1, num_ticks, dtype=int)

    # Get the corresponding times and positions
    tick_times = atlid_times[tick_indices]
    tick_lons = atlid_lons[tick_indices]  # Your longitude data
    tick_lats = atlid_lats[tick_indices]  # Your latitude data

    # Create formatted labels (adjust format as needed)
    tick_labels = [f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}\n{abs(lon):.1f}°{'E' if lon >= 0 else 'W'}" for lat, lon in zip(tick_lats, tick_lons)]

    #tick_labels = [f"{lat:.1f}°N\n{lon:.1f}°E" for lat, lon in zip(tick_lats, tick_lons)]

    # Set the ticks and labels
    axs.set_xticks(tick_times)
    axs.set_xticklabels(tick_labels, ha='right',fontsize='xx-small')

    axs.set_title('AOD '+orbit_sequence,fontsize='xx-small')
    axs.set_ylabel('AOD',fontsize='xx-small')
    axs.tick_params(labelsize=15)
    axs.legend(frameon=False, fontsize='xx-small')
    fig.savefig('slices_regions/'+fname[15:-4]+'_atlid_vs_cams_AOD_'+orbit_sequence+'.jpg')

    from cartopy.crs import Globe
    my_globe = Globe(semimajor_axis=6378137, semiminor_axis=6356752.314245179,
                     inverse_flattening=298.257223563)

    fig_name = 'slices_regions/'+fname[15:-4]+'_atlid_orbit_'+orbit_sequence+'.jpg'
    fig_title = orbit_sequence
    projections.plot_on_orthographic(atlid_lons,atlid_lats, fig_name, fig_title,globe=my_globe)

