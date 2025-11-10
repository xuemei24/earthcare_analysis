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
import ectools
#print(dir(ectools))
#ecio=ectools.ecio
#ecplot=ectools.ecplot
from ectools import ecio
from ectools import ecplot as ecplt
from ectools import colormaps
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
from select_aerosol_extinction import read_hdf5,read_wind
from ATC_category_colors import ecplt_cmap
from scipy.interpolate import interp1d
from matplotlib.colors import ListedColormap, BoundaryNorm

#reload(ecio)
#reload(ecplt)
#reload(colormaps)

# CAMS Data (xxx)
fcams = '/net/pc190625/nobackup_1/users/wangxu/cams_data/extinction_355nm_multi/aerosol_extinction_coe_355nm_march2025_3.nc'
if fcams[-4:-3] == '0':
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
ds['forecast_reference_time'] = ds['forecast_reference_time'] + ds['forecast_period']
forecast_reference_time = ds['forecast_reference_time'].values
print('forecast_reference_time',forecast_reference_time)

#cams_h = loadtxt('/usr/people/wangxu/Desktop/earthcare_scripts/grib_height/geometric_height.csv',skiprows=2)[48:] #keep only the lowest 20 km
#cams_h = cams_h[::-1]

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

dslwc = xr.open_dataset('/net/pc190625/nobackup_1/users/wangxu/cams_data/lwc/specific_cloud_liquid_water_content_march_2025_3.nc')
lwc = dslwc['clwc'].values[0,1:]

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
cams_extcoe_clear = np.expand_dims(cams_extcoe_clear,axis=0)
print(cams_extcoe_clear.shape)

ds_clear = ds.copy(deep=True)
ds_clear['aerext355'].values = cams_extcoe_clear[:,:,::-1,:,:]
 
srcdir = '/net/pc190625/nobackup/users/wangxu/earthcare_data/march_2025/EBD/'

cmap = ecplt.colormaps.chiljet2
ATC = ecio.load_ATC('/scratch/nld6854/earthcare/earthcare_data/march_2025/TC_/ECA_EXBA_ATL_TC__2A_20250321T122819Z_20250913T131504Z_04614F.h5', prodmod_code="ECA_EXBA")

cmap_tc,bounds,categories_formatted,norm_tc = ecplt_cmap(ATC,'classification_low_resolution')
#category_colors = ecplt.ATC_category_colors
#cmap_tc = ListedColormap(sns.color_palette(category_colors[:len(u)]).as_hex())

fname = "selected_files_Greenland.txt"
fname = "selected_files_Africa.txt"
fname = "selected_files_Southern_Ocean.txt"
fname = "selected_files_Australia.txt"

print(fname[15:-4])
f = open(fname, "r")
ebd_files = [line.strip() for line in f]
f.close()

#ebd_files = sorted(glob.glob(srcdir+'*h5'))
for i,filen in enumerate(ebd_files):
    print(i,filen)
    print(utcs)
    if filen[-34:-32] not in utcs:
        continue

    orbit_sequence=filen[-9:-3]
    atlid_extcoe,atlid_lats,atlid_lons,atlid_times,atlid_h,tc_cld,tc_all = read_hdf5(filen)
    u_along,w_wind = read_wind(filen)

    #atlid_extcoe = atlid_extcoe#[::nagg]
    #atlid_lats = atlid_lats#[::nagg]
    #atlid_lons = atlid_lons#[::nagg]
    #atlid_times = atlid_times#[::nagg]
    #atlid_h = atlid_h#[::nagg]

    print("ds_clear['aerext355'].values>0",ds_clear['aerext355'].values[ds_clear['aerext355'].values>0])
    print('terminated here, will also need to add cloud features to the plot')
    cams_interp = ds_clear.interp(latitude=('points',atlid_lats), longitude=('points',atlid_lons), forecast_reference_time=('points',atlid_times), method='nearest')
    print('cams_interp',cams_interp)
    cams_ext = cams_interp['aerext355'].values[0,:,:89]
    print('cams_ext.shape',cams_ext.shape)
    print('cams_ext',cams_ext[cams_ext>0])

    cams_hinterp = cams_altitude.interp(latitude=('points',atlid_lats), longitude=('points',atlid_lons),method='nearest')
    cams_alt = cams_hinterp['altitude'].values[:89,:]
    print('cams_alt',cams_alt)
    print('cams_alt.shape',cams_alt.shape)
    print('atlid_h',atlid_h)

    cams_orog_hinterp = orography.interp(latitude=('points',atlid_lats), longitude=('points',atlid_lons),method='nearest')
    cams_orog = cams_orog_hinterp['orography']
    print('cams_orography.shape',cams_orog.shape)

    cams_interp_varp = []
    for i in range(len(atlid_times)):
   
        #interpolate CAMS to ATLID vertical axis
        f = interp1d(cams_alt[:,i], cams_ext[i], bounds_error=False, fill_value=np.nan)
        cams_interp_varp.append(f(atlid_h[i]))

    cams_interp_varp = np.array(cams_interp_varp)
   
    print(cams_interp_varp)
    print(cams_interp_varp[cams_interp_varp>0],len(cams_interp_varp[cams_interp_varp>0]))
   
    yy,atlid_timesp = np.meshgrid(atlid_h[0],atlid_times)
    print('pcolormesh.shape',atlid_timesp.shape,atlid_h.shape,cams_interp_varp.shape)
    fig, axs = plt.subplots(3, 1, figsize=(25,7*3), gridspec_kw={'hspace':0.67}, sharex=True)
   
    # LIDAR curtain
    if atlid_extcoe.any()>0:
        vmax = np.nanmax(atlid_extcoe)
    else:
        vmax = 1e-3
    c1 = axs[0].pcolormesh(atlid_timesp, atlid_h//1000, atlid_extcoe, cmap=cmap, shading='auto',norm=matplotlib.colors.LogNorm(vmax=vmax,vmin=1e-6))

    tc_cld2 = tc_cld.copy()
    cld_index = np.array([1, 2, 3, 20, 21, 22])
    tc_cld2[np.isin(tc_cld2, cld_index)] = -1

    c1_1 = axs[0].pcolormesh(atlid_timesp, atlid_h//1000, tc_cld2, cmap=cmap_tc,norm=norm_tc)
    scale=200
    magnitude = np.sqrt(u_along[::scale]**2 + w_wind[::scale]**2)
    axs[0].quiver(atlid_timesp[::scale], atlid_h[::scale]//1000,u_along[::scale],w_wind[::scale],magnitude,scale=200,width=0.002, cmap='coolwarm')

    num_ticks = 6  
    tick_indices = np.linspace(0, len(atlid_times)-1, num_ticks, dtype=int)
   
    # Get the corresponding times and positions
    tick_times = atlid_times[tick_indices]
    tick_lons = atlid_lons[tick_indices]  # Your longitude data
    tick_lats = atlid_lats[tick_indices]  # Your latitude data
   
    # Create formatted labels (adjust format as needed)
    #tick_labels = [f"{lat:.1f}°N\n{lon:.1f}°E" for lat, lon in zip(tick_lats, tick_lons)]
    tick_labels = [f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}\n{abs(lon):.1f}°{'E' if lon >= 0 else 'W'}" for lat, lon in zip(tick_lats, tick_lons)]


    
    # Set the ticks and labels
    axs[0].set_xticks(tick_times)
    axs[0].set_xticklabels(tick_labels, rotation=45, ha='right')

    axs[0].set_title('ATLID Extinction 355 nm & TC clouds '+orbit_sequence)
    axs[0].set_ylabel('Altitude [km]')
    axs[0].set_ylim(-0.3,20)
    fig.colorbar(c1, ax=axs[0], label='Extinction')
   
    # CAMS curtain
    c2 = axs[1].pcolormesh(atlid_timesp, atlid_h//1000, cams_interp_varp, cmap=cmap, shading='auto',norm=matplotlib.colors.LogNorm(vmax=vmax,vmin=1e-6))
    axs[1].fill_between(atlid_timesp[:,0],cams_orog//1000,0,color='brown',label='orography')
    axs[1].set_title('CAMS Extinction 355 nm (Interpolated to ATLID track)')
    axs[1].set_ylabel('Altitude [km]')
    axs[1].set_xticks(tick_times)
    axs[1].set_xticklabels(tick_labels, rotation=45, ha='right')
    #axs[1].set_xlabel('Time')
    axs[1].set_ylim(-0.3,20)
    axs[1].legend(frameon=False,fontsize='xx-small')
    fig.colorbar(c2, ax=axs[1], label='Extinction')

    print('tc_cld 1 & -2',tc_cld[tc_cld==1],tc_cld[tc_cld==-2])
    print(np.nanmin(tc_cld.reshape(-1)),np.nanmax(tc_cld.reshape(-1)))
    c3 = axs[2].pcolormesh(atlid_timesp, atlid_h//1000, tc_all, cmap=cmap_tc,norm=norm_tc)
  
    axs[2].set_ylim(-0.3,20)
    axs[2].set_title('Target Classification')
    axs[2].set_ylabel('Altitude [km]')
    axs[2].set_xticks(tick_times)
    axs[2].set_xticklabels(tick_labels, ha='right')

    cbar = fig.colorbar(c3,ax=axs[2])
    cbar.set_ticks(bounds[:-1]+np.diff(bounds)/2.)
    cbar.ax.set_yticklabels(categories_formatted, fontsize='xx-small')

    plt.tight_layout()
    fig.savefig('slices/'+fname[15:-4]+'_atlid_vs_cams_extinction_'+orbit_sequence+'.jpg')

    plt.close(fig)
