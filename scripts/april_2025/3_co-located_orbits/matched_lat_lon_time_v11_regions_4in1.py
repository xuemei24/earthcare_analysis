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
# CAMS Data (xxx)
cams_dir = '/net/pc200254/nobackup/users/wangxu/cams_data/aerosol_mmr/august_fire/'
fcams = cams_dir+'aerosol_extinction_coe_355nm_'+month+'_20250801-10.nc'

ds = xr.open_dataset(fcams)
print(ds)
valid_time = ds['forecast_period']+ds['forecast_reference_time']
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

cams_extcoe = ds['aerext355'].values
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
print('time',ds['time'])
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

dslwc = xr.open_dataset(cams_dir+'specific_cloud_liquid_water_content_'+month+'_20250801-10.nc')
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
print(cams_extcoe.shape)

for i in range(cams_extcoe.shape[0]):
    cams_extcoe[i] = np.where(lwc_final>=0.0001,np.nan,cams_extcoe[i])

new_dims = ("model_level","latitude","longitude","timeN")
timeN = ds["time"].rename({"time": "timeN"})

new_var = xr.DataArray(
    cams_extcoe,
    dims=("model_level","latitude", "longitude", "timeN"),
    coords={
        "model_level": ds["model_level"],
        "latitude": ds["latitude"],
        "longitude": ds["longitude"],
        "timeN": timeN
    })

#new_var = xr.DataArray(aod_cams,dims=new_dims,coords={#'forecast_periodN':ds['forecast_period'][::3].expand_dims(dim='forecast_periodN'),
#                          #'forecast_reference_time':ds['forecast_reference_time'],
#                          'latitude':ds['latitude'],'longitude':ds['longitude'],'timeN':ds['time'][::3].values})
print(cams_extcoe.shape)
ds_clear = ds.copy(deep=True)
ds_clear = xr.Dataset({'aerext355':new_var})
print(ds_clear)
print('longitude',ds_clear['longitude'])


srcdir = '/net/pc190625/nobackup_1/users/wangxu/earthcare_data/'+month+'_2025/EBD/'

cmap = ecplt.colormaps.chiljet2
ATC = ecio.load_ATC('/scratch/nld6854/earthcare/earthcare_data/march_2025/TC_/ECA_EXBA_ATL_TC__2A_20250321T122819Z_20250913T131504Z_04614F.h5', prodmod_code="ECA_EXBA")

cmap_tc,bounds,categories_formatted,norm_tc = ATC_category_colors.ecplt_cmap(ATC,'classification_low_resolution')
#category_colors = ecplt.ATC_category_colors
#cmap_tc = ListedColormap(sns.color_palette(category_colors[:len(u)]).as_hex())

fname = "selected_files_smoke_region_"+month+'_2025.txt'
#fname = "selected_files_Southern_Ocean.txt"
#fname = "selected_files_Australia.txt"
print(fname[15:-4])
f = open(fname, "r")
ebd_files = [line.strip() for line in f]
f.close()

#ebd_files = sorted(glob.glob(srcdir+'*h5'))
for i,filen in enumerate(ebd_files[54:]):
    print(i,filen)
    orbit_sequence=filen[-9:-3]
    atlid_extcoe,atlid_lats,atlid_lons,atlid_times,atlid_h,tc_cld,tc_all,err = read_h5.get_ext_col(filen)
    #atlid_extcoe = atlid_extcoe#[::nagg]
    #atlid_lats = atlid_lats#[::nagg]
    #atlid_lons = atlid_lons#[::nagg]
    #atlid_times = atlid_times#[::nagg]
    #atlid_h = atlid_h#[::nagg]

    print("ds_clear['aerext355'].values>0",ds_clear['aerext355'].values[ds_clear['aerext355'].values>0])
    print('terminated here, will also need to add cloud features to the plot')
    cams_interp = ds_clear.interp(latitude=('points',atlid_lats), longitude=('points',atlid_lons), timeN=('points',atlid_times), method='nearest')
    print('cams_interp',cams_interp)
    cams_ext = cams_interp['aerext355'].values[:89,:]
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
        f = interp1d(cams_alt[:,i], cams_ext[:,i], bounds_error=False, fill_value=np.nan)
        cams_interp_varp.append(f(atlid_h[i]))

    cams_interp_varp = np.array(cams_interp_varp)
   
    print(cams_interp_varp)
    print(cams_interp_varp[cams_interp_varp>0],len(cams_interp_varp[cams_interp_varp>0]))
   
    yy,atlid_timesp = np.meshgrid(atlid_h[0],atlid_times)
    print('pcolormesh.shape',atlid_timesp.shape,atlid_h.shape,cams_interp_varp.shape)
    fig, axs = plt.subplots(4, 1, figsize=(25,7*4), gridspec_kw={'hspace':0.67}, sharex=True)
   
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
    axs[0].set_xticks(tick_times)
    axs[0].set_xticklabels(tick_labels, ha='right')

    axs[0].set_title('ATLID Extinction 355 nm & TC clouds '+orbit_sequence)
    axs[0].set_ylabel('Altitude / km')
    axs[0].set_ylim(-0.3,20)
    fig.colorbar(c1, ax=axs[0], label='Extinction')
   
    # CAMS curtain
    c2 = axs[1].pcolormesh(atlid_timesp, atlid_h//1000, cams_interp_varp, cmap=cmap, shading='auto',norm=matplotlib.colors.LogNorm(vmax=vmax,vmin=1e-6))
    axs[1].fill_between(atlid_timesp[:,0],cams_orog//1000,0,color='brown',label='orography')
    axs[1].set_title('CAMS Extinction 355 nm (Interpolated to ATLID track)')
    axs[1].set_ylabel('Altitude / km')
    #axs[1].set_xlabel('Time')
    axs[1].set_ylim(-0.3,20)
    axs[1].set_xticks(tick_times)
    axs[1].set_xticklabels(tick_labels, rotation=45, ha='right')

    axs[1].legend(frameon=False,fontsize='xx-small')
    fig.colorbar(c2, ax=axs[1], label='Extinction')
    
    print('tc_cld 1 & -2',tc_cld[tc_cld==1],tc_cld[tc_cld==-2])
    print(np.nanmin(tc_cld.reshape(-1)),np.nanmax(tc_cld.reshape(-1)))
    c3 = axs[2].pcolormesh(atlid_timesp, atlid_h//1000, tc_all, cmap=cmap_tc,norm=norm_tc)

    #cbar = ecplt.add_colorbar(axs[2],c1_1,'',horz_buffer=0.01)
    #cbar.set_ticks(bounds[:-1]+np.diff(bounds)/2.)
    #cbar.ax.set_yticklabels(categories_formatted, fontsize='xx-small')

    axs[2].set_ylim(-0.3,20)
    axs[2].set_title('Target Classification')
    axs[2].set_ylabel('Altitude / km')
    axs[2].set_xticks(tick_times)
    axs[2].set_xticklabels(tick_labels, ha='right')

    #axs[2].set_xlabel('Time')

    cbar = fig.colorbar(c3,ax=axs[2])
    cbar.set_ticks(bounds[:-1]+np.diff(bounds)/2.)
    cbar.ax.set_yticklabels(categories_formatted, fontsize='xx-small')

    c4 = axs[3].pcolormesh(atlid_timesp, atlid_h//1000, atlid_extcoe/err, cmap=ecplt.colormaps.calipso,vmin=0,vmax=20)
    axs[3].set_ylim(-0.3,20)
    axs[3].set_title('SNR')
    axs[3].set_ylabel('Altitude / km')
    axs[3].set_xticks(tick_times)
    axs[3].set_xticklabels(tick_labels, ha='right')
    fig.colorbar(c4, ax=axs[3], label='SNR')


    plt.tight_layout()
    fig.savefig('slices_regions/'+fname[15:-4]+'_atlid_vs_cams_extinction_'+orbit_sequence+'.jpg')

    plt.close(fig)


    #fig,ax1 = plt.subplots(1)#,figsize=(5,4))
 
    cams_interp_varp3 = cams_interp_varp[(atlid_extcoe >= 0) & (cams_interp_varp >= 0)]
    atlid_extcoe3 = atlid_extcoe[(atlid_extcoe >= 0) & (cams_interp_varp >= 0)]

    #ax1.scatter(atlid_extcoe3,cams_interp_varp3,s=1)
    #ax1.set_xlabel('ATLID extinction coefficient',fontsize=15)
    #ax1.set_ylabel('CAMS extinction coefficient',fontsize=15)
    #fig.savefig('slices_regions/'+fname[15:-4]+'_atlid_vs_cams_extinction_'+orbit_sequence+'_correlation.jpg')

    plt.close(fig)
    np.savetxt('atlid_cams_ext.txt',np.array([atlid_extcoe3,cams_interp_varp3]).transpose(),header='ATLID,CAMS',delimiter=',')

    x_bins = np.logspace(np.log10(1e-6), np.log10(1e-3), 100)
    y_bins = np.logspace(np.log10(1e-6), np.log10(1e-3), 100)

    fig,ax1 = plt.subplots(1,figsize=(10,8))
    c4 = ax1.hist2d(atlid_extcoe3,cams_interp_varp3,bins=[x_bins, y_bins],cmap=cmap)#, norm=LogNorm())
    cb=fig.colorbar(c4[3], ax=ax1, label='')
    cb.ax.tick_params(labelsize=15)
    ax1.set_xlabel('ATLID extinction coefficient',fontsize=15)
    ax1.set_ylabel('CAMS extinction coefficient',fontsize=15)
    ax1.tick_params(labelsize=15)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(5e-6,1e-3)
    ax1.set_ylim(5e-6,1e-3)

    fig.savefig('slices_regions/'+fname[15:-4]+'_atlid_vs_cams_extinction_'+orbit_sequence+'_correlation_hist.jpg')
    plt.close(fig)

    from cartopy.crs import Globe
    my_globe = Globe(semimajor_axis=6378137, semiminor_axis=6356752.314245179,
                     inverse_flattening=298.257223563)

    fig_name = 'slices_regions/'+fname[15:-4]+'_atlid_orbit_'+orbit_sequence+'.jpg'
    fig_title = orbit_sequence
    projections.plot_on_orthographic(atlid_lons,atlid_lats, fig_name, fig_title,central_longitude=-30,central_latitude=45,globe=my_globe)
