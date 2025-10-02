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
from scipy.ndimage import binary_dilation

script_path = '/usr/people/wangxu/Desktop/earthcare_scripts/scripts/april_2025/'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), script_path)))

from ectools import ecio
from ectools import ecplot as ecplt
from ectools import colormaps
from plotting_tools import read_h5,ATC_category_colors


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
#cams_h = cams_h[::-1,:,:]
#print(cams_h.shape)

orography = xr.open_dataset('/scratch/nld6854/earthcare/earthcare_scripts/scripts/cams_orography.nc')
orography = orography.assign_coords(longitude=(('longitude',), cams_lon))
#orog = orography['orography'][:]

dslwc = xr.open_dataset('/net/pc190625/nobackup_1/users/wangxu/cams_data/lwc/specific_cloud_liquid_water_content_march_2025_3.nc')
lwc = dslwc['clwc'].values[0,1:]
dslwc = dslwc.assign_coords(longitude=(('longitude',), cams_lon))

srcdir = '/net/pc190625/nobackup/users/wangxu/earthcare_data/march_2025/EBD/'

cmap = ecplt.colormaps.chiljet2
ATC = ecio.load_ATC('/net/pc190625/nobackup_1/users/wangxu/earthcare_data/march_2025/TC_/ECA_EXAE_ATL_TC__2A_20250321T133730Z_20250321T152847Z_04615D.h5', prodmod_code="ECA_EXAE")

cmap_tc,bounds,categories_formatted,norm_tc = ATC_category_colors.ecplt_cmap(ATC,'classification_low_resolution')
#category_colors = ecplt.ATC_category_colors
#cmap_tc = ListedColormap(sns.color_palette(category_colors[:len(u)]).as_hex())

fname = "selected_files_Greenland.txt"
fname = "selected_files_Africa2.txt"
#fname = "selected_files_Southern_Ocean.txt"
#fname = "selected_files_Australia.txt"
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
    atlid_extcoe,atlid_lats,atlid_lons,atlid_times,atlid_h,tc_cld,tc_all = read_h5.get_ext(filen)

    if not np.any(tc_all == 1):
        continue

    lwc_interp = dslwc.interp(latitude=('points',atlid_lats), longitude=('points',atlid_lons), forecast_reference_time=('points',atlid_times), method='nearest')
    cams_lwc0 = lwc_interp['clwc'].values[0,:,::-1]
    cams_lwc  = cams_lwc0[:,:89]
    print('cams_lwc.shape',cams_lwc.shape)

    if not np.any(cams_lwc >= 0.0001):
        continue

    print(cams_lwc[cams_lwc>=0.0001])
    cams_interp = ds.interp(latitude=('points',atlid_lats), longitude=('points',atlid_lons), forecast_reference_time=('points',atlid_times), method='nearest')
    cams_ext0 = cams_interp['aerext355'].values[0,:,::-1]
    cams_ext  = cams_ext0[:,:89]
    print('cams_ext.shape',cams_ext.shape)

    cams_hinterp = cams_altitude.interp(latitude=('points',atlid_lats), longitude=('points',atlid_lons),method='nearest')
    cams_alt = cams_hinterp['altitude'].values[:89,:]
    print('cams_alt.shape',cams_alt.shape)

    cams_orog_hinterp = orography.interp(latitude=('points',atlid_lats), longitude=('points',atlid_lons),method='nearest')
    cams_orog = cams_orog_hinterp['orography']
    print('cams_orography.shape',cams_orog.shape)

    cams_interp_temp = []
    cams_interp_lwc  = []
    for i in range(len(atlid_times)):
   
        #interpolate CAMS to ATLID vertical axis
        f = interp1d(cams_alt[:,i], cams_ext[i], bounds_error=False, fill_value=np.nan)
        cams_interp_temp.append(f(atlid_h[i]))

        f2 = interp1d(cams_alt[:,i], cams_lwc[i], bounds_error=False, fill_value=np.nan)
        cams_interp_lwc.append(f(atlid_h[i]))

    cams_interp_temp = np.array(cams_interp_temp)
    cams_interp_lwc  = np.array(cams_interp_lwc)
   
    cams_lwcp = np.where(cams_interp_lwc >= 0.0001,-1,0)
    mask_lwc = cams_interp_lwc >= 0.0001
    dilated_lwc = binary_dilation(mask_lwc, structure=np.ones((11,11))) # 5km along track, 500m up- and downward
    surrounding_mask_lwc = dilated_lwc & (~mask_lwc)

    cams_interp_varp = np.full_like(cams_interp_temp,np.nan,dtype=float)
    cams_interp_varp[surrounding_mask_lwc] = cams_interp_temp[surrounding_mask_lwc]
    print('cams_interp_varp.shape',cams_interp_varp.shape)
    print(cams_interp_varp[cams_interp_varp>0],len(cams_interp_varp[cams_interp_varp>0]))

    if len(cams_interp_varp[cams_interp_varp>0]) == 0:
        continue
   
    yy,atlid_timesp = np.meshgrid(atlid_h[0],atlid_times)
    print('pcolormesh.shape',atlid_timesp.shape,atlid_h.shape,cams_interp_varp.shape)

    mask_tc = tc_all == 1 # select liquid clouds
    dilated = binary_dilation(mask_tc, structure=np.ones((11,11))) # 5km along track, 500m up- and downward
    surrounding_mask = dilated & (~mask_tc)
    atlid_extcoep = np.full_like(atlid_extcoe,np.nan,dtype=float)
    atlid_extcoep[surrounding_mask] = atlid_extcoe[surrounding_mask]
    print('atlid_extcoep.shape',atlid_extcoep.shape)
 
    #start plotting
    fig, axs = plt.subplots(3, 1, figsize=(25,7*3), gridspec_kw={'hspace':0.67}, sharex=True)
   
    # LIDAR curtain
    if atlid_extcoep.any()>0:
        vmax = np.nanmax(atlid_extcoep)
    else:
        vmax = 1e-3

    c1 = axs[0].pcolormesh(atlid_timesp, atlid_h//1000, atlid_extcoep, cmap=cmap, shading='auto',norm=matplotlib.colors.LogNorm(vmax=vmax,vmin=1e-6))

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
    #tick_labels = [f"{lat:.1f}°N\n{lon:.1f}°E" for lat, lon in zip(tick_lats, tick_lons)]
    tick_labels = [f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}\n{abs(lon):.1f}°{'E' if lon >= 0 else 'W'}" for lat, lon in zip(tick_lats, tick_lons)]


    # Set the ticks and labels
    axs[0].set_xticks(tick_times)
    axs[0].set_xticklabels(tick_labels, ha='right')
    axs[0].set_title('ATLID Extinction 355 nm & TC clouds '+orbit_sequence)
    axs[0].set_ylabel('Altitude [km]')
    axs[0].set_ylim(-0.3,20)
    fig.colorbar(c1, ax=axs[0], label='Extinction')
   
    # CAMS curtain
    c2_1 = axs[1].pcolormesh(atlid_timesp,atlid_h//1000, cams_lwcp, cmap=cmap_tc,norm=norm_tc)
    c2 = axs[1].pcolormesh(atlid_timesp, atlid_h//1000, cams_interp_varp, cmap=cmap, shading='auto',norm=matplotlib.colors.LogNorm(vmax=vmax,vmin=1e-6))
    axs[1].fill_between(atlid_timesp[:,0],cams_orog//1000,0,color='brown',label='orography')
    axs[1].set_title('CAMS Extinction 355 nm (Interpolated to ATLID track)')
    axs[1].set_ylabel('Altitude [km]')
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
    axs[2].set_ylabel('Altitude [km]')
    axs[2].set_xticks(tick_times)
    axs[2].set_xticklabels(tick_labels, ha='right')
    #axs[2].set_xlabel('Time')
    cbar = fig.colorbar(c3,ax=axs[2])
    cbar.set_ticks(bounds[:-1]+np.diff(bounds)/2.)
    cbar.ax.set_yticklabels(categories_formatted, fontsize='xx-small')

    plt.tight_layout()
    fig.savefig('slices_regions/'+fname[15:-4]+'_atlid_vs_cams_extinction_'+orbit_sequence+'.jpg')
    plt.close(fig)

    cams_interp_varp3 = cams_interp_varp[(atlid_extcoe >= 0) & (cams_interp_varp >= 0)]
    atlid_extcoe3 = atlid_extcoe[(atlid_extcoe >= 0) & (cams_interp_varp >= 0)]

    #fig,ax1 = plt.subplots(1)#,figsize=(5,4))
    #ax1.scatter(atlid_extcoe3,cams_interp_varp3,s=1)
    #ax1.set_xlabel('ATLID extinction coefficient',fontsize=15)
    #ax1.set_ylabel('CAMS extinction coefficient',fontsize=15)

    #fig.savefig('slices_regions/'+fname[15:-4]+'_atlid_vs_cams_extinction_'+orbit_sequence+'_correlation.jpg')

    #plt.close(fig)
    #np.savetxt('atlid_cams_ext.txt',np.array([atlid_extcoe3,cams_interp_varp3]).transpose(),header='ATLID,CAMS',delimiter=',')

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


