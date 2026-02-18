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
from multiprocessing import Pool



script_path = '/home/nld6854/earthcare_scripts/scripts/april_2025'
sys.path.append(script_path)

from ectools.ectools_edited import ecio
from ectools.ectools_edited import ecplot as ecplt
from ectools.ectools_edited import colormaps
from plotting_tools import read_h5,ATC_category_colors,projections

month = 'march'
forecast_period = '3'

year = '2024' if month == 'december' else '2025'
# CAMS Data (xxx)
cams_dir = '/scratch/nld6854/earthcare/cams_data/'+month+'_'+year+'/'

srcdir = '/scratch/nld6854/earthcare/earthcare_data/'+month+'_'+year+'/CAMS/'

cmap = ecplt.colormaps.chiljet2
ATC = ecio.load_ATC('/scratch/nld6854/earthcare/earthcare_data/march_2025/TC_/ECA_EXBA_ATL_TC__2A_20250321T122819Z_20250913T131504Z_04614F.h5')

cmap_tc,bounds,categories_formatted,norm_tc = ATC_category_colors.ecplt_cmap(ATC,'classification_low_resolution')

cams_files = sorted(glob.glob(srcdir+'*nc'))[:10]
def atlid_cams_ext(cams_file):
    filen = cams_file.replace('CAMS','EBD')
    filen = filen.replace('nc','h5')
    print(filen)
    orbit_sequence=filen[-9:-3]
    atlid_extcoe,atlid_lats,atlid_lons,atlid_times,atlid_h,tc_cld,tc_all,err = read_h5.get_ext(filen,np.array([10,11,12,13,14,15,25,26,27]))

    fcams = xr.open_dataset(cams_file)
    cams_interp_varp,cams_orog,level_number,time = fcams['cams_interp_varp'],fcams['cams_orography'],fcams['level_number'],fcams['time']

    org2 = cams_orog.values[:, None]     # shape becomes (4825, 1)
    org2 = np.repeat(org2, cams_interp_varp.shape[1], axis=1)   # now (4825, 241)

    cams_interp_varp = np.where(atlid_h>org2,cams_interp_varp,0)

    print(cams_interp_varp.shape,atlid_extcoe.shape)

    yy,atlid_timesp = np.meshgrid(atlid_h[0],atlid_times)
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
    fig.savefig('slices_regions/atlid_vs_cams_extinction_'+orbit_sequence+'.jpg')

    plt.close(fig)

    cams = cams_interp_varp      # (4946, 241)
    mask = (atlid_extcoe >= 0) & (cams >= 0)
    cams_interp_varp3 = cams[mask]
    atlid_extcoe3     = atlid_extcoe[mask]

    plt.close(fig)
    np.savetxt('atlid_cams_ext.txt',np.array([atlid_extcoe3,cams_interp_varp3]).transpose(),header='ATLID,CAMS',delimiter=',')

    x_bins = np.logspace(np.log10(1e-6), np.log10(1e-3), 100)
    y_bins = np.logspace(np.log10(1e-6), np.log10(1e-3), 100)

    fig,ax1 = plt.subplots(1,figsize=(10,8))
    c4 = ax1.hist2d(atlid_extcoe3,cams_interp_varp3,bins=[x_bins, y_bins],cmap=cmap)#, norm=LogNorm())

    x,y = atlid_extcoe3,cams_interp_varp3
    mask = (np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0))
    x_fit,y_fit = x[mask],y[mask]
    a, b = np.polyfit(x_fit, y_fit, 1)
    y_pred = a * x_fit + b
    rmse = np.sqrt(np.mean((y_fit - y_pred)**2))
    x_line = np.logspace(np.log10(5e-6),np.log10(1e-3),200)
    y_line = a * x_line + b

    ax1.plot(x_line, y_line,color="red",linewidth=2,label=(f"$y = {a:.2f}x + {b:.2e}$\n"f"RMSE = {rmse:.2e}"))

    cb=fig.colorbar(c4[3], ax=ax1, label='')
    cb.ax.tick_params(labelsize=15)
    ax1.set_xlabel('ATLID extinction coefficient',fontsize=15)
    ax1.set_ylabel('CAMS extinction coefficient',fontsize=15)
    ax1.tick_params(labelsize=15)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(5e-6,1e-3)
    ax1.set_ylim(5e-6,1e-3)

    ax1.set_axisbelow(False)  # grid on top of artists

    ax1.grid(True,which="both",color='gray',linestyle="--",linewidth=0.7,alpha=0.7)

    ax1.legend(frameon=False)
    fig.savefig('slices_regions/atlid_vs_cams_extinction_'+orbit_sequence+'_correlation_hist.jpg')
    plt.close(fig)

    from cartopy.crs import Globe
    my_globe = Globe(semimajor_axis=6378137, semiminor_axis=6356752.314245179,
                     inverse_flattening=298.257223563)

    fig_name = 'slices_regions/atlid_orbit_'+orbit_sequence+'.jpg'
    fig_title = orbit_sequence
    print(atlid_lons,atlid_lats)
    projections.plot_on_orthographic(atlid_lons,atlid_lats, fig_name, fig_title,central_longitude=np.nanmean(atlid_lons),central_latitude=atlid_lats[np.nanmean(atlid_lats)],globe=my_globe)


with Pool(processes=8) as pool:  # adjust number of processes
    pool.map(atlid_cams_ext, cams_files)

