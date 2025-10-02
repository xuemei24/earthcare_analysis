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

script_path = '/usr/people/wangxu/Desktop/earthcare_scripts/scripts/april_2025/'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), script_path)))

from ectools import ecio
from ectools import ecplot as ecplt
from ectools import colormaps
from plotting_tools import read_h5,ATC_category_colors

# CAMS Data (xxx)
forecast_period = '03'


srcdir = '/net/pc190625/nobackup_1/users/wangxu/earthcare_data/april_2025/EBD/'

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
    orbit_sequence=filen[-9:-3]
    atlid_extcoe,atlid_lats,atlid_lons,atlid_times,atlid_h,tc_cld,tc_all,err = read_h5.get_ext_col(filen)
    print(atlid_times[0],atlid_times[1000])
    #atlid_extcoe = atlid_extcoe#[::nagg]
    #atlid_lats = atlid_lats#[::nagg]
    #atlid_lons = atlid_lons#[::nagg]
    #atlid_times = atlid_times#[::nagg]
    #atlid_h = atlid_h#[::nagg]

    yy,atlid_timesp = np.meshgrid(atlid_h[0],atlid_times)
    fig, axs = plt.subplots(3, 1, figsize=(25,7*3), gridspec_kw={'hspace':0.67}, sharex=True)
   
    # LIDAR curtain
    if atlid_extcoe.any()>0:
        vmax = np.nanmax(atlid_extcoe)
    else:
        vmax = 1e-3
    num_ticks = 6  
    #tick_indices = np.linspace(0, len(atlid_times)-1, num_ticks, dtype=int)
    startT = 0
    endT = 1000
    tick_indices = np.linspace(startT,endT-1,num_ticks, dtype=int)

    # Get the corresponding times and positions
    tick_times = atlid_times[tick_indices]
    tick_lons = atlid_lons[tick_indices]  # Your longitude data
    tick_lats = atlid_lats[tick_indices]  # Your latitude data
    
    # Create formatted labels (adjust format as needed)
    tick_labels = [f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}\n{abs(lon):.1f}°{'E' if lon >= 0 else 'W'}" for lat, lon in zip(tick_lats, tick_lons)]

    print('tc_cld 1 & -2',tc_cld[tc_cld==1],tc_cld[tc_cld==-2])
    print(np.nanmin(tc_cld.reshape(-1)),np.nanmax(tc_cld.reshape(-1)))
    c3 = axs[2].pcolormesh(atlid_timesp[startT:endT,:], atlid_h[startT:endT,:]//1000, tc_all[startT:endT,:], cmap=cmap_tc,norm=norm_tc)

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

    plt.tight_layout()
    fig.savefig(fname[15:-4]+'_atlid_vs_cams_extinction_'+orbit_sequence+'.jpg')

    plt.close(fig)



