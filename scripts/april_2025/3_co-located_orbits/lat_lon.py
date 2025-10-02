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
from plotting_tools import read_h5,ATC_category_colors,projections

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# CAMS Data (xxx)
forecast_period = '03'
fcams = '/net/pc190625/nobackup_1/users/wangxu/cams_data/extinction_355nm_multi/aerosol_extinction_coe_355nm_april2025_'+str(int(forecast_period))+'.nc'

if fcams[-4:-3] == '0':
    utcs = ['00','01','11','12','13','23']
elif fcams[-4:-3] == '3':
    utcs = ['02','03','04','14','15','16']
elif fcams[-4:-3] == '6':
    utcs = ['05','06','07','17','18','19']
elif fcams[-4:-3] == '9':
    utcs = ['08','09','10','20','21','22']


srcdir = '/net/pc190625/nobackup_1/users/wangxu/earthcare_data/april_2025/EBD/'

cmap = ecplt.colormaps.chiljet2
ATC = ecio.load_ATC('/net/pc190625/nobackup_1/users/wangxu/earthcare_data/march_2025/TC_/ECA_EXAE_ATL_TC__2A_20250321T133730Z_20250321T152847Z_04615D.h5', prodmod_code="ECA_EXAE")

cmap_tc,bounds,categories_formatted,norm_tc = ATC_category_colors.ecplt_cmap(ATC,'classification_low_resolution')
#category_colors = ecplt.ATC_category_colors
#cmap_tc = ListedColormap(sns.color_palette(category_colors[:len(u)]).as_hex())

fname = "selected_files_Greenland.txt"
fname = "selected_files_Africa.txt"
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
    AEBD = ecio.load_AEBD(filen)
    lats = AEBD['latitude']
    lons = AEBD['longitude']
    
    from cartopy.crs import Globe
    my_globe = Globe(semimajor_axis=6378137, semiminor_axis=6356752.314245179,
                     inverse_flattening=298.257223563)

    fig_name = 'slices_regions/'+fname[15:-4]+'_atlid_orbit_'+orbit_sequence+'.jpg'
    fig_title = orbit_sequence
    projections.plot_on_orthographic(lons,lats, fig_name, fig_title,globe=my_globe)
