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

from ectools import ecio
from ectools import ecplot as ecplt
from ectools import colormaps
from plotting_tools import read_h5,ATC_category_colors,projections

month = 'july'
forecast_period = '3'

year = '2024' if month == 'december' else '2025'
# CAMS Data (xxx)
cams_dir = '/scratch/nld6854/earthcare/cams_data/'+month+'_'+year+'/'

srcdir = '/scratch/nld6854/earthcare/earthcare_data/'+month+'_'+year+'/CAMS/'

cmap = ecplt.colormaps.chiljet2
ATC = ecio.load_ATC('/scratch/nld6854/earthcare/earthcare_data/march_2025/TC_/ECA_EXBA_ATL_TC__2A_20250321T122819Z_20250913T131504Z_04614F.h5', prodmod_code="ECA_EXBA")

cmap_tc,bounds,categories_formatted,norm_tc = ATC_category_colors.ecplt_cmap(ATC,'classification_low_resolution')

cams_files = sorted(glob.glob(srcdir+'*nc'))
def atlid_cams_ext(cams_file):
    filen = cams_file.replace('CAMS','EBD')
    filen = filen.replace('nc','h5')
    print(filen)
    orbit_sequence=filen[-9:-3]
    atlid_extcoe,atlid_lats,atlid_lons,atlid_times,atlid_h,tc_cld,tc_all,err = read_h5.get_ext(filen,np.array([10,11,12,13,14,15,25,26,27]))

    from cartopy.crs import Globe
    my_globe = Globe(semimajor_axis=6378137, semiminor_axis=6356752.314245179,
                     inverse_flattening=298.257223563)

    fig_name = 'slices_regions/atlid_orbit_'+orbit_sequence+'.jpg'
    fig_title = orbit_sequence
    projections.plot_on_orthographic(atlid_lons,atlid_lats, fig_name, fig_title,central_longitude=atlid_lons[len(atlid_lons)//2],central_latitude=atlid_lats[len(atlid_lons)//2],globe=my_globe)

for fname in cams_files:
    atlid_cams_ext(fname)
#with Pool(processes=8) as pool:  # adjust number of processes
#    pool.map(atlid_cams_ext, cams_files)

