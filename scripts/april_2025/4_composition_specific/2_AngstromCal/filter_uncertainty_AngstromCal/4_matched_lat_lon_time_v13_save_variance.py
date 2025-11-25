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
from plotting_tools import read_h5,ATC_category_colors

month = 'december'
which_aerosol = 'dust'
#which_aerosol = 'ssa'
fname = 'sea_salt' if which_aerosol == 'ssa' else 'dust'
tcs   = [11] if which_aerosol == 'ssa' else [10,13,14,15,27] #12=continental pollution?

if which_aerosol == 'dust':
    fname = 'dust_om'

year = '2024' if month == 'december' else '2025'
cams_dir = '/scratch/nld6854/earthcare/cams_data/'+month+'_'+year+'/TTcal/'
srcdir = '/scratch/nld6854/earthcare/earthcare_data/'+month+'_'+year+'/EBD/'

cmap = ecplt.colormaps.chiljet2
ATC = ecio.load_ATC('/scratch/nld6854/earthcare/earthcare_data/march_2025/TC_/ECA_EXBA_ATL_TC__2A_20250321T122819Z_20250913T131504Z_04614F.h5', prodmod_code="ECA_EXBA")

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
    atlid_aod, atlid_lats, atlid_lons, atlid_times = read_h5.get_aod_var(filen, np.array(tcs))

    return (atlid_lats, atlid_lons, atlid_aod)

# run in parallel
ebd_files = sorted(glob.glob(srcdir+'*h5'))
with Pool(processes=8) as pool:  # adjust number of processes
    results = pool.map(process_file, ebd_files)

# combine results
all_lat, all_lon, a_aod = [], [], []
for res in results:
    if res is None:
        continue
    lat, lon, aod = res
    all_lat.extend(lat)
    all_lon.extend(lon)
    a_aod.extend(aod)

# save once
np.savetxt(cams_dir+fname+'_aod355nm_per_composition_'+month+'_'+year+'_cams_atlid_co-located_aod_variance_snr_gr_2.txt',
           np.array([all_lat, all_lon, a_aod]).T,
           header='latitude,longitude,aod_355nm_var_atlid',
           delimiter=',')
