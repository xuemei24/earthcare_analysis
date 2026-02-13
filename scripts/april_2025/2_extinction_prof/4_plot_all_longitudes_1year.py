import h5py
import concurrent.futures
import glob
import numpy as np
import cartopy.crs as ccrs
import xarray as xr
from netCDF4 import Dataset
#from pyresample import geometry, kd_tree
from netCDF4 import Dataset
from pylab import *
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import sys
import os
script_path = '/home/nld6854/earthcare_scripts/scripts/april_2025'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), script_path)))

from ectools import ecio
from ectools import ecplot as ecplt
from ectools import colormaps
from plotting_tools import read_h5,ATC_category_colors


def landsea_mean(var):
    cams_lsm = Dataset('/scratch/nld6854/earthcare/cams_data/landsea_mask.nc')
    lsm = cams_lsm.variables['lsm'][0,0]
    land = np.nanmean(var[np.where(lsm==1)])
    sea = np.nanmean(var[np.where(lsm==0)])
    print('not weighted by area')
    return land,sea

vname = 'extinction_coefficient'
figname = 'extinction_coefficient'


months = [
    ('december', '2024'),
    ('january',  '2025'),
    ('february', '2025'),
    ('march',    '2025'),
    ('april',    '2025'),
    ('may',      '2025'),
    ('june',     '2025'),
    ('july',     '2025'),
    ('august',   '2025'),
    ('september','2025'),
    ('october',  '2025'),
    ('november', '2025'),
]

atlid_sum = None
cams_sum  = None
n_months  = 0

atlid_all = np.zeros((12,90,180,241))
cams_all  = np.zeros((12,90,180,241))
lat_all         = np.zeros((12,90))
atlid_h_all     = np.zeros((12,241))
cams_h_all      = np.zeros((12,241))

for i, monthyear in enumerate(months):
    month,year = monthyear[0],monthyear[1]
    print('Processing', month, year)

    cams_dir = '/scratch/nld6854/earthcare/cams_data/' + month + '_' + year + '/'
    file_name = (
        cams_dir +
        'regridded_satellite_total_extinction_coe_2deg_masknan_mean_single_alt_' +
        month + '_' + year + '_snr_gr_2.nc'
    )
    cams_file = file_name.replace('satellite','CAMS')

    atlid = Dataset(file_name)
    cams  = Dataset(cams_file)

    atlid_all[i] = atlid.variables[vname][:]
    cams_all[i]  = cams.variables[vname][:]
    atlid_h_all[i] = atlid.variables['height'][:]
    cams_h_all[i] = cams.variables['height'][:]
    lat_atlid = atlid.variables['latitude'][:]
    lat_cams = cams.variables['latitude'][:]

for i in range(12):
    for j in range(90):
        for k in range(180):
            atlid_all[i,j,k] = np.interp(atlid_h_all[0],atlid_h_all[i],atlid_all[i,j,k])
            cams_all[i,j,k]  = np.interp(atlid_h_all[0],cams_h_all[i],cams_all[i,j,k])

var_zonal_atlid = np.nanmean(np.nanmean(atlid_all, axis=0),axis=1)
var_zonal_cams = np.nanmean(np.nanmean(cams_all, axis=0),axis=1)

vmax = 1.e-4

atlid_hpx,lat_atlidpa = np.meshgrid(atlid_h_all[0],lat_atlid)
cams_hpx,lat_atlidpc  = np.meshgrid(atlid_h_all[0],lat_cams)

print(lat_atlidpa)
print(lat_atlidpc)
fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(8,12))
im=ax1.pcolormesh(lat_atlidpa,atlid_hpx//1000,var_zonal_atlid,cmap=ecplt.colormaps.calipso,norm=matplotlib.colors.LogNorm(vmax=vmax,vmin=vmax/100.))
bar = plt.colorbar(im, orientation='vertical',ax=ax1,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('m$^{-1}$',fontsize=15)
bar.ax.tick_params(labelsize=15)
ax1.set_title('Zonal mean extinction coefficient ATLID Dec 2024 - Nov 2025',fontsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax1.set_ylim(0,20)
ax1.set_ylabel('Altitude / km',fontsize=15)

im=ax2.pcolormesh(lat_atlidpc,cams_hpx//1000,var_zonal_cams,cmap=ecplt.colormaps.calipso,norm=matplotlib.colors.LogNorm(vmax=vmax,vmin=vmax/100.))
bar = plt.colorbar(im, orientation='vertical',ax=ax2,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('m$^{-1}$',fontsize=15)
bar.ax.tick_params(labelsize=15)
ax2.set_title('Zonal mean extinction coefficient CAMS Dec 2024 - Nov 2025',fontsize=15)
ax2.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)
ax2.set_ylim(0,20)
ax2.set_ylabel('Altitude / km',fontsize=15)

linthresh = 1e-6
#im=ax3.pcolormesh(lat_atlidpa,atlid_hpx//1000,var_cams_interp-var_zonal_atlid,cmap='RdBu_r',norm=colors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax))
cmap = plt.colormaps['RdBu_r']
bounds = -np.logspace(-6,-4,num=5)
bounds = np.sort(np.append(bounds,-bounds))
print(bounds)
norm = colors.BoundaryNorm(bounds, ncolors=cmap.N, clip=True)
im=ax3.pcolormesh(lat_atlidpa,atlid_hpx//1000,var_zonal_cams-var_zonal_atlid,cmap='RdBu_r',norm=norm)
bar = plt.colorbar(im, orientation='vertical',ax=ax3,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('m$^{-1}$',fontsize=15)
bar.ax.tick_params(labelsize=15)
ax3.set_title('CAMS (interpolated) -ATLID',fontsize=15)
ax3.tick_params(axis='x', labelsize=15)
ax3.tick_params(axis='y', labelsize=15)
ax3.set_ylim(0,20)
ax3.set_ylabel('Altitude / km',fontsize=15)

plt.tight_layout()
fig.savefig('time_lat_lon_co_located_zonal_nanmean_extinction_coefficient_'+figname+'_2deg_mean_122024-112025.jpg',bbox_inches='tight')


