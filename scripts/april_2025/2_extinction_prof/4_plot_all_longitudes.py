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
    land = np.nanmedian(var[np.where(lsm==1)])
    sea = np.nanmedian(var[np.where(lsm==0)])
    print('not weighted by area')
    return land,sea

month='may'
fmonth='May'
vname = 'extinction_coefficient'
figname = 'extinction_coefficient'

cams_dir = '/scratch/nld6854/earthcare/cams_data/'+month+'_2025/'
file_name = cams_dir+'regridded_satellite_total_extinction_coe_2deg_masknan_median_single_alt_'+month+'_2025_snr_gr_2.nc'
cams_file = file_name.replace('satellite','CAMS')#'regridded_CAMS_total_extinction_coe_2deg_masknan_mean_single_alt_'+month+'_2025.nc'

cams_ha = Dataset(cams_file)
cams_h = cams_ha.variables['height'][:]

atlid = Dataset(file_name)
var_atlid = atlid.variables[vname][:,:,:]
print(var_atlid.shape)
var_zonal_atlid = np.nanmedian(var_atlid,axis=1) #zonal mean
lon_atlid = atlid.variables['longitude'][:]
lat_atlid = atlid.variables['latitude'][:]
atlid_h = atlid.variables['height'][:]
print(var_zonal_atlid.shape) #(451, 900, 254)

cams = Dataset(cams_file)
var_cams = cams.variables[vname][:,:,:]
var_zonal_cams = np.nanmedian(var_cams,axis=1) #zonal mean
lon_cams = cams.variables['longitude'][:]
lat_cams = cams.variables['latitude'][:]
cams_h = cams.variables['height'][:]
print(var_cams.shape) #(451, 900, 254)
print(var_zonal_cams.shape)

vmax = 1.e-4

atlid_hpx,lat_atlidpa = np.meshgrid(atlid_h,lat_atlid)
cams_hpx,lat_atlidpc  = np.meshgrid(cams_h,lat_atlid)

fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(8,12))
im=ax1.pcolormesh(lat_atlidpa,atlid_hpx//1000,var_zonal_atlid,cmap=ecplt.colormaps.calipso,norm=matplotlib.colors.LogNorm(vmax=vmax,vmin=vmax/100.))
bar = plt.colorbar(im, orientation='vertical',ax=ax1,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('m${-1}$',fontsize=15)
bar.ax.tick_params(labelsize=15)
ax1.set_title('Zonal median extinction coefficient ATLID '+fmonth+' 2025',fontsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax1.set_ylim(0,20)
ax1.set_ylabel('Altitude / km',fontsize=15)

im=ax2.pcolormesh(lat_atlidpc,cams_hpx//1000,var_zonal_cams,cmap=ecplt.colormaps.calipso,norm=matplotlib.colors.LogNorm(vmax=vmax,vmin=vmax/100.))
bar = plt.colorbar(im, orientation='vertical',ax=ax2,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('m${-1}$',fontsize=15)
bar.ax.tick_params(labelsize=15)
ax2.set_title('Zonal median extinction coefficient CAMS '+fmonth+' 2025',fontsize=15)
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
bar.ax.set_ylabel('m${-1}$',fontsize=15)
bar.ax.tick_params(labelsize=15)
ax3.set_title('CAMS (interpolated) -ATLID',fontsize=15)
ax3.tick_params(axis='x', labelsize=15)
ax3.tick_params(axis='y', labelsize=15)
ax3.set_ylim(0,20)
ax3.set_ylabel('Altitude / km',fontsize=15)

plt.tight_layout()
fig.savefig('time_lat_lon_co_located_zonal_nanmedian_extinction_coefficient_'+figname+'_2deg_median_'+month+'_2025.jpg',bbox_inches='tight')

lon_bins = np.arange(-180.1, 180.1, 2)
lat_cams,cams_hp = np.meshgrid(lat_cams,cams_h)

for ilon in range(var_atlid.shape[1]):
    #interpolating CAMS to ATLID height
    vmax = 1.e-4

    fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(8,12))
    im=ax1.pcolormesh(lat_atlidpa.transpose(),atlid_hpx.transpose()//1000,var_atlid[:,ilon,:].transpose(),cmap=ecplt.colormaps.calipso,norm=matplotlib.colors.LogNorm(vmax=vmax,vmin=vmax/100.))    

    bar = plt.colorbar(im, orientation='vertical',ax=ax1,shrink=0.7, pad=0.1)
    bar.ax.set_ylabel('m${-1}$',fontsize=15)
    bar.ax.tick_params(labelsize=15)
    ax1.set_title('Extinction coefficient ATLID at '+str(lon_atlid[ilon])+' '+fmonth+' 2025',fontsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.set_ylim(0,20)

    im=ax2.pcolormesh(lat_cams,cams_hp//1000,var_cams[:,ilon,:].transpose(),cmap=ecplt.colormaps.calipso,norm=matplotlib.colors.LogNorm(vmax=vmax,vmin=vmax/100.))
    bar = plt.colorbar(im, orientation='vertical',ax=ax2,shrink=0.7, pad=0.1)
    bar.ax.set_ylabel('m${-1}$',fontsize=15)
    bar.ax.tick_params(labelsize=15)
    ax2.set_title('Extinction coefficient CAMS at '+str(lon_cams[ilon])+' '+fmonth+' 2025',fontsize=15)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.set_ylim(0,20)

    #im=ax3.pcolormesh(lat_atlidpa.transpose(),atlid_hpx.transpose()//1000,var_cams_interp2-var_atlid[:,ilon,:].transpose(),cmap='RdBu_r',norm=colors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax))
    cmap = plt.colormaps['RdBu_r']
    im=ax3.pcolormesh(lat_atlidpa.transpose(),atlid_hpx.transpose()//1000,var_cams[:,ilon,:].transpose()-var_atlid[:,ilon,:].transpose(),cmap='RdBu_r',norm=norm)
    bar = plt.colorbar(im, orientation='vertical',ax=ax3,shrink=0.7, pad=0.1)
    bar.ax.set_ylabel('m${-1}$',fontsize=15)
    bar.ax.tick_params(labelsize=15)
    ax3.set_title('CAMS (interpolated) -ATLID',fontsize=15)
    ax3.tick_params(axis='x', labelsize=15)
    ax3.tick_params(axis='y', labelsize=15)
    ax3.set_ylim(0,20)

    plt.tight_layout()
    fig.savefig('all_lons/time_lat_lon_co_located_extinction_coefficient_'+figname+'_at_'+str(lon_bins[ilon])+'_0.4deg_masknan_'+month+'_2025.jpg',bbox_inches='tight')

