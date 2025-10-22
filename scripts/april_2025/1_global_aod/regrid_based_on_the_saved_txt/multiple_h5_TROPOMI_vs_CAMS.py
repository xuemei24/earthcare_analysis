import h5py
import concurrent.futures
import glob
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
from scipy.stats import binned_statistic_2d
from pylab import *
import sys
import pandas as pd
import matplotlib.colors as colors
import sys
import os

script_path = '/home/nld6854/earthcare_scripts/scripts/april_2025'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), script_path)))
from plotting_tools import statistics

month = 'april'
which_aerosol='total'

mean_or_std = 'mean'

lat_bins = np.arange(-90.1, 90.1, 2)
lon_bins = np.arange(-180.1, 180.1, 2)

if month == 'april':
    tropomi_dir = '/net/pc230039/nobackup/users/hemminga/earthcare/LSR_ATLID/TROPOMI_data/s5p-l3grd-aod-354nm-001-month-20250401-20250210.nc'
elif month == 'march':
    tropomi_dir = '/net/pc230039/nobackup/users/hemminga/earthcare/LSR_ATLID/TROPOMI_data/s5p-l3grd-aod-354nm-001-month-20250301-20250210.nc'
tropomi = Dataset(tropomi_dir)
t_aod = tropomi.variables['aerosol_optical_depth'][:]
uncertainty = tropomi.variables['aerosol_optical_depth_uncertainty'][:]
t_aod = np.where(uncertainty<1,t_aod,np.nan)
t_lat = tropomi.variables['latitude'][:]
t_lon = tropomi.variables['longitude'][:]
t_lon,t_lat = np.meshgrid(t_lon,t_lat)

t_lon_1 = t_lon.flatten()
t_lat_1 = t_lat.flatten()
t_aod_1 = t_aod.flatten()

mask = ~np.isnan(t_aod_1)
lon_t = t_lon_1[mask]
lat_t = t_lat_1[mask]
aod_t = t_aod_1[mask]

all_lat = lat_t
all_lon = lon_t
all_aod = aod_t

print(aod_t.shape,lat_t.shape,lon_t.shape)
stat, x_edge, y_edge, _ = binned_statistic_2d(
    lat_t, lon_t, aod_t, statistic=mean_or_std, bins=[lat_bins, lon_bins])

# Replace NaN with zeros (or another value if necessary)
#stat = np.nan_to_num(stat)

aod_tropomi = stat
#aod_cams = np.where(aod_cams>0,aod_cams,np.nan)
print(np.isnan(aod_tropomi).sum(), "grid cells have missing data")

# Replace NaN with zeros (or another value if necessary)
#stat = np.nan_to_num(stat)

lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

clons,clats = np.meshgrid(lon_centers,lat_centers)
nan_percentage = np.isnan(stat).sum() / stat.size * 100
print(f"Percentage of NaNs in regridded data: {nan_percentage:.2f}%")

#test

cams_dir = '/net/pc190625/nobackup_1/users/wangxu/cams_data/'
cams = Dataset(cams_dir+'total_aerosol_optical_depth_355nm_'+month+'_2025.nc')
#aod_cams = np.mean(np.mean(cams.variables['aod355'][:],axis=0),axis=0)
aod_o_cams = cams.variables['aod355'][::3]
print('aod_o_cams.shape',aod_o_cams.shape)

clwc0 = Dataset(cams_dir+'lwc/specific_cloud_liquid_water_content_'+month+'_2025_0.nc')
clwc3 = Dataset(cams_dir+'lwc/specific_cloud_liquid_water_content_'+month+'_2025_3.nc')
clwc6 = Dataset(cams_dir+'lwc/specific_cloud_liquid_water_content_'+month+'_2025_6.nc')
clwc9 = Dataset(cams_dir+'lwc/specific_cloud_liquid_water_content_'+month+'_2025_9.nc')

#delete columns where there are liquid clouds
def where_cloudy(cloud,data):
    print('before',data.shape)
    mask = (cloud>=0.0001).any(axis=1)
    print('mask_expanded.shape',mask.shape)
    data = np.where(mask, np.nan, data)
    print('after',data.shape)
    return data

aod_cams0 = np.nanmean(where_cloudy(clwc0.variables['clwc'][0,:],aod_o_cams[0]),axis=0)
aod_cams3 = np.nanmean(where_cloudy(clwc3.variables['clwc'][0,:],aod_o_cams[1]),axis=0)
aod_cams6 = np.nanmean(where_cloudy(clwc6.variables['clwc'][0,:],aod_o_cams[2]),axis=0)
aod_cams9 = np.nanmean(where_cloudy(clwc9.variables['clwc'][0,:],aod_o_cams[3]),axis=0)
aod_cams_0 = (aod_cams0+aod_cams3+aod_cams6+aod_cams9)/4.

lat_cams_0 = cams.variables['latitude'][:]
lon_cams_0 = cams.variables['longitude'][:]
ilon = np.where(lon_cams_0>=180)
lon_cams_0[ilon] = lon_cams_0[ilon]-360.

lon_cams_1,lat_cams_1 = np.meshgrid(lon_cams_0,lat_cams_0)
lon_cams_1 = lon_cams_1.flatten()
lat_cams_1 = lat_cams_1.flatten()
aod_cams_1 = aod_cams_0.flatten()

mask = ~np.isnan(aod_cams_1)
aod_cams = aod_cams_1[mask]
lat_cams = lat_cams_1[mask]
lon_cams = lon_cams_1[mask]

print(aod_cams.shape,lat_cams.shape,lon_cams.shape)
stat, x_edge, y_edge, _ = binned_statistic_2d(
    lat_cams, lon_cams, aod_cams, statistic=mean_or_std, bins=[lat_bins, lon_bins])

# Replace NaN with zeros (or another value if necessary)
#stat = np.nan_to_num(stat)

aod_cams = stat
#aod_cams = np.where(aod_cams>0,aod_cams,np.nan)
print(np.isnan(aod_cams).sum(), "grid cells have missing data")

vmax = aod_cams.max()
vmax = 1
print('aod_cams.max()=',aod_cams.max())

def landsea_mean(var):
    cams_lsm = Dataset('/net/pc190625/nobackup_1/users/wangxu/cams_data/landsea_mask.nc')
    lsm0 = cams_lsm.variables['lsm'][0,0]

    stat, x_edge, y_edge, _ = binned_statistic_2d(
    lat_cams_1, lon_cams_1, lsm0.flatten(), statistic='mean', bins=[lat_bins, lon_bins])

    lsm = np.round(stat)
    print(lsm.shape,var.shape)

    land = np.nanmean(var[np.where(lsm==1)])
    sea = np.nanmean(var[np.where(lsm==0)])
    print('not weighted by area')
    return land,sea

aland,asea = landsea_mean(aod_tropomi)
print('TROPOMI land=',aland,'sea=',asea)
cland,csea = landsea_mean(aod_cams)
print('CAMS land=',cland,'sea=',csea)

print("Satellite Data Lat Range:", np.min(all_lat), np.max(all_lat))
print("Satellite Data Lon Range:", np.min(all_lon), np.max(all_lon))
print("Grid Lat Range:", np.min(clats), np.max(clats))
print("Grid Lon Range:", np.min(clons), np.max(clons))

print('CAMS AOD >0 mean=',np.nanmean(aod_cams[aod_tropomi>0]))
print('TROPOMI AOD >0 mean=',np.nanmean(aod_tropomi[aod_tropomi>0]))
print('CAMS-TROPOMI >0 mean=',np.nanmean(aod_cams[aod_tropomi>0]-aod_tropomi[aod_tropomi>0]))

print('CAMS AOD mean=',np.nanmean(aod_cams[aod_tropomi>-9]))
print('TROPOMI AOD mean=',np.nanmean(aod_tropomi[aod_tropomi>-9]))
print('CAMS-TROPOMI mean=',np.nanmean(aod_cams[aod_tropomi>-9]-aod_tropomi[aod_tropomi>-9]))
print('CAMS,TROPOMI NMB=',statistics.normalized_mean_bias(aod_cams,aod_tropomi))

#fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(20,30),subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0)))
fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(20,30),subplot_kw=dict(projection=ccrs.PlateCarree()))
#plt.figure(figsize=(6,3))
#ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax1.set_extent([-180, 180, -89.9, 89.9])#,crs=ccrs.PlateCarree(central_longitude=180))
all_lon = np.where(clons>0,clons,clons+360)
im=ax1.pcolormesh(all_lon,clats,aod_tropomi,cmap='plasma',transform=ccrs.PlateCarree(),norm=colors.LogNorm(vmin=1e-2, vmax=1))
gl = ax1.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
             linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False

ax1.coastlines(resolution='110m')#,color='white')
ax1.gridlines()
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

bar = plt.colorbar(im, orientation='vertical',ax=ax1,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('TROPOMI Particle optical depth / -',fontsize=15)
bar.ax.tick_params(labelsize=15)

ax1.set_title('TROPOMI AOD',fontsize=15)



ax2.set_extent([-180, 180, -89.9, 89.9])
im=ax2.pcolormesh(clons,clats,aod_cams,cmap='plasma',transform=ccrs.PlateCarree(),norm=colors.LogNorm(vmin=1e-2, vmax=1))
gl = ax2.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
             linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False

ax2.coastlines(resolution='110m')
ax2.gridlines()
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

bar = plt.colorbar(im, orientation='vertical',ax=ax2,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('CAMS Particle optical depth / -',fontsize=15)
bar.ax.tick_params(labelsize=15)
ax2.set_title('CAMS total AOD',fontsize=15)

ax3.set_extent([-180, 180, -89.9, 89.9])
im=ax3.pcolormesh(clons,clats,aod_cams-aod_tropomi,cmap='RdBu_r',transform=ccrs.PlateCarree(),vmax=vmax,vmin=-vmax)
gl = ax3.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
             linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False

ax3.coastlines(resolution='110m')
ax3.gridlines()
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

bar = plt.colorbar(im, orientation='vertical',ax=ax3,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('CAMS-TROPOMI Particle optical depth / -',fontsize=15)
bar.ax.tick_params(labelsize=15)
ax3.set_title('CAMS-TROPOMI AOD',fontsize=15)

plt.tight_layout()
fig.savefig('global_aod_'+which_aerosol+'_2deg_binned_'+mean_or_std+'_'+month+'_2025_trimedges_filterCAMScloud_TC_vs_TROPOMI.jpg',bbox_inches='tight')

nbins = 150
binsc = np.linspace(0,aod_cams.max(),nbins)
histc,binsc = np.histogram(aod_cams,bins=binsc,density=False)
bcc = 0.5*(binsc[1:] + binsc[:-1])

binsa = np.linspace(0,np.nanmax(aod_tropomi),nbins)
hista,binsa = np.histogram(aod_tropomi,bins=binsa,density=False)
bca = 0.5*(binsa[1:] + binsa[:-1])

binsa0 = np.linspace(0,np.nanmax(all_aod),nbins)
hista0,binsa0 = np.histogram(all_aod,bins=binsa0,density=False)
bca0 = 0.5*(binsa0[1:] + binsa0[:-1])

binsa_ = np.linspace(0,np.nanmax(aod_tropomi),nbins)
hista_,binsa_ = np.histogram(aod_tropomi,bins=binsa_,density=True)
bca_ = 0.5*(binsa_[1:] + binsa_[:-1])

binsa0_ = np.linspace(0,np.nanmax(all_aod),nbins)
hista0_,binsa0_ = np.histogram(all_aod,bins=binsa0_,density=True)
bca0_ = 0.5*(binsa0_[1:] + binsa0_[:-1])

fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5),sharey=False)
ax1.plot(bcc,histc,label='CAMS')
ax1.plot(bca,hista,label='TROPOMI')
ax2.plot(bca0,hista0,label='original TROPOMI')
ax3.plot(bca0_,hista0_,label='original TROPOMI')
ax3.plot(bca_,hista_,label='regridded TROPOMI')
ax2.set_ylim(1e1,1e7)
ax2.set_yscale('log')
#ax3.set_ylim(1e1,1e7)
#ax3.set_yscale('log')

ax1.set_xlabel('AOD',fontsize=15) 
ax1.set_ylabel('Counts',fontsize=15)
ax2.set_xlabel('AOD',fontsize=15)
ax2.set_ylabel('Counts',fontsize=15)
ax3.set_xlabel('AOD',fontsize=15)
ax3.set_ylabel('Density',fontsize=15)

ax1.tick_params(labelsize=12)
ax2.tick_params(labelsize=12)
ax3.tick_params(labelsize=12)

ax1.legend(frameon=False,fontsize=15)
ax2.legend(frameon=False,fontsize=15)
ax3.legend(frameon=False,fontsize=15)
fig.savefig('histograms_CAMS_TROPOMI_2deg_binned_'+mean_or_std+'_'+month+'_2025_trimedges_filterCAMScloud_TC.jpg')


