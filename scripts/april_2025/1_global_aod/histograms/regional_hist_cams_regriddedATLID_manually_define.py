import h5py
import concurrent.futures
import glob
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
from pyresample import geometry, kd_tree
from scipy.stats import binned_statistic_2d
from pylab import *
import sys
import pandas as pd
import matplotlib.colors as colors

#Europe
lat_min, lat_max = 35, 71  # Latitude range
lon_min, lon_max = -10, 30  # Longitude range
region_name = 'europe'

##North Atlantic
lat_min, lat_max = 20, 60  # Latitude range
lon_min, lon_max = -60, 0  # Longitude range
region_name = 'north_atlantic'

#Asia (China & India)
lat_min, lat_max = 0,40
lon_min, lon_max = 65,125
region_name = 'china_india'

##Africa
lat_min, lat_max = -15,30
lon_min, lon_max = -30,30
region_name = 'africa'

###Amazon
#lat_min, lat_max = -20,10
#lon_min, lon_max = -80,-40
#region_name = 'amazon'

##Arctic
lat_min, lat_max = 66.5,90
lon_min, lon_max = -180,180
region_name = 'arctic'


# Load your dataset (assuming it's a CSV with 'lat', 'lon', 'aod')
df = pd.read_csv("../2025_april_aod.txt", delimiter=",")  # Adjust delimiter if needed

#simple_classification
which_aerosol='total'
#which_aerosol='sea_salt'
#which_aerosol='dust'
# Define a function that reads data from an HDF5 file

#2 deg resolution
all_aod = df['aod355nm'].values
all_lat = df['# latitude'].values
all_lon = df['longitude'].values

print('max and min of latitude before mask',np.nanmax(all_lat),np.nanmin(all_lat))
mask = ~np.isnan(all_aod)
all_aod = all_aod[mask]
all_lat = all_lat[mask]
all_lon = all_lon[mask]

lat_bins = np.arange(-90.1, 90.1, 2)
lon_bins = np.arange(-180.1, 180.1, 2)

# Compute 2D histogram for the mean wind
stat, x_edge, y_edge, _ = binned_statistic_2d(
    all_lat, all_lon, all_aod, statistic='mean', bins=[lat_bins, lon_bins])

# Replace NaN with zeros (or another value if necessary)
#stat = np.nan_to_num(stat)

lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

clons,clats = np.meshgrid(lon_centers,lat_centers)
nan_percentage = np.isnan(stat).sum() / stat.size * 100
print(f"Percentage of NaNs in regridded data: {nan_percentage:.2f}%")

#plot the figures
aod_atlid = stat

#data selection according to lat and lon ranges
mask = (all_lat >= lat_min) & (all_lat <= lat_max) & (all_lon >= lon_min) & (all_lon <= lon_max)
all_aod_masked = np.ma.masked_where(~mask, all_aod)
all_aod_masked = np.ma.masked_where(~mask, all_aod)

mask = (clats >= lat_min) & (clats <= lat_max) & (clons >= lon_min) & (clons <= lon_max)
aod_atlid_masked = np.ma.masked_where(~mask, aod_atlid)
aod_atlid_masked = np.ma.filled(aod_atlid_masked, np.nan)



#test

cams = Dataset('/net/pc190625/nobackup_1/users/wangxu/cams_data/total_aerosol_optical_depth_355nm_apr_2025.nc')
aod_cams = np.mean(np.mean(cams.variables['aod355'][:],axis=0),axis=0)
lat_cams = cams.variables['latitude'][:]
lon_cams = cams.variables['longitude'][:]
ilon = np.where(lon_cams>=180)
lon_cams[ilon] = lon_cams[ilon]-360.

lon_cams,lat_cams = np.meshgrid(lon_cams,lat_cams)
lon_cams = lon_cams.flatten()
lat_cams = lat_cams.flatten()
aod_cams = aod_cams.flatten()
print(aod_cams.shape,lat_cams.shape,lon_cams.shape)
stat, x_edge, y_edge, _ = binned_statistic_2d(
    lat_cams, lon_cams, aod_cams, statistic='mean', bins=[lat_bins, lon_bins])

# Replace NaN with zeros (or another value if necessary)
#stat = np.nan_to_num(stat)

aod_cams = stat
#aod_cams = np.where(aod_cams>0,aod_cams,np.nan)
print(np.isnan(aod_cams).sum(), "grid cells have missing data")

vmax = aod_cams.max()
vmax = 1
print('aod_cams.max()=',aod_cams.max())


print("Satellite Data Lat Range:", np.min(all_lat), np.max(all_lat))
print("Satellite Data Lon Range:", np.min(all_lon), np.max(all_lon))
print("Grid Lat Range:", np.min(clats), np.max(clats))
print("Grid Lon Range:", np.min(clons), np.max(clons))

print('CAMS AOD mean=',np.mean(aod_cams[aod_atlid>0]))
print('ATLID AOD mean=',np.mean(aod_atlid[aod_atlid>0]))
print('CAMS-ATLID mean=',np.mean(aod_cams[aod_atlid>0]-aod_atlid[aod_atlid>0]))

print('CAMS AOD mean=',np.mean(aod_cams[aod_atlid>-9]))
print('ATLID AOD mean=',np.mean(aod_atlid[aod_atlid>-9]))
print('CAMS-ATLID mean=',np.mean(aod_cams[aod_atlid>-9]-aod_atlid[aod_atlid>-9]))

#data selection according to lat and lon ranges
aod_cams_masked = np.ma.masked_where(~mask, aod_cams)
aod_cams_masked = np.ma.filled(aod_cams_masked, np.nan)



#fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(20,30),subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0)))
fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(6,10),subplot_kw=dict(projection=ccrs.PlateCarree()))
ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax3.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
#plt.figure(figsize=(6,3))
#ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
#ax1.set_extent([-180, 180, -89.9, 89.9])#,crs=ccrs.PlateCarree(central_longitude=180))
all_lon = np.where(clons>0,clons,clons+360)
im=ax1.pcolormesh(all_lon,clats,aod_atlid_masked,cmap='plasma',transform=ccrs.PlateCarree(),norm=colors.LogNorm(vmin=1e-2, vmax=1))
gl = ax1.gridlines(crs=ccrs.PlateCarree(central_longitude=0),draw_labels=["bottom", "left"],
             linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False

ax1.coastlines(resolution='110m')#,color='white')
ax1.gridlines()
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

bar = plt.colorbar(im, orientation='vertical',ax=ax1,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('AOD / -',fontsize=15)
bar.ax.tick_params(labelsize=15)

ax1.set_title('ATLID integrated AOD',fontsize=15)



#ax2.set_extent([-180, 180, -89.9, 89.9])
im=ax2.pcolormesh(clons,clats,aod_cams_masked,cmap='plasma',transform=ccrs.PlateCarree(),norm=colors.LogNorm(vmin=1e-2, vmax=1))
gl = ax2.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=["bottom", "left"],
             linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False

ax2.coastlines(resolution='110m')
ax2.gridlines()
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

bar = plt.colorbar(im, orientation='vertical',ax=ax2,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('AOD / -',fontsize=15)
bar.ax.tick_params(labelsize=15)

ax2.set_title('CAMS total AOD',fontsize=15)

#ax3.set_extent([-180, 180, -89.9, 89.9])
im=ax3.pcolormesh(clons,clats,aod_cams_masked-aod_atlid_masked,cmap='RdBu_r',transform=ccrs.PlateCarree(),vmax=vmax/2.,vmin=-vmax/2.)
gl = ax3.gridlines(crs=ccrs.PlateCarree(central_longitude=0),
             linewidth=2, color='gray', alpha=0.5, linestyle='--',draw_labels=["bottom", "left"])
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False

ax3.coastlines(resolution='110m')
ax3.gridlines()
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

bar = plt.colorbar(im, orientation='vertical',ax=ax3,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('AOD / -',fontsize=15)
bar.ax.tick_params(labelsize=15)

ax3.set_title('CAMS-ATLID AOD',fontsize=15)

plt.tight_layout()
fig.savefig(region_name+'_aod_'+which_aerosol+'_2deg_log_mean_april_2025.jpg',bbox_inches='tight')

nbins = 150
binsc = np.linspace(0,np.nanmax(aod_cams_masked),nbins)
histc,binsc = np.histogram(aod_cams_masked,bins=binsc,density=False)
bcc = 0.5*(binsc[1:] + binsc[:-1])

binsa = np.linspace(0,np.nanmax(aod_atlid_masked),nbins)
hista,binsa = np.histogram(aod_atlid_masked,bins=binsa,density=False)
bca = 0.5*(binsa[1:] + binsa[:-1])

binsa0 = np.linspace(0,np.nanmax(all_aod_masked),nbins)
hista0,binsa0 = np.histogram(all_aod_masked,bins=binsa0,density=False)
bca0 = 0.5*(binsa0[1:] + binsa0[:-1])

binsa_ = np.linspace(0,np.nanmax(aod_atlid_masked),nbins)
hista_,binsa_ = np.histogram(aod_atlid_masked,bins=binsa_,density=True)
bca_ = 0.5*(binsa_[1:] + binsa_[:-1])

binsa0_ = np.linspace(0,np.nanmax(all_aod_masked),nbins)
hista0_,binsa0_ = np.histogram(all_aod_masked,bins=binsa0_,density=True)
bca0_ = 0.5*(binsa0_[1:] + binsa0_[:-1])

fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5),sharey=False)
ax1.plot(bcc,histc,label='CAMS')
ax1.plot(bca,hista,label='regridded ATLID')
ax2.plot(bca0,hista0,label='original ATLID')
ax3.plot(bca0_,hista0_,label='original ATLID')
ax3.plot(bca_,hista_,label='regridded ATLID')
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
fig.savefig(region_name+'_histograms_CAMS_ATLID_2deg_binned_mean_april_2025.jpg')

#plot regional domain averaged exxtinction profile
#the 2 files below have been regridded to 2 deg with binned method
fname='/usr/people/wangxu/Desktop/earthcare_scripts/scripts/april_2025/2_extinction_prof/regridded_satellite_total_extinction_coe_2deg_masknan_mean_single_alt_april_2025.nc'
a=Dataset(fname)
extinction = a.variables['extinction_coefficient'][:]
height = a.variables['height'][:]

expanded_mask = mask[:,:,None]
expanded_mask = np.broadcast_to(expanded_mask, extinction.shape)

extinction_masked = np.ma.masked_where(~expanded_mask, extinction)
extinction_masked = np.ma.filled(extinction_masked,np.nan)

extinc_prof = np.nanmean(extinction_masked,axis=0)
extinc_prof = np.nanmean(extinc_prof,axis=0)

ext_temp_atlid = extinction_masked.reshape(-1,extinction_masked.shape[2])
uncertainty_atlid = np.nanstd(ext_temp_atlid,axis=0)/np.sqrt(np.sum(np.isfinite(ext_temp_atlid),axis=0))

fcams = '/usr/people/wangxu/Desktop/earthcare_scripts/scripts/april_2025/2_extinction_prof/regridded_cams_extinction_coe_2deg_mean_april_2025.nc'
a=Dataset(fcams)
extinction_cams = a.variables['extinction_coefficient_cams'][:]
h_cams = a.variables['height'][:]

expanded_mask_cams = mask[:,:,None]
expanded_mask_cams = np.broadcast_to(expanded_mask_cams, extinction_cams.shape)

extinction_masked_cams = np.ma.masked_where(~expanded_mask_cams, extinction_cams)
extinction_masked_cams = np.ma.filled(extinction_masked_cams,np.nan)

extinc_prof_cams = np.nanmean(extinction_masked_cams,axis=0)
extinc_prof_cams = np.nanmean(extinc_prof_cams,axis=0)

ext_temp_cams = extinction_masked_cams.reshape(-1,extinction_masked_cams.shape[2])
uncertainty_cams = np.nanstd(ext_temp_cams,axis=0)/np.sqrt(np.sum(np.isfinite(ext_temp_cams),axis=0))

fig,ax1=plt.subplots(1,figsize=(4,5))
ax1.plot(extinc_prof,height//1000,'k-',label='ATLID')
ax1.fill_betweenx(height//1000,extinc_prof-uncertainty_atlid,extinc_prof+uncertainty_atlid,
                  color='gray',alpha=0.5,label='ATLID standard error')
ax1.plot(extinc_prof_cams,h_cams//1000,'r-',label='CAMS')
ax1.fill_betweenx(h_cams//1000,extinc_prof_cams-uncertainty_cams,extinc_prof_cams+uncertainty_cams,
                  color='red',alpha=0.3,label='CAMS standard error')
ax1.set_xlabel('Extinction coefficient',fontsize=15)
ax1.set_ylabel('Altitude',fontsize=15)
ax1.set_ylim(0,20)
ax1.tick_params(axis='both', labelsize=15)
ax1.legend(frameon=False)
plt.tight_layout()
fig.savefig(region_name+'_extinction_profile_CAMS_ATLID_2deg_binned_mean_april_2025.jpg')

