import pandas as pd
import numpy as np
import xarray as xr

def get_AOD(aod_old, wave_old, wave_new, angstrom):
    print('aod_old>0',aod_old[aod_old>0], 'wave_old',wave_old, 'wave_new',wave_new, 'angstrom>0',angstrom[angstrom>0])
    return ((wave_new / wave_old) ** (-angstrom)) * aod_old

#Use this script after ATLID AOD has been processed
month = '04'
month2 = 'April'
month3 = 'april'

aeronet_path = '/net/pc190625/nobackup_1/users/wangxu/aeronet/'
df = pd.read_table(aeronet_path+'2025'+month+'_all_sites_aod15_dailyAVG.txt', delimiter=',', header=[7])
df = df.replace(-999.0, np.nan)

print(df.keys())
sites = df['AERONET_Site']
print(sites)
print(len(sites))
aod340 = df['AOD_340nm']
angstrom = df['340-440_Angstrom_Exponent']
aod355 = get_AOD(aod340,340,355,angstrom)
df['AOD_355nm']=aod355
print('aod355',aod355,len(aod355))

print(df['Date(dd:mm:yyyy)'])
#df['Date(dd:mm:yyyy)'] = pd.to_datetime(df['Date(dd:mm:yyyy)'])
df['Date'] = pd.to_datetime(df['Date(dd:mm:yyyy)'], format='%d:%m:%Y')

df['Month'] = df['Date'].dt.to_period('M')  # e.g., 2025-04
print("df['Date']",df['Date'])
print("df['Month']",df['Month'])
#monthly_avg = df.groupby(['AERONET_Site','Site_Latitude(Degrees)','Site_Longitude(Degrees)', 'Month'])['AOD_355nm'].mean().reset_index()
monthly_avg = (
    df.groupby(['AERONET_Site', 'Site_Latitude(Degrees)', 'Site_Longitude(Degrees)', 'Month'])
    .agg({'AOD_355nm': np.nanmean})
    .reset_index()
)
df_final = monthly_avg[monthly_avg['Month'] == '2025-'+month]
print(df_final,len(df_final))
aod = df_final['AOD_355nm']

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors

# --- Load your DataFrame ---
# Make sure df_final is ready and contains:
# 'AOD_355nm', 'Site_Latitude(Degrees)', 'Site_Longitude(Degrees)', 'AERONET_Site'

# --- Create the plot ---
fig,ax=plt.subplots(1,figsize=(20,12),subplot_kw=dict(projection=ccrs.PlateCarree()))
#plt.figure(figsize=(6,3))
#ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([-180, 180, -89.9, 89.9])#,crs=ccrs.PlateCarree(central_longitude=180))
gl = ax.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
             linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False

ax.coastlines(resolution='110m')#,color='white')
ax.gridlines()
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

# Plot AOD values
scatter = ax.scatter(
    df_final['Site_Longitude(Degrees)'],
    df_final['Site_Latitude(Degrees)'],
    c=df_final['AOD_355nm'],
    s=60,
    edgecolor='k',
    cmap='plasma',
    norm=colors.LogNorm(vmin=1e-2, vmax=1),
    transform=ccrs.PlateCarree()
)

bar = plt.colorbar(scatter, orientation='vertical',ax=ax,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('AERONET Particle optical depth / -',fontsize=15)
bar.ax.tick_params(labelsize=15)

ax.set_title('AERONET AOD '+month2,fontsize=15)


plt.tight_layout()
fig.savefig('global_AERONET_aod_'+month2+'_2025.jpg',bbox_inches='tight')

#regrid
from scipy.stats import binned_statistic_2d,pearsonr

# Your grid bins (edges)
reso = 2
lat_bins = np.arange(-90., 90.+reso, reso)
lon_bins = np.arange(-180, 180.+reso, reso)

# Compute bin centers
lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2

stat, x_edges, y_edges, _ = binned_statistic_2d(
    df_final['Site_Latitude(Degrees)'],
    df_final['Site_Longitude(Degrees)'],
    df_final['AOD_355nm'],
    statistic='mean',
    bins=[lat_bins, lon_bins]
)

grid_aod_2d = stat
print(len(stat.flatten()),len(stat[stat>=0]))

# Now grid_aod_2d is your regridded AOD data on 2x2 grid
# --- Create the plot ---
fig,ax=plt.subplots(1,figsize=(20,12),subplot_kw=dict(projection=ccrs.PlateCarree()))
#plt.figure(figsize=(6,3))
#ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([-180, 180, -89.9, 89.9])#,crs=ccrs.PlateCarree(central_longitude=180))
gl = ax.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
             linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False

ax.coastlines(resolution='110m')#,color='white')
ax.gridlines()
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

# Plot AOD values
im = ax.pcolormesh(
    lon_centers,
    lat_centers,
    grid_aod_2d,
    cmap='plasma',
    norm=colors.LogNorm(vmin=1e-2, vmax=1),
    transform=ccrs.PlateCarree()
)

bar = plt.colorbar(im, orientation='vertical',ax=ax,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('AERONET Particle optical depth / -',fontsize=15)
bar.ax.tick_params(labelsize=15)

ax.set_title('AERONET 2 Deg AOD '+month2,fontsize=15)


plt.tight_layout()
fig.savefig('global_2deg_AERONET_aod_'+month2+'_2025.jpg',bbox_inches='tight')

regridded_data_xr = xr.Dataset(
    {
        "aod355_aeronet": (["latitude", "longitude"], stat)
    },
    coords={
        "latitude": lat_centers,
        "longitude": lon_centers
    }
)
output_filename = "regridded_AERONET_aod355_2deg_"+month2+"_2025.nc"
regridded_data_xr.to_netcdf(output_filename)


import os
import sys
script_path = '/home/nld6854/earthcare_scripts/scripts/april_2025'
sys.path.append(script_path)
from plotting_tools import statistics

file_dir = '/net/pc190625/nobackup_1/users/wangxu/cams_data/'
print('Month=',month3)
df = pd.read_csv(file_dir+"2025_"+month3+"_cams_atlid_co-located_aod.txt", delimiter=",")

#simple_classification
which_aerosol='total'
#which_aerosol='sea_salt'
#which_aerosol='dust'
# Define a function that reads data from an HDF5 file

mean_or_std = 'mean'

#2 deg resolution
a_aod = df['aod355nm_atlid'].values
c_aod = df['aod355nm_cams'].values
all_lat = df['# latitude'].values
all_lon = df['longitude'].values

print('max and min of latitude before mask',np.nanmax(all_lat),np.nanmin(all_lat))
mask = ~np.isnan(a_aod) & ~np.isnan(c_aod)
a_aod = a_aod[mask]
c_aod = c_aod[mask]
all_lat = all_lat[mask]
all_lon = all_lon[mask]

reso = 2
lat_bins = np.arange(-90., 90.+reso, reso)
lon_bins = np.arange(-180, 180.+reso, reso)

# Compute 2D histogram for the mean wind
aod_atlid, x_edge, y_edge, _ = binned_statistic_2d(
    all_lat, all_lon, a_aod, statistic=mean_or_std, bins=[lat_bins, lon_bins])

aod_cams, x_edge, y_edge, _ = binned_statistic_2d(
    all_lat, all_lon, c_aod, statistic=mean_or_std, bins=[lat_bins, lon_bins])

# Replace NaN with zeros (or another value if necessary)
#stat = np.nan_to_num(stat)

lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

clons,clats = np.meshgrid(lon_centers,lat_centers)
nan_percentage = np.isnan(aod_cams).sum() / aod_cams.size * 100
print(f"Percentage of NaNs in regridded data: {nan_percentage:.2f}%")

print('*********Differences between AERONET & ATLID*********')
mask = ~np.isnan(grid_aod_2d) & ~np.isnan(aod_atlid)
print('ATLID mean=',np.nanmean(aod_atlid[mask]))
print('AERONET mean=',np.nanmean(grid_aod_2d[mask]))
print('ATLID-AERONET mean=',np.nanmean(aod_atlid[mask]-grid_aod_2d[mask]))
print('AERONET,ATLID NMB=',statistics.normalized_mean_bias(aod_atlid[mask],grid_aod_2d[mask]))
print('RMSE=',np.sqrt(np.nanmean((grid_aod_2d[mask]-aod_atlid[mask])**2)))


r, p_value = pearsonr(grid_aod_2d[mask],aod_atlid[mask])
print('Pearson r=',r,'p-value=',p_value)


print('aod_atlid.shape',aod_atlid.shape)
lon,lat = np.meshgrid(lon_centers,lat_centers)
print('West of Africa')
mask = (lat>0) & (lat<22) & (lon>-35) & (lon<-14)
a_atlid,a_aeronet = np.where(mask,aod_atlid,np.nan),np.where(mask,grid_aod_2d,np.nan)
print('West of Africa diffAOD (ATLID-AERONET)=',np.nanmean(a_atlid-a_aeronet))
print('North of Africa')
mask = (lat>10) & (lat<30) & (lon>-15) & (lon<32)
a_atlid,a_aeronet = np.where(mask,aod_atlid,np.nan),np.where(mask,grid_aod_2d,np.nan)
print('North of Africa diffAOD (ATLID-AERONET)=',np.nanmean(a_atlid-a_aeronet))
print('East China')
mask = (lat>21) & (lat<38) & (lon>110) & (lon<122)
a_atlid,a_aeronet = np.where(mask,aod_atlid,np.nan),np.where(mask,grid_aod_2d,np.nan)
print('East China diffAOD (ATLID-AERONET)=',np.nanmean(a_atlid-a_aeronet))
print('Thailand, Cambodia, Laos, Vietnam')
mask = (lat>9) & (lat<21) & (lon>93) & (lon<110)
a_atlid,a_aeronet = np.where(mask,aod_atlid,np.nan),np.where(mask,grid_aod_2d,np.nan)
print('Thailand, Cambodia, Laos, Vietnam diffAOD (ATLID-AERONET)=',np.nanmean(a_atlid-a_aeronet))



print('*********Differences between AERONET & CAMS*********')
mask = ~np.isnan(grid_aod_2d) & ~np.isnan(aod_cams)
print('CAMS mean=',np.nanmean(aod_cams[mask]))
print('AERONET mean=',np.nanmean(grid_aod_2d[mask]))
print('CAMS-AERONET mean=',np.nanmean(aod_cams[mask]-grid_aod_2d[mask]))
print('AERONET,CAMS NMB=',statistics.normalized_mean_bias(aod_cams[mask],grid_aod_2d[mask]))
print('RMSE=',np.sqrt(np.nanmean((grid_aod_2d[mask]-aod_cams[mask])**2)))


r, p_value = pearsonr(grid_aod_2d[mask],aod_cams[mask])
print('Pearson r=',r,'p-value=',p_value)


print('aod_cams.shape',aod_cams.shape)
lon,lat = np.meshgrid(lon_centers,lat_centers)
print('West of Africa')
mask = (lat>0) & (lat<22) & (lon>-35) & (lon<-14)
a_cams,a_aeronet = np.where(mask,aod_cams,np.nan),np.where(mask,grid_aod_2d,np.nan)
print('West of Africa diffAOD (CAMS-AERONET)=',np.nanmean(a_cams-a_aeronet))
print('North of Africa')
mask = (lat>10) & (lat<30) & (lon>-15) & (lon<32)
a_cams,a_aeronet = np.where(mask,aod_cams,np.nan),np.where(mask,grid_aod_2d,np.nan)
print('North of Africa diffAOD (CAMS-AERONET)=',np.nanmean(a_cams-a_aeronet))
print('East China')
mask = (lat>21) & (lat<38) & (lon>110) & (lon<122)
a_cams,a_aeronet = np.where(mask,aod_cams,np.nan),np.where(mask,grid_aod_2d,np.nan)
print('East China diffAOD (CAMS-AERONET)=',np.nanmean(a_cams-a_aeronet))
print('Thailand, Cambodia, Laos, Vietnam')
mask = (lat>9) & (lat<21) & (lon>93) & (lon<110)
a_cams,a_aeronet = np.where(mask,aod_cams,np.nan),np.where(mask,grid_aod_2d,np.nan)
print('Thailand, Cambodia, Laos, Vietnam diffAOD (CAMS-AERONET)=',np.nanmean(a_cams-a_aeronet))
