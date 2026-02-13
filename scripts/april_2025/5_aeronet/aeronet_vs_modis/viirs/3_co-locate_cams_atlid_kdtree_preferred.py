import pandas as pd
import numpy as np
import xarray as xr
import sys
import os
from scipy.stats import binned_statistic_2d,pearsonr
script_path = '/home/nld6854/earthcare_scripts/scripts/april_2025'
sys.path.append(script_path)
from plotting_tools import statistics
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors

month = '11'
month2 = 'November'
month3 = 'november'

year = '2024' if month3 == 'december' else '2025'
print('year=',year)
print('month=',month2)

file_dir = '/scratch/nld6854/earthcare/modis/VIIRS/hdffiles/monthly_aod/'
print('Month=',month3)
df = pd.read_csv(file_dir+year+"_"+month3+"_modis_aeronet_co-located_100km_30min.csv", delimiter=",")

modis_aod = df['co_located_modis'].values

aeronet_aod = df['aeronet_aod'].values
aeronet_lat = df['lat'].values
aeronet_lon = df['lon'].values

mask = ~np.isnan(modis_aod) & ~np.isnan(aeronet_aod)
modis_aod = modis_aod[mask]

aeronet_aod = aeronet_aod[mask]
aeronet_lat = aeronet_lat[mask]
aeronet_lon = aeronet_lon[mask]

lat,lon,aer_aod,atl_aod = [],[],[],[]
for ij in np.unique(aeronet_lat):
    lon.append(aeronet_lon[aeronet_lat==ij][0])
    lat.append(ij)
    aer_aod.append(np.nanmean(aeronet_aod[aeronet_lat==ij]))
    atl_aod.append(np.nanmean(modis_aod[aeronet_lat==ij]))

bounds = np.linspace(-0.5,0.5,11)
fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(20,36),subplot_kw=dict(projection=ccrs.PlateCarree()))
#plt.figure(figsize=(6,3))
#ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax1.set_extent([-180, 180, -90, 90])#,crs=ccrs.PlateCarree(central_longitude=180))
gl = ax1.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
             linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False

ax1.coastlines(resolution='110m')#,color='white')
ax1.gridlines()
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

# Plot AOD values
scatter = ax1.scatter(
    lon,
    lat,
    c=aer_aod,
    s=60,
    edgecolor='k',
    cmap='plasma',
    norm=colors.LogNorm(vmin=1e-2, vmax=1),
    transform=ccrs.PlateCarree()
)

bar = plt.colorbar(scatter, orientation='vertical',ax=ax1,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('AERONET Particle optical depth / -',fontsize=15)
bar.ax.tick_params(labelsize=15)

ax1.set_title('AERONET AOD '+month2,fontsize=15)


ax2.set_extent([-180, 180, -90, 90])#,crs=ccrs.PlateCarree(central_longitude=180))
gl = ax2.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
             linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False

ax2.coastlines(resolution='110m')#,color='white')
ax2.gridlines()
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

# Plot AOD values
scatter = ax2.scatter(
    lon,
    lat,
    c=atl_aod,
    s=60,
    edgecolor='k',
    cmap='plasma',
    norm=colors.LogNorm(vmin=1e-2, vmax=1),
    transform=ccrs.PlateCarree()
)

bar = plt.colorbar(scatter, orientation='vertical',ax=ax2,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('VIIRS Particle optical depth / -',fontsize=15)
bar.ax.tick_params(labelsize=15)

ax2.set_title('VIIRS AOD '+month2,fontsize=15)


gl = ax3.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
             linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False

ax3.coastlines(resolution='110m')#,color='white')
ax3.gridlines()
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

# Plot AOD values
scatter = ax3.scatter(
    lon,
    lat,
    c=np.array(atl_aod)-np.array(aer_aod),
    s=60,
    edgecolor='k',
    cmap='RdBu_r',
    norm=colors.BoundaryNorm(boundaries=bounds, ncolors=256),#colors.SymLogNorm(linthresh=0.001,vmin=-1, vmax=1),
    transform=ccrs.PlateCarree()
)

bar = plt.colorbar(scatter, orientation='vertical',ax=ax3,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('Differences in Particle optical depth / -',fontsize=15)
bar.ax.tick_params(labelsize=15)

ax3.set_title('VIIRS-AERONET AOD '+month2,fontsize=15)


plt.tight_layout()
fig.savefig('global_AERONET_VIIRS_aod_'+month2+'_'+year+'_100km.jpg',bbox_inches='tight')

atl_aod = np.array(atl_aod)
aer_aod = np.array(aer_aod)
print('*********Differences between AERONET & MODIS*********')
mask = ~np.isnan(aer_aod) & ~np.isnan(atl_aod)
print('MODIS mean=',np.nanmean(atl_aod))
print('AERONET mean=',np.nanmean(aer_aod[mask]))
print('MODIS-AERONET mean=',np.nanmean(atl_aod[mask]-aer_aod[mask]))
print('(MODIS-AERONET)/AERONET mean=',np.nanmean((atl_aod[mask]-aer_aod[mask])/aer_aod[mask]))
print('AERONET,MODIS NMB=',statistics.normalized_mean_bias(atl_aod[mask],aer_aod[mask]))
print('RMSE=',np.sqrt(np.nanmean((aer_aod[mask]-atl_aod[mask])**2)))
print('MODIS std=',np.std(atl_aod[mask]))
print('AERONET std=',np.std(aer_aod[mask]))


r, p_value = pearsonr(aer_aod[mask],atl_aod[mask])
print('Pearson r=',r,'p-value=',p_value)

print('atl_aod.shape',atl_aod.shape)
lonpr,latpr = np.meshgrid(lon,lat)
mask = (latpr>0) & (latpr<22) & (lonpr>-35) & (lonpr<-14)
a_cams,a_aeronet = np.where(mask,atl_aod,np.nan),np.where(mask,aer_aod,np.nan)
print('West of Africa diffAOD (MODIS-AERONET)=',np.nanmean(a_cams-a_aeronet))
mask = (latpr>10) & (latpr<30) & (lonpr>-15) & (lonpr<32)
a_cams,a_aeronet = np.where(mask,atl_aod,np.nan),np.where(mask,aer_aod,np.nan)
print('North of Africa diffAOD (MODIS-AERONET)=',np.nanmean(a_cams-a_aeronet))
mask = (latpr>21) & (latpr<38) & (lonpr>110) & (lonpr<122)
a_cams,a_aeronet = np.where(mask,atl_aod,np.nan),np.where(mask,aer_aod,np.nan)
print('East China diffAOD (MODIS-AERONET)=',np.nanmean(a_cams-a_aeronet))
mask = (latpr>9) & (latpr<21) & (lonpr>93) & (lonpr<110)
a_cams,a_aeronet = np.where(mask,atl_aod,np.nan),np.where(mask,aer_aod,np.nan)
print('Thailand, Cambodia, Laos, Vietnam diffAOD (MODIS-AERONET)=',np.nanmean(a_cams-a_aeronet))
print('----------------------------------------------------------------------------------')
