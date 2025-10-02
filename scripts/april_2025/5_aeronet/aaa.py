import pandas as pd
import numpy as np
import xarray as xr
import sys
import os
from scipy.stats import binned_statistic_2d,pearsonr
script_path = '/usr/people/wangxu/Desktop/earthcare_scripts/scripts/april_2025/'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), script_path)))
from plotting_tools import statistics

def get_AOD(aod_old, wave_old, wave_new, angstrom):
    print('aod_old>0',aod_old[aod_old>0], 'wave_old',wave_old, 'wave_new',wave_new, 'angstrom>0',angstrom[angstrom>0])
    return ((wave_new / wave_old) ** (-angstrom)) * aod_old

#Use this script after ATLID AOD has been processed
month = '07'
month2 = 'July'
month3 = 'july'

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors

from scipy.spatial import cKDTree

bounds = np.linspace(-0.5,0.5,11)
#-ATLID-------------------------------------------------------------------------
file_dir = '/net/pc190625/nobackup_1/users/wangxu/cams_data/'
print('Month=',month3)
df = pd.read_csv(file_dir+"2025_"+month3+"_atlid_aeronet_co-located_100km_10atlid_per_aeronet.csv", delimiter=",")

atlid_aod = df['atlid_aod'].values
atlid_lat = df['atlid_lat'].values
atlid_lon = df['atlid_lon'].values

aeronet_aod = df['aeronet_aod'].values
aeronet_lat = df['aeronet_lat'].values
aeronet_lon = df['aeronet_lon'].values

mask = ~np.isnan(atlid_aod) & ~np.isnan(aeronet_aod)
atlid_aod = atlid_aod[mask]
atlid_lat = atlid_lat[mask]
atlid_lon = atlid_lon[mask]

aeronet_aod = aeronet_aod[mask]
aeronet_lat = aeronet_lat[mask]
aeronet_lon = aeronet_lon[mask]

lat,lon,aer_aod,atl_aod = [],[],[],[]
for ij in np.unique(aeronet_lat):
    print(aeronet_lon[aeronet_lat==ij])
    lon.append(aeronet_lon[aeronet_lat==ij][0])
    lat.append(ij)
    aer_aod.append(np.nanmean(aeronet_aod[aeronet_lat==ij]))
    atl_aod.append(np.nanmean(atlid_aod[aeronet_lat==ij]))

print(lat,lon,aer_aod,atl_aod)

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
bar.ax.set_ylabel('ATLID Particle optical depth / -',fontsize=15)
bar.ax.tick_params(labelsize=15)

ax2.set_title('ATLID AOD '+month2,fontsize=15)


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

ax3.set_title('ATLID-AERONET AOD '+month2,fontsize=15)


plt.tight_layout()
fig.savefig('global_AERONET_ATLID_aod_'+month2+'_2025_100km_10atlid_per_aeronet.jpg',bbox_inches='tight')

atl_aod = np.array(atl_aod)
aer_aod = np.array(aer_aod)
print('*********Differences between AERONET & ATLID*********')
mask = ~np.isnan(aer_aod) & ~np.isnan(atl_aod)
print('ATLID mean=',np.nanmean(atl_aod))
print('AERONET mean=',np.nanmean(aer_aod[mask]))
print('ATLID-AERONET mean=',np.nanmean(atl_aod[mask]-aer_aod[mask]))
print('AERONET,ATLID NMB=',statistics.normalized_mean_bias(atl_aod[mask],aer_aod[mask]))
print('RMSE=',np.sqrt(np.nanmean((aer_aod[mask]-atl_aod[mask])**2)))


r, p_value = pearsonr(aer_aod[mask],atl_aod[mask])
print('Pearson r=',r,'p-value=',p_value)

print('atl_aod.shape',atl_aod.shape)
lonpr,latpr = np.meshgrid(lon,lat)
mask = (latpr>0) & (latpr<22) & (lonpr>-35) & (lonpr<-14)
a_cams,a_aeronet = np.where(mask,atl_aod,np.nan),np.where(mask,aer_aod,np.nan)
print('West of Africa diffAOD (ATLID-AERONET)=',np.nanmean(a_cams-a_aeronet))
mask = (latpr>10) & (latpr<30) & (lonpr>-15) & (lonpr<32)
a_cams,a_aeronet = np.where(mask,atl_aod,np.nan),np.where(mask,aer_aod,np.nan)
print('North of Africa diffAOD (ATLID-AERONET)=',np.nanmean(a_cams-a_aeronet))
mask = (latpr>21) & (latpr<38) & (lonpr>110) & (lonpr<122)
a_cams,a_aeronet = np.where(mask,atl_aod,np.nan),np.where(mask,aer_aod,np.nan)
print('East China diffAOD (ATLID-AERONET)=',np.nanmean(a_cams-a_aeronet))
mask = (latpr>9) & (latpr<21) & (lonpr>93) & (lonpr<110)
a_cams,a_aeronet = np.where(mask,atl_aod,np.nan),np.where(mask,aer_aod,np.nan)
print('Thailand, Cambodia, Laos, Vietnam diffAOD (ATLID-AERONET)=',np.nanmean(a_cams-a_aeronet))

