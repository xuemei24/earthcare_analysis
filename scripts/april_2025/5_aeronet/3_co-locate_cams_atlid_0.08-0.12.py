import pandas as pd
import numpy as np
import xarray as xr
import sys
import os
from scipy.stats import binned_statistic_2d,pearsonr
script_path = '/home/nld6854/earthcare_scripts/scripts/april_2025'
sys.path.append(script_path)
from plotting_tools import statistics

def get_AOD(aod_old, wave_old, wave_new, angstrom):
    print('aod_old>0',aod_old[aod_old>0], 'wave_old',wave_old, 'wave_new',wave_new, 'angstrom>0',angstrom[angstrom>0])
    return ((wave_new / wave_old) ** (-angstrom)) * aod_old

#Use this script after ATLID AOD has been processed
# Define year and all months to process
year = '2025'
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
month_names_lower = [m.lower() for m in month_names]

range1 = 0.12
range2 = 0.25

aeronet_path = '/scratch/nld6854/earthcare/aeronet/'
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors

# Plot AOD values
#-ATLID-------------------------------------------------------------------------
file_dir = '/scratch/nld6854/earthcare/cams_data/'

# Load ATLID data for all months
atlid_dfs = []
for month, month3 in zip(months, month_names_lower):
    try:
        df_atlid = pd.read_csv(
            f'{file_dir}{month3}_{year}/{year}_{month3}_atlid_aeronet_co-located_100km_10atlid_per_aeronet_2nd_method.csv',
            delimiter=","
        )
        atlid_dfs.append(df_atlid)
    except FileNotFoundError:
        print(f"Warning: ATLID file not found for {month3} {year}")
        continue

# Combine and average ATLID data
df_atlid_all = pd.concat(atlid_dfs, ignore_index=True)
df_atlid_yearly = (
    df_atlid_all.groupby(['lat', 'lon'])
    .agg({'co_located_atlid': np.nanmean, 'aeronet_aod': np.nanmean})
    .reset_index())

atlid_aod = df_atlid_yearly['co_located_atlid'].values

aeronet_aod = df_atlid_yearly['aeronet_aod'].values
aeronet_lat = df_atlid_yearly['lat'].values
aeronet_lon = df_atlid_yearly['lon'].values

mask = ~np.isnan(atlid_aod) & ~np.isnan(aeronet_aod)
atlid_aod = atlid_aod[mask]

aeronet_aod = aeronet_aod[mask]
aeronet_lat = aeronet_lat[mask]
aeronet_lon = aeronet_lon[mask]

lat,lon,aer_aod,atl_aod = [],[],[],[]
for ij in np.unique(aeronet_lat):
    lon.append(aeronet_lon[aeronet_lat==ij][0])
    lat.append(ij)
    aer_aod.append(np.nanmean(aeronet_aod[aeronet_lat==ij]))
    atl_aod.append(np.nanmean(atlid_aod[aeronet_lat==ij]))

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

aer_aod = np.array(aer_aod)
atl_aod = np.array(atl_aod)
mask = (aer_aod >= range1) & (aer_aod<=range2)
aer_aod = np.where(mask,aer_aod,np.nan)
mask = (atl_aod >= range1) & (atl_aod<=range2)
atl_aod = np.where(mask,atl_aod,np.nan)

# Plot AOD values
scatter = ax1.scatter(
    lon,
    lat,
    c=aer_aod,
    s=60,
    edgecolor='k',
    cmap='plasma',
    norm=colors.LogNorm(vmin=range1, vmax=range2),
    transform=ccrs.PlateCarree()
)

bar = plt.colorbar(scatter, orientation='vertical',ax=ax1,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('AERONET Particle optical depth / -',fontsize=15)
bar.ax.tick_params(labelsize=15)

ax1.set_title('AERONET AOD Dec 2024 - Nov 2025',fontsize=15)


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
    norm=colors.LogNorm(vmin=range1, vmax=range2),
    transform=ccrs.PlateCarree()
)

bar = plt.colorbar(scatter, orientation='vertical',ax=ax2,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('ATLID Particle optical depth / -',fontsize=15)
bar.ax.tick_params(labelsize=15)

ax2.set_title('ATLID AOD Dec 2024 - Nov 2025',fontsize=15)


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

ax3.set_title('ATLID-AERONET AOD Dec 2024 - Nov 2025',fontsize=15)


plt.tight_layout()
fig.savefig('global_AERONET_ATLID_aod_122024-112025_100km_'+str(range1)+'-'+str(range2)+'.jpg',bbox_inches='tight')


