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


aeronet_path = '/scratch/nld6854/earthcare/aeronet/'
# Collect all monthly data
all_dfs = []
for month, month2, month3 in zip(months, month_names, month_names_lower):
    try:
        # Adjust year for December if needed
        year_temp = '2024' if month3 == 'december' else year
        
        df = pd.read_table(aeronet_path + year_temp + month + '_all_sites_aod15_dailyAVG.txt', 
                          delimiter=',', header=[7])
        df = df.replace(-999.0, np.nan)
        
        # Calculate AOD at 355nm
        ratio = df['340-440_Angstrom_Exponent']/df['440-675_Angstrom_Exponent']
        df['angstrom_exp_ratio'] = ratio
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date(dd:mm:yyyy)'], format='%d:%m:%Y')
        df['Month'] = df['Date'].dt.to_period('M')
        
        all_dfs.append(df)
    except FileNotFoundError:
        print(f"Warning: File not found for {month2} {year_temp}")
        continue

# Combine all months
df_all = pd.concat(all_dfs, ignore_index=True)

# Calculate YEARLY average per site
yearly_avg = (
    df_all.groupby(['AERONET_Site', 'Site_Latitude(Degrees)', 'Site_Longitude(Degrees)'])
    .agg({'angstrom_exp_ratio': np.nanmean})
    .reset_index()
)

df_final = yearly_avg
print(f"Number of sites with yearly data: {len(df_final)}")

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors

# --- Load your DataFrame ---
# Make sure df_final is ready and contains:
# 'AOD_355nm', 'Site_Latitude(Degrees)', 'Site_Longitude(Degrees)', 'AERONET_Site'

# --- Create the plot ---
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
    df_final['Site_Longitude(Degrees)'],
    df_final['Site_Latitude(Degrees)'],
    c=df_final['angstrom_exp_ratio'],
    s=60,
    edgecolor='k',
    cmap='jet',
    #norm=colors.LogNorm(vmin=1e-2, vmax=1),
    transform=ccrs.PlateCarree()
)

bar = plt.colorbar(scatter, orientation='vertical',ax=ax1,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('Ratio of Angstrom Exponent (340-440)/(440-675)',fontsize=15)
bar.ax.tick_params(labelsize=15)


plt.tight_layout()
fig.savefig('global_AERONET_angstrom_ratio_122024-112025.jpg',bbox_inches='tight')


print('mean',np.nanmean(df_final['angstrom_exp_ratio']))
print('median',np.nanmedian(df_final['angstrom_exp_ratio']))
print('25%',np.nanpercentile(df_final['angstrom_exp_ratio'],25))
print('75%',np.nanpercentile(df_final['angstrom_exp_ratio'],75))
