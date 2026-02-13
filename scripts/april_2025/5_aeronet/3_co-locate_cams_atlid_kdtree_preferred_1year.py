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
        aod340 = df['AOD_340nm']
        angstrom = df['340-440_Angstrom_Exponent']
        aod355 = get_AOD(aod340, 340, 355, angstrom)
        df['AOD_355nm'] = aod355
        
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
    .agg({'AOD_355nm': np.nanmean})
    .reset_index()
)

df_final = yearly_avg
print(f"Number of sites with yearly data: {len(df_final)}")
aod = df_final['AOD_355nm']

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
    c=df_final['AOD_355nm'],
    s=60,
    edgecolor='k',
    cmap='plasma',
    norm=colors.LogNorm(vmin=1e-2, vmax=1),
    transform=ccrs.PlateCarree()
)

bar = plt.colorbar(scatter, orientation='vertical',ax=ax1,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('AERONET Particle optical depth / -',fontsize=15)
bar.ax.tick_params(labelsize=15)

ax1.set_title('AERONET AOD Dec 2024 - Nov 2025',fontsize=15)


#-CAMS-------------------------------------------------------------------------
def collapse_lwc(cloud,data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            mask = (cloud[i,j]>=0.0001).any(axis=0)
            data[i,j,:,:] = np.where(mask, np.nan, data[i,j,:,:])
    return data

cams_aod_list = []
for month3 in month_names_lower:
    try:
        fcams = f'/scratch/nld6854/earthcare/cams_data/{month3}_{year}/total_aerosol_optical_depth_355nm_{month3}_{year}.nc'
        ds = xr.open_dataset(fcams)
        cams_aod_r = ds['aod355'].values[::3]
        
        # Load LWC for cloud masking
        dslwc = xr.open_dataset(f'/scratch/nld6854/earthcare/cams_data/{month3}_{year}/specific_cloud_liquid_water_content_{month3}_{year}.nc')
        lwc = dslwc['clwc'].values[:, :]
        
        lwc_cams = collapse_lwc(lwc, lwc[:, :, 0, :, :])
        aod_cams_month = np.where(lwc_cams >= 0.0001, np.nan, cams_aod_r)
        aod_cams_month = np.nanmean(np.nanmean(aod_cams_month, axis=0), axis=0)
        
        cams_aod_list.append(aod_cams_month)
    except FileNotFoundError:
        print(f"Warning: CAMS file not found for {month3} {year}")
        continue

# Average across all months
aod_cams = np.nanmean(cams_aod_list, axis=0)
print('Yearly CAMS AOD shape:', aod_cams.shape)

cams_ds = xr.Dataset({"cams_aod": (["latitude", "longitude"], aod_cams)},
                     coords={"latitude": ds['latitude'], "longitude": ds['longitude']})

from scipy.spatial import cKDTree

def latlon_to_xyz(lat, lon):
    lat = np.radians(lat); lon = np.radians(lon)
    x = np.cos(lat)*np.cos(lon); y = np.cos(lat)*np.sin(lon); z = np.sin(lat)
    return np.vstack([x,y,z]).T

def get_nearest_cams_points(df_sites, cams):
    # Build tree on CAMS grid
    cams_lat = cams['latitude'].values; cams_lon = (((cams['longitude'].values + 180)%360)-180)
    print('cams_lon=',cams_lon)
    tree = cKDTree(latlon_to_xyz(*np.meshgrid(cams_lat, cams_lon, indexing='ij')).reshape(-1,3))
   
    # Query stations
    st_xyz = latlon_to_xyz(df_final['Site_Latitude(Degrees)'].values,
                            df_final['Site_Longitude(Degrees)'].values)
    d, idx = tree.query(st_xyz, k=1)
    ii, jj = np.unravel_index(idx, (cams_lat.size, cams_lon.size))
    return ii,jj,d

ii, jj, d = get_nearest_cams_points(df_final, cams_ds)

cams_at_stations = [cams_ds["cams_aod"].values[i,j] for i,j in zip(ii,jj)]
#cams_at_stations = cams_ds.isel(latitude=("points", ii),longitude=("points", jj))  # still a Dataset

df_final["cams_aod355"] = cams_at_stations

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
    df_final['Site_Longitude(Degrees)'],
    df_final['Site_Latitude(Degrees)'],
    c=cams_at_stations,
    s=60,
    edgecolor='k',
    cmap='plasma',
    norm=colors.LogNorm(vmin=1e-2, vmax=1),
    transform=ccrs.PlateCarree()
)

bar = plt.colorbar(scatter, orientation='vertical',ax=ax2,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('CAMS Particle optical depth / -',fontsize=15)
bar.ax.tick_params(labelsize=15)

ax2.set_title('Co-located CAMS AOD Dec 2024 - Nov 2025',fontsize=15)

#diffs--------------------------------------------------------------------------

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
bounds = np.linspace(-0.5,0.5,11)
scatter = ax3.scatter(
    df_final['Site_Longitude(Degrees)'],
    df_final['Site_Latitude(Degrees)'],
    c=cams_at_stations-df_final['AOD_355nm'],
    s=60,
    edgecolor='k',
    cmap='RdBu_r',
    norm=colors.BoundaryNorm(boundaries=bounds, ncolors=256),#colors.SymLogNorm(linthresh=0.001,vmin=-1, vmax=1),
    transform=ccrs.PlateCarree()
)

bar = plt.colorbar(scatter, orientation='vertical',ax=ax3,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('Differences in Particle optical depth / -',fontsize=15)
bar.ax.tick_params(labelsize=15)

ax3.set_title('CAMS-AERONET AOD Dec 2024 - Nov 2025',fontsize=15)


plt.tight_layout()
fig.savefig('global_AERONET_CAMS_aod_122024-112025_100km.jpg',bbox_inches='tight')

print('*********Differences between AERONET & CAMS*********')
cams_at_stations = np.array(cams_at_stations)
mask = ~np.isnan(df_final['AOD_355nm'].values) & ~np.isnan(cams_at_stations)
print('CAMS mean=',np.nanmean(cams_at_stations[mask]))
print('AERONET mean=',np.nanmean(df_final['AOD_355nm'].values[mask]))
print('CAMS-AERONET mean=',np.nanmean(cams_at_stations[mask]-df_final['AOD_355nm'].values[mask]))
print('(CAMS-AERONET)/AERONET mean=',np.nanmean((cams_at_stations[mask]-df_final['AOD_355nm'].values[mask])/df_final['AOD_355nm'].values[mask]))
print('AERONET,CAMS NMB=',statistics.normalized_mean_bias(cams_at_stations[mask],df_final['AOD_355nm'].values[mask]))
print('RMSE=',np.sqrt(np.nanmean((df_final['AOD_355nm'].values[mask]-cams_at_stations[mask])**2)))


r, p_value = pearsonr(df_final['AOD_355nm'].values[mask],cams_at_stations[mask])
print('Pearson r=',r,'p-value=',p_value)

print('cams_at_stations.shape',cams_at_stations.shape)
lonpr,latpr = np.meshgrid(df_final['Site_Longitude(Degrees)'].values,df_final['Site_Latitude(Degrees)'].values)
mask = (latpr>0) & (latpr<22) & (lonpr>-35) & (lonpr<-14)
a_cams,a_aeronet = np.where(mask,cams_at_stations,np.nan),np.where(mask,df_final['AOD_355nm'].values,np.nan)
print('West of Africa diffAOD (CAMS-AERONET)=',np.nanmean(a_cams-a_aeronet))
mask = (latpr>10) & (latpr<30) & (lonpr>-15) & (lonpr<32)
a_cams,a_aeronet = np.where(mask,cams_at_stations,np.nan),np.where(mask,df_final['AOD_355nm'].values,np.nan)
print('North of Africa diffAOD (CAMS-AERONET)=',np.nanmean(a_cams-a_aeronet))
mask = (latpr>21) & (latpr<38) & (lonpr>110) & (lonpr<122)
a_cams,a_aeronet = np.where(mask,cams_at_stations,np.nan),np.where(mask,df_final['AOD_355nm'].values,np.nan)
print('East China diffAOD (CAMS-AERONET)=',np.nanmean(a_cams-a_aeronet))
mask = (latpr>9) & (latpr<21) & (lonpr>93) & (lonpr<110)
a_cams,a_aeronet = np.where(mask,cams_at_stations,np.nan),np.where(mask,df_final['AOD_355nm'].values,np.nan)
print('Thailand, Cambodia, Laos, Vietnam diffAOD (CAMS-AERONET)=',np.nanmean(a_cams-a_aeronet))

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
    norm=colors.LogNorm(vmin=1e-2, vmax=1),
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
fig.savefig('global_AERONET_ATLID_aod_122024-112025_100km.jpg',bbox_inches='tight')

nbins = 150
binsc = np.linspace(0,np.nanmax(aer_aod.flatten()),nbins)
histc,binsc = np.histogram(aer_aod,bins=binsc,density=False)
bcc = 0.5*(binsc[1:] + binsc[:-1])

binsa = np.linspace(0,np.nanmax(atl_aod),nbins)
hista,binsa = np.histogram(atl_aod,bins=binsa,density=False)
bca = 0.5*(binsa[1:] + binsa[:-1])

a_errors = np.sqrt(hista)
c_errors = np.sqrt(histc)
#fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5),gridspec_kw={'width_ratios': [2, 1, 1]},sharey=False)
fig,ax1 = plt.subplots(1,figsize=(6,5))
ax1.plot(bcc,histc,color='blue',label='AERONET')
ax1.fill_between(bcc,histc-c_errors,histc+c_errors,color='blue',alpha=0.3)
ax1.plot(bca,hista,color='orange',label='ATLID')
ax1.fill_between(bca,hista-a_errors,hista+a_errors,color='orange',alpha=0.3)

#CAMS mode
def find_mode(counts,bin_edges):
    # Find the maximum frequency
    peaks, properties = find_peaks(counts)
    sorted_indices = np.argsort(counts[peaks])[::-1]
    nsel = 4 if len(sorted_indices) >= 4 else len(sorted_indices)
        
    top_peaks = peaks[sorted_indices[:nsel]]

    modes = [round((bin_edges[i] + bin_edges[i+1]) / 2,4) for i in top_peaks]
    print("Estimated modes from histogram:", modes)
    return modes,counts[top_peaks]

cmodes,chpeaks = find_mode(histc[:],binsc[:])
amodes,ahpeaks = find_mode(hista[:],binsa[:])

#for i,m in enumerate(cmodes):
#    ax1.scatter(m,chpeaks[i], color='blue', zorder=5, label=f'Mode ~ {m:.2f}')

#for i,m in enumerate(amodes):
#    ax1.scatter(m,ahpeaks[i], color='orange', zorder=5, label=f'Mode ~ {m:.2f}')

ax1.set_xlim(1e-3,2)
ax1.set_xscale('log')

ax1.set_xlabel('AOD',fontsize=15)
ax1.set_ylabel('Counts',fontsize=15)

ax1.tick_params(labelsize=12)
ax1.set_title('Dec 2024 - Nov 2025',fontsize=15)
ax1.legend(frameon=False,fontsize=15)

fig.savefig('histograms_AERONET_ATLID_122024-112025_co-located_filtered_uncertainty_with_modes.jpg')


atl_aod = np.array(atl_aod)
aer_aod = np.array(aer_aod)
print('*********Differences between AERONET & ATLID*********')
mask = ~np.isnan(aer_aod) & ~np.isnan(atl_aod)
print('ATLID mean=',np.nanmean(atl_aod))
print('AERONET mean=',np.nanmean(aer_aod[mask]))
print('ATLID-AERONET mean=',np.nanmean(atl_aod[mask]-aer_aod[mask]))
print('(ATLID-AERONET)/AERONET mean=',np.nanmean((atl_aod[mask]-aer_aod[mask])/aer_aod[mask]))
print('AERONET,ATLID NMB=',statistics.normalized_mean_bias(atl_aod[mask],aer_aod[mask]))
print('RMSE=',np.sqrt(np.nanmean((aer_aod[mask]-atl_aod[mask])**2)))
print('ATLID std=',np.std(atl_aod[mask]))
print('AERONET std=',np.std(aer_aod[mask]))


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


