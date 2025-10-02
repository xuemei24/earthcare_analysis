import pandas as pd
import numpy as np
import xarray as xr
from scipy.stats import binned_statistic_2d

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


#-CAMS-------------------------------------------------------------------------
fcams = '/net/pc190625/nobackup_1/users/wangxu/cams_data/total_aerosol_optical_depth_355nm_'+month3+'_2025.nc'

ds = xr.open_dataset(fcams)
cams_aod_r = ds['aod355'].values[::3]
print('cams_aod_r',cams_aod_r.shape)

#this file should be merged
dslwc = xr.open_dataset('/net/pc190625/nobackup_1/users/wangxu/cams_data/lwc/specific_cloud_liquid_water_content_'+month3+'_2025.nc')
lwc = dslwc['clwc'].values[:,:]
def collapse_lwc(cloud,data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            mask = (cloud[i,j]>=0.0001).any(axis=0)
            data[i,j,:,:] = np.where(mask, np.nan, data[i,j,:,:])
    return data

lwc_cams = collapse_lwc(lwc,lwc[:,:,0,:,:])
print('lwc_cams.shape',lwc_cams.shape)
aod_cams = np.where(lwc_cams>=0.0001,np.nan,cams_aod_r)
#aod_cams = np.expand_dims(aod_cams,axis=1)
print(aod_cams.shape)
aod_cams = np.nanmean(np.nanmean(aod_cams,axis=0),axis=0)
print('cams_aod',aod_cams.shape)

new_ds = xr.Dataset(
    {"aod_cams": (["latitude", "longitude"], aod_cams)},
    coords={
        "latitude": ds["latitude"],
        "longitude": ds["longitude"]})

df_final["Site_Longitude_0_360"] = df_final["Site_Longitude(Degrees)"] % 360

# ---- Option 1: nearest grid point ----
cams_at_stations = new_ds['aod_cams'].sel(
    latitude=xr.DataArray(df_final['Site_Longitude_0_360'], dims="points"),
    longitude=xr.DataArray(df_final['Site_Latitude(Degrees)'], dims="points"),
    method="nearest")

cams_aod_sel = cams_at_stations.values


################ KDTree #################
cams_ds = xr.Dataset({"cams_aod": (["latitude", "longitude"],aod_cams)},
         coords={"latitude": ds['latitude'],"longitude": ds['longitude']})

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

cams_aod_kdtree = cams_at_stations


################ Binned :statistics #################
lons = ds['longitude'].values
ilon = np.where(lons>180)
lons[ilon] = lons[ilon]-360
mean_or_std = 'mean'
lons,lats = np.meshgrid(lons,ds['latitude'].values)
mask = ~np.isnan(aod_cams)
aod_cams = aod_cams[mask]
all_lat = lats[mask]
all_lon = lons[mask]

reso = 0.4
lat_bins = np.arange(-90, 90+reso, reso)
lon_bins = np.arange(-180, 180+reso, reso)

# Compute 2D histogram for the mean wind
cams_aod_binned, x_edge, y_edge, _ = binned_statistic_2d(
    all_lat, all_lon, aod_cams, statistic='mean', bins=[lat_bins, lon_bins])






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
    c=cams_aod_sel-cams_aod_kdtree,
    s=60,
    edgecolor='k',
    cmap='RdBu_r',
    vmin = -1,vmax=1,
    transform=ccrs.PlateCarree()
)

bar = plt.colorbar(scatter, orientation='vertical',ax=ax1,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('CAMS_sel - CAMS_kdtree / -',fontsize=15)
bar.ax.tick_params(labelsize=15)

ax1.set_title('AERONET AOD '+month2,fontsize=15)




plt.tight_layout()
fig.savefig('global_AERONET_CAMS_aod_'+month2+'_2025_temp.jpg',bbox_inches='tight')


