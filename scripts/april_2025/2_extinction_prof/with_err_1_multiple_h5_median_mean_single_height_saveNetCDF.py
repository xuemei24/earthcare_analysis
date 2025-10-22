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
import sys
import os

script_path = '/home/nld6854/earthcare_scripts/scripts/april_2025'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), script_path)))

from ectools import ecio
from ectools import ecplot as ecplt
from ectools import colormaps
from plotting_tools import read_h5

#simple_classification
which_aerosol='total'
# List of file paths you want to read concurrently
file_paths = glob.glob('/net/pc190625/nobackup_1/users/wangxu/earthcare_data/april_2025/EBD/*.h5')
file_paths.sort()

# Use ProcessPoolExecutor to read files concurrently
results = []
with concurrent.futures.ProcessPoolExecutor() as executor:
    for result in executor.map(read_h5.get_ext_col, file_paths):
        if result is not None:  # Filter out None results from failed reads
            results.append(result[:6])

print('Reading all EBD files finished')
# Extract AOD, lat, lon from results
all_extinction_coe, all_error, all_lat, all_lon, all_height = [], [], [], [],[]
for extinction_coe, err, lat, lon, time,height in results:
    all_extinction_coe.append(extinction_coe)
    all_error.append(err)
    all_lat.append(lat)
    all_lon.append(lon)

print('Appending all parameters finished')
# Find the maximum height dimension
max_height = max(a.shape[0] for a in all_lat)

#all_extinction = [np.array(a) for a in all_extinction_coe]
#all_lat = [np.array(a) for a in all_lat]
#all_lon = [np.array(a) for a in all_lon]
#all_height = [np.array(a) for a in all_height]

# Function to pad arrays with NaNs
def pad_array(arr, target_shape):
    pad_width = target_shape - arr.shape[0]
    if pad_width > 0:
        return np.pad(arr, (0, pad_width), constant_values=np.nan)
    return arr

all_lat_padded = [pad_array(a, max_height) for a in all_lat]
all_lon_padded = [pad_array(a, max_height) for a in all_lon]
print('Coordinates padding finished')

all_lat = np.concatenate(all_lat_padded, axis=0)
all_lon = np.concatenate(all_lon_padded, axis=0)
print('Concatenating coordinates finished')

lat_bins = np.arange(-90.1, 90.1, 2)
lon_bins = np.arange(-180.1, 180.1, 2)
 
max_cols = max(arr.shape[1] for arr in all_extinction_coe)
# Pad arrays with 241 columns to 242
all_extinction_coe_padded = [np.pad(arr, ((0, 0), (0, max_cols - arr.shape[1])), mode='constant')
    if arr.shape[1] < max_cols else arr
    for arr in all_extinction_coe]

all_error_padded = [np.pad(arr, ((0, 0), (0, max_cols - arr.shape[1])), mode='constant')
    if arr.shape[1] < max_cols else arr
    for arr in all_error]

regridded_data = np.zeros((len(lat_bins)-1,len(lon_bins)-1, all_extinction_coe[0].shape[1]))
regridded_err  = np.zeros((len(lat_bins)-1,len(lon_bins)-1, all_extinction_coe[0].shape[1]))
print(regridded_data.shape)

# Replace NaN with zeros (or another value if necessary)
#stat = np.nan_to_num(stat)
lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

clons,clats = np.meshgrid(lon_centers,lat_centers)

for i in range(all_extinction_coe[0].shape[1]):

    all_extinction1 = [all_extinction_ij[:,i] for all_extinction_ij in all_extinction_coe_padded]
    all_extinction_padded = [pad_array(a, max_height) for a in all_extinction1]
    all_extinction = np.concatenate(all_extinction_padded, axis=0)

   
    # Compute 2D histogram for the mean wind
    stat, x_edge, y_edge, _ = binned_statistic_2d(
        all_lat, all_lon, all_extinction, statistic='mean', bins=[lat_bins, lon_bins])
    
    nan_percentage = np.isnan(stat).sum() / stat.size * 100
    #print(f"Percentage of NaNs in regridded data: {nan_percentage:.2f}%")
    
    regridded_data[:,:,i] = stat

for i in range(all_error[0].shape[1]):

    all_error1 = [all_error_ij[:,i] for all_error_ij in all_error_padded]
    all_error2_padded = [pad_array(a, max_height) for a in all_error1]
    all_error3 = np.concatenate(all_error2_padded, axis=0)


    # Compute 2D histogram for the mean wind
    stat, x_edge, y_edge, _ = binned_statistic_2d(
        all_lat, all_lon, all_error3, statistic='mean', bins=[lat_bins, lon_bins])

    nan_percentage = np.isnan(stat).sum() / stat.size * 100
    #print(f"Percentage of NaNs in regridded data: {nan_percentage:.2f}%")

    regridded_err[:,:,i] = stat

print('Regridding finished')

regridded_data_xr = xr.Dataset(
    {
        "extinction_coefficient": (["latitude", "longitude", "height"], regridded_data),
        "error": (["latitude", "longitude", "height"], regridded_err)
    },
    coords={
        "latitude": lat_centers,
        "longitude": lon_centers,
        "height": height[0,:]
    }
)
'''
print(regridded_data.shape)
regridded_data_xr = xr.DataArray(
    regridded_data,
    coords=[lat_centers, lon_centers,np.arange(max_cols)],
    dims=['latitude', 'longitude','height'],
    name='extinction_coefficient' # Assign the variable name here
)
'''
# Save the regridded data to a NetCDF file
output_filename = "regridded_satellite_"+which_aerosol+"_extinction_coe_2deg_mean_single_alt_april_2025.nc"
regridded_data_xr.to_netcdf(output_filename)









regridded_data = np.zeros((clons.shape[0],clons.shape[1], all_extinction_coe[0].shape[1]))

for i in range(all_extinction_coe[0].shape[1]):
    all_extinction1 = [all_extinction_ij[:,i] for all_extinction_ij in all_extinction_coe_padded]
    all_extinction_padded = [pad_array(a, max_height) for a in all_extinction1]
    all_extinction = np.concatenate(all_extinction_padded, axis=0)

    mask = ~np.isnan(all_extinction)
    all_extinction = all_extinction[mask]
    all_lat2 = all_lat[mask]
    all_lon2 = all_lon[mask]

    # Compute 2D histogram for the mean wind
    stat, x_edge, y_edge, _ = binned_statistic_2d(
        all_lat2, all_lon2, all_extinction, statistic='mean', bins=[lat_bins, lon_bins])

    # Replace NaN with zeros (or another value if necessary)
    #stat = np.nan_to_num(stat)

    nan_percentage = np.isnan(stat).sum() / stat.size * 100
    #print(f"Percentage of NaNs in regridded data: {nan_percentage:.2f}%")

    regridded_data[:,:,i] = stat

for i in range(all_error[0].shape[1]):

    all_error1 = [all_error_ij[:,i] for all_error_ij in all_error_padded]
    all_error2_padded = [pad_array(a, max_height) for a in all_error1]
    all_error3 = np.concatenate(all_error2_padded, axis=0)

    mask = ~np.isnan(all_error3)
    all_error3 = all_error3[mask]
    all_lat2 = all_lat[mask]
    all_lon2 = all_lon[mask]

    # Compute 2D histogram for the mean wind
    stat, x_edge, y_edge, _ = binned_statistic_2d(
        all_lat, all_lon, all_error3, statistic='mean', bins=[lat_bins, lon_bins])

    nan_percentage = np.isnan(stat).sum() / stat.size * 100 
    #print(f"Percentage of NaNs in regridded data: {nan_percentage:.2f}%")

    regridded_err[:,:,i] = stat


print('Regridding finished')
regridded_data_xr = xr.Dataset(
    {
        "extinction_coefficient": (["latitude", "longitude", "height"], regridded_data),
        "error": (["latitude", "longitude", "height"], regridded_err)
    },
    coords={
        "latitude": lat_centers,
        "longitude": lon_centers,
        "height": height[0,:]
    }
)

# Save the regridded data to a NetCDF file
output_filename = "regridded_satellite_"+which_aerosol+"_extinction_coe_2deg_masknan_mean_single_alt_april_2025.nc"
regridded_data_xr.to_netcdf(output_filename)


sys.exit()

im=ax.scatter(lon,lat,c=np.nansum(data,axis=1),s=0.1,cmap='jet',transform=ccrs.platecarree())
gl = ax.gridlines(crs=ccrs.platecarree(central_longitude=0), draw_labels=true,
             linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = false
gl.ylabels_right = false
gl.xlines = false

ax.coastlines(resolution='110m')
ax.gridlines()
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

plt.tight_layout()
fig.savefig('orbit_202412'+str(path[-2:])+'.jpg',bbox_inches='tight')
bar = plt.colorbar(im, orientation='horizontal',ax=ax,shrink=0.6, pad=0.1)
bar.ax.set_xlabel('particle optical depth (unfiltered) / -',fontsize=15)
bar.ax.tick_params(labelsize=15)
fig.savefig('od_orbits_'+str(path[-2:])+'.jpg',bbox_inches='tight')


