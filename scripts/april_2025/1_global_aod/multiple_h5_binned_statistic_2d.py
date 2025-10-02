import h5py
import concurrent.futures
import glob
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from scipy.stats import binned_statistic_2d

import sys
import os

script_path = '/usr/people/wangxu/Desktop/earthcare_scripts/scripts/april_2025/'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), script_path)))

from ectools import ecio
from ectools import ecplot as ecplt
from ectools import colormaps
from plotting_tools import read_h5

def get_ssa_aod_wrapper(args):
    return read_h5.get_aod(*args)

month = 'may'
# List of file paths you want to read concurrently
file_paths = glob.glob('/net/pc190625/nobackup_1/users/wangxu/earthcare_data/'+month+'_2025/EBD/*.h5')
file_paths.sort()

args = [(fp, np.array([10,11,12,13,14,15,25,26,27])) for fp in file_paths] # 11 = SSA

# Use ProcessPoolExecutor to read files concurrently
results = []
with concurrent.futures.ProcessPoolExecutor() as executor:
    for result in executor.map(get_ssa_aod_wrapper, args):
#    for result in executor.map(read_h5.get_aod, file_paths):
        if result is not None:  # Filter out None results from failed reads
            results.append(result)

# Extract AOD, lat, lon from results
all_aod, all_lat, all_lon = [], [], []
i=0
for aod, lat, lon, t in results:
    all_aod.append(aod)
    all_lat.append(lat)
    all_lon.append(lon)
    print(i)
    i=i+1

# Ensure all_aod is a list of NumPy arrays
all_aod = [np.array(a) for a in all_aod]  
all_lat = [np.array(a) for a in all_lat]
all_lon = [np.array(a) for a in all_lon]

print(all_aod[0].shape)
# Find the maximum height dimension
max_height = max(a.shape[0] for a in all_aod)

# Function to pad arrays with NaNs
def pad_array(arr, target_shape):
    pad_width = target_shape - arr.shape[0]
    if pad_width > 0:
        return np.pad(arr, (0, pad_width), constant_values=np.nan)
    return arr

# Apply padding to all arrays
all_aod_padded = [pad_array(a, max_height) for a in all_aod]
all_lat_padded = [pad_array(a, max_height) for a in all_lat]
all_lon_padded = [pad_array(a, max_height) for a in all_lon]

# Convert lists to NumPy arrays
all_aod = np.concatenate(all_aod_padded, axis=0)
all_lat = np.concatenate(all_lat_padded, axis=0)
all_lon = np.concatenate(all_lon_padded, axis=0)

print(all_lat.shape,all_lon.shape,all_aod.shape)
np.savetxt('2025_'+month+'_aod.txt',np.array([all_lat,all_lon,all_aod]).transpose(),header='latitude,longitude,aod355nm',delimiter=',')

#lines below are not relevant
lat_bins = np.arange(-90.1, 90.1, 2)
lon_bins = np.arange(-180.1, 180.1, 2)

# Compute 2D histogram for the mean wind
stat, x_edge, y_edge, _ = binned_statistic_2d(
    all_lat, all_lon, all_aod, statistic='median', bins=[lat_bins, lon_bins])

# Replace NaN with zeros (or another value if necessary)
stat = np.nan_to_num(stat)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_global()
levels=np.linspace(0, 1, 150)

lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
mesh = ax.contourf(lon_centers, lat_centers, stat, levels=levels, cmap='viridis', transform=ccrs.PlateCarree())
print('global mean stat',np.mean(stat))

# Add land boundaries
ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='none')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
cbar = plt.colorbar(mesh, orientation='horizontal', pad=0.1)
cbar.set_label('check')

# Set 10 equally spaced ticks
ticks = np.linspace(levels.min(), levels.max(), 7)
cbar.set_ticks(ticks)

title='AOD_'+month+'_2025'
plt.title(f'Global Average {title} at 2 deg Resolution')

xlim = [-180,180]
ylim = [-90,90]
plt.xlim(xlim)
plt.ylim(ylim)

plt.savefig(f'globe_{title}.png', bbox_inches='tight')
