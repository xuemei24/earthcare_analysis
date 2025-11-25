import h5py
import concurrent.futures
import glob
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
#from pyresample import geometry, kd_tree
from scipy.stats import binned_statistic_2d
import sys
import os

script_path = '/home/nld6854/earthcare_scripts/scripts/april_2025'
sys.path.append(script_path)

from ectools import ecio
from ectools import ecplot as ecplt
from ectools import colormaps
from plotting_tools import read_h5

def get_ext_wrapper(args):
    return read_h5.get_ext(*args)

month = 'february'
#simple_classification
which_aerosol='total'

year = '2024' if month == 'december' else '2025'
mean_or_std = 'mean'
# List of file paths you want to read concurrently
cams_dir = '/scratch/nld6854/earthcare/cams_data/'+month+'_'+year+'/'
file_paths = glob.glob('/scratch/nld6854/earthcare/earthcare_data/'+month+'_'+year+'/EBD/*.h5')
file_paths.sort()

args = [(fp, np.array([10,11,12,13,14,15,25,26,27])) for fp in file_paths]

# Use ProcessPoolExecutor to read files concurrently
results = []
with concurrent.futures.ProcessPoolExecutor() as executor:
    for result in executor.map(get_ext_wrapper, args):
        if result is not None:  # Filter out None results from failed reads
            results.append(result[:5])

print('Reading all EBD files finished')

# Unpack lists
ext_list = []
lat_list = []
lon_list = []
height_list = []

for ext, lat, lon,time, h in results:
    ext_list.append(np.asarray(ext))
    lat_list.append(np.asarray(lat))
    lon_list.append(np.asarray(lon))
    height_list.append(np.asarray(h))


# ============================================================
# HEIGHT CONSISTENCY CHECK
# ============================================================

# ============================================================
# INTERPOLATE ALL PROFILES TO A COMMON HEIGHT GRID
# ============================================================

# Use the height levels of the first file as target grid
target_h = height_list[0][0,:]
n_height = target_h.shape[0]

print("Using target height grid from first file:", target_h.shape)

ext_list_interp = []
lat_list_interp = []
lon_list_interp = []

from scipy.interpolate import interp1d

for idx, (ext, lat, lon, time, h) in enumerate(results):

    # ext: (n_obs, n_height_i)
    # h:   (n_obs, n_height_i)   <-- each profile has its own height grid

    n_obs, _ = ext.shape
    ext_interp = np.full((n_obs, n_height), np.nan)   # fill with NaN by default

    for i in range(n_obs):

        # Profile-specific height vector (1D)
        h_i = h[i, :]
        ext_i = ext[i, :]

        # Remove NaNs from h_i (can happen after cloud masking)
        valid = np.isfinite(h_i) & np.isfinite(ext_i)
        if np.sum(valid) < 2:
            # Not enough points to interpolate
            continue

        h_valid = h_i[valid]
        ext_valid = ext_i[valid]

        # Skip profiles that are completely outside the target range
        if (target_h.min() > h_valid.max()) or (target_h.max() < h_valid.min()):
            continue

        # Build interpolation function WITHOUT extrapolation
        f = interp1d(
            h_valid,
            ext_valid,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan   # <--- fill NaN outside valid range
        )

        # Interpolate this profile to the common height grid
        ext_interp[i, :] = f(target_h)

    ext_list_interp.append(ext_interp)
    lat_list_interp.append(lat)
    lon_list_interp.append(lon)

    print(f"Interpolated file {idx}: {ext_interp.shape}")

# NOW overwrite original lists with interpolated versions
ext_list = ext_list_interp
lat_list = lat_list_interp
lon_list = lon_list_interp
height = target_h

print("All CAMS extinction profiles are now interpolated to a common height grid.")

# ============================================================
# TARGET GRID
# ============================================================

reso = 2.0

lat_bins = np.arange(-90., 90. + reso, reso)
lon_bins = np.arange(-180., 180. + reso, reso)

lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
lon_centers = 0.5 * (lon_bins[:-1] + lon_bins[1:])

n_lat = len(lat_centers)
n_lon = len(lon_centers)

print(f"Regridded grid shape: {n_lat} × {n_lon} × {n_height}")


# ============================================================
# REGRID — VERSION 1 (KEEP NAN)
# ============================================================

regridded_data = np.full((n_lat, n_lon, n_height), np.nan)

for k in range(n_height):
    ext_k = []
    lat_k = []
    lon_k = []

    for ext, lat, lon, _,_ in results:
        if ext.shape[1] <= k:
            continue

        ext_k.append(ext[:, k])
        lat_k.append(lat)
        lon_k.append(lon)

    all_ext = np.concatenate(ext_k)
    all_lat = np.concatenate(lat_k)
    all_lon = np.concatenate(lon_k)

    stat, _, _, _ = binned_statistic_2d(
        all_lat, all_lon, all_ext,
        statistic=mean_or_std,
        bins=[lat_bins, lon_bins]
    )

    regridded_data[:, :, k] = stat

print("Finished regridding: Version WITHOUT masknan")


# ============================================================
# REGRID — VERSION 2 (REMOVE NAN BEFORE BINNING)
# ============================================================

regridded_data_masknan = np.full((n_lat, n_lon, n_height), np.nan)

for k in range(n_height):
    ext_k = []
    lat_k = []
    lon_k = []

    for ext, lat, lon, _,_ in results:
        if ext.shape[1] <= k:
            continue

        # Remove NaNs BEFORE binning
        mask = np.isfinite(ext[:, k])
        if not np.any(mask):
            continue

        ext_k.append(ext[:, k][mask])
        lat_k.append(lat[mask])
        lon_k.append(lon[mask])

    if len(ext_k) == 0:
        continue

    all_ext = np.concatenate(ext_k)
    all_lat = np.concatenate(lat_k)
    all_lon = np.concatenate(lon_k)

    stat, _, _, _ = binned_statistic_2d(
        all_lat, all_lon, all_ext,
        statistic=mean_or_std,
        bins=[lat_bins, lon_bins]
    )

    regridded_data_masknan[:, :, k] = stat

print("Finished regridding: Version WITH masknan")


# ============================================================
# SAVE BOTH FILES
# ============================================================

outdir = f"/scratch/nld6854/earthcare/cams_data/{month}_{year}"
os.makedirs(outdir, exist_ok=True)

outfile_normal = (
    f"{outdir}/regridded_satellite_{which_aerosol}"
    f"_extinction_coe_2deg_{mean_or_std}_single_alt_{month}_{year}_snr_gr_2.nc"
)

outfile_masknan = (
    f"{outdir}/regridded_satellite_{which_aerosol}"
    f"_extinction_coe_2deg_masknan_{mean_or_std}_single_alt_{month}_{year}_snr_gr_2.nc"
)

# Save version 1 (regular)
xr.Dataset(
    {
        "extinction_coefficient": (["latitude", "longitude", "height"], regridded_data)
    },
    coords={
        "latitude": lat_centers,
        "longitude": lon_centers,
        "height": height
    }
).to_netcdf(outfile_normal)

# Save version 2 (masknan)
xr.Dataset(
    {
        "extinction_coefficient": (["latitude", "longitude", "height"], regridded_data_masknan)
    },
    coords={
        "latitude": lat_centers,
        "longitude": lon_centers,
        "height": height
    }
).to_netcdf(outfile_masknan)

print("Saved BOTH files:")
print(" →", outfile_normal)
print(" →", outfile_masknan)

