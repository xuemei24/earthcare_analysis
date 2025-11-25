import h5py
import concurrent.futures
import glob
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
from scipy.stats import binned_statistic_2d
import sys
import os
from multiprocessing import Pool


script_path = '/home/nld6854/earthcare_scripts/scripts/april_2025'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), script_path)))

from ectools import ecio
from ectools import ecplot as ecplt
from ectools import colormaps
from plotting_tools import read_h5

def read_cams_ext(filen):
    a=xr.open_dataset(filen,engine="h5netcdf")
    ext = a['cams_interp_varp']
    org = a['cams_orography']
    a.close()
    febd = filen.replace('CAMS','EBD')
    febd = febd.replace('.nc','.h5')
    _,lat,lon,_,h,_,_,_ = read_h5.get_ext(febd,np.array([10,11,12,13,14,15,25,26,27]))
    org2 = org.values[:, None]     # shape becomes (4825, 1)
    org2 = np.repeat(org2, h.shape[1], axis=1)   # now (4825, 241)

    ext = np.where(h>org2,ext,0)
    return ext,lat,lon,h

month = 'february'
#simple_classification
which_aerosol='total'

mean_or_std = 'mean'

year = '2024' if month == 'december' else '2025'
# List of file paths you want to read concurrently
file_paths = glob.glob('/scratch/nld6854/earthcare/earthcare_data/'+month+'_'+year+'/CAMS/*.nc')
file_paths.sort()
# Use ProcessPoolExecutor to read files concurrently
results = []
##with concurrent.futures.ProcessPoolExecutor() as executor:
##    for result in executor.map(read_cams_ext, file_paths):
##        if result is not None:  # Filter out None results from failed reads
##            results.append(result)
with Pool(processes=8) as pool:  # adjust number of processes
    for result in pool.map(read_cams_ext, file_paths):
        if result is not None:  # Filter out None results from failed reads
            results.append(result)


if len(results) == 0:
    raise RuntimeError("No CAMS files read successfully.")

print("Reading CAMS + EBD files finished")


# Unpack lists
ext_list = []
lat_list = []
lon_list = []
height_list = []

for ext, lat, lon, h in results:
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

for idx, (ext, lat, lon, h) in enumerate(results):

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

    for ext, lat, lon, _ in results:
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

    for ext, lat, lon, _ in results:
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
    f"{outdir}/regridded_CAMS_{which_aerosol}"
    f"_extinction_coe_2deg_{mean_or_std}_single_alt_{month}_{year}_snr_gr_2.nc"
)

outfile_masknan = (
    f"{outdir}/regridded_CAMS_{which_aerosol}"
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

