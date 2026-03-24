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
os.environ['MAAP_credentials'] = '/home/nld6854/earthcare_scripts/scripts/april_2025/ectools/ectools/maap_credentials.txt'

from ectools.ectools_edited import ecio
from ectools.ectools_edited import ecplot as ecplt
from ectools.ectools_edited import colormaps
from plotting_tools import read_h5

def get_ext_wrapper(args):
    return read_h5.get_ext(*args)

month = 'november'
#simple_classification
which_aerosol='total'

months = ['january','february','march','april','may','june','july','august','september','october','november','december']
months = ['january','february','march','april','may','june','july','august','september','october','november','december']

args = np.array([10,11,12,13,14,15,25,26,27])
hfile = [glob.glob('/scratch/nld6854/earthcare/earthcare_data/march_2025/EBD/*.h5')[0]]
for f in hfile:
    results = read_h5.get_ext(f,args)

target_h = results[4][0]
for month in months:
    print(month)
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
                results.append(result[:6])
    
    print('Reading all EBD files finished')
   
    # Unpack lists
    #ext_list = []
    lat_list = []
    lon_list = []
    elev_list = []
   
    for ext, lat, lon,time, h, elev in results:
        #ext_list.append(np.asarray(ext))
        lat_list.append(np.asarray(lat))
        lon_list.append(np.asarray(lon))
        elev_list.append(np.asarray(elev))

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
    # Use the height levels of the first file as target grid
    n_height = target_h.shape[0]
    print("Using target height grid from first file:", n_height,target_h.shape)   
 
    # ==== Grid elevation to 2degree ====
    flat_lat = np.concatenate(lat_list).ravel()
    flat_lon = np.concatenate(lon_list).ravel()
    flat_elev = np.concatenate(elev_list).ravel()

    '''
    var = lon_list
    valid_count = sum(np.isfinite(sub).sum() for sub in var)
    total_count = sum(len(sub) for sub in var)
    valid_percentage = (valid_count / total_count) * 100
    print(f'Valid elevation: {valid_count} / {total_count} ({valid_percentage:.2f}%)')

    var = flat_elev
    valid_count = np.isfinite(var).sum()
    total_count = len(var)
    valid_ratio = (valid_count / total_count) * 100
    print(f"--- Latitude Data Check ---")
    print(f"Total Points: {total_count}")
    print(f"Valid (Finite) Points: {valid_count}")
    print(f"Valid Percentage: {valid_ratio:.2f}%")
    print(f"Array Shape: {flat_lat.shape}")
    '''
    elevation, _, _, _ = binned_statistic_2d(
        flat_lat, flat_lon, flat_elev,
        statistic='mean',
        bins=[lat_bins, lon_bins]
    )

    # ============================================================
    # INTERPOLATE ALL PROFILES TO A COMMON HEIGHT GRID
    # ============================================================
   
  
    ext_list_interp = []
    lat_list_interp = []
    lon_list_interp = []
   
    from scipy.interpolate import interp1d
   
    for idx, (ext, lat, lon, time, h, elev) in enumerate(results):
    
        # ext: (n_obs, n_height_i)
        # h:   (n_obs, n_height_i)   <-- each profile has its own height grid
    
        n_obs, _ = ext.shape
        ext_interp = np.full((n_obs, n_height), np.nan)   # fill with NaN by default
    
        for i in range(n_obs):
   
            # Profile-specific height vector (1D)
            h_i = h[i, :]
            ext_i = ext[i, :]
   
            # Remove NaNs from h_i (can happen after cloud masking)
            #valid = np.isfinite(h_i) & np.isfinite(ext_i)
            #if np.sum(valid) < 2:
            #    # Not enough points to interpolate
            #    continue
   
            h_valid = h_i#[valid]
            ext_valid = ext_i#[valid]
    
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

    '''
    fig,ax=plt.subplots(1)
    all_data = np.concatenate([arr.flatten() for arr in ext_list])
    all_data = all_data[np.isfinite(all_data)]

    ax.hist(all_data, bins=100, color='skyblue', edgecolor='black')
   
    ax.set_title('Histogram of Extinction Coefficients')
    ax.set_xlabel('Extinction Value')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig('quicklook1.jpg')
    '''
    print("All CAMS extinction profiles are now interpolated to a common height grid.")
   

 
    # ============================================================
    # REGRID — VERSION 2 (REMOVE NAN BEFORE BINNING)
    # ============================================================
    
    regridded_data_masknan = np.full((n_lat, n_lon, n_height), np.nan)
    regridded_std_masknan = np.full((n_lat, n_lon, n_height), np.nan)
    regridded_count_masknan = np.full((n_lat, n_lon, n_height), np.nan)
    regridded_total_count = np.full((n_lat, n_lon, n_height), np.nan)
    
    #fig,ax=plt.subplots(1)
    for k in range(n_height):
        ext_k = []
        lat_k = []
        lon_k = []
    
        ext_k_allnan,lat_k_allnan,lon_k_allnan = [],[],[]
        for ext, lat, lon in zip(ext_list, lat_list, lon_list):
            if ext.shape[1] <= k:
                continue
    
            # keep all data including NaN
            ext_k_allnan.append(ext[:, k])
            lat_k_allnan.append(lat)
            lon_k_allnan.append(lon)

            # Remove NaNs BEFORE binning
            mask = np.isfinite(ext[:, k])
            if not np.any(mask):
                continue
    
            ext_k.append(ext[:, k][mask])
            lat_k.append(lat[mask])
            lon_k.append(lon[mask])
    
            #t = ext[:, k][ext[:, k]>=0]
            #ax.plot(np.arange(len(t)),t)

        if len(ext_k) == 0:
            continue
    
        # Grid and save all counts for 2 deg grids
        allnan_ext = np.concatenate(ext_k_allnan)
        allnan_lat = np.concatenate(lat_k_allnan)
        allnan_lon = np.concatenate(lon_k_allnan)

        all_count, _, _, _ = binned_statistic_2d(
            allnan_lat, allnan_lon, allnan_ext,
            statistic='count',
            bins=[lat_bins, lon_bins]
        )
    
        regridded_total_count[:, :, k] = all_count

        # Grid and save all with valid values for 2 deg grids
        all_ext = np.concatenate(ext_k)
        all_lat = np.concatenate(lat_k)
        all_lon = np.concatenate(lon_k)
    
        stat, _, _, _ = binned_statistic_2d(
            all_lat, all_lon, all_ext,
            statistic=mean_or_std,
            bins=[lat_bins, lon_bins]
        )
    
        stat_count, _, _, _ = binned_statistic_2d(
            all_lat, all_lon, all_ext,
            statistic='count',
            bins=[lat_bins, lon_bins]
        )
        
        stat_std, _, _, _ = binned_statistic_2d(
            all_lat, all_lon, all_ext,
            statistic='std',
            bins=[lat_bins, lon_bins]
        )

        #stat[stat_count < min_count] = np.nan
        #stat_std[stat_count < min_count] = np.nan
        #stat_count[stat_count < min_count] = np.nan
        regridded_data_masknan[:, :, k] = stat
        regridded_std_masknan[:, :, k] = stat_std
        regridded_count_masknan[:, :, k] = stat_count

    #fig.tight_layout()
    #fig.savefig('quicklook2.jpg')
    
    
    # ============================================================
    # SAVE BOTH FILES
    # ============================================================
    
    outdir = f"/scratch/nld6854/earthcare/cams_data/{month}_{year}"
    os.makedirs(outdir, exist_ok=True)
    
    outfile_masknan = (
        f"{outdir}/regridded_satellite_{which_aerosol}"
        f"_extinction_coe_2deg_masknan_{mean_or_std}_single_alt_{month}_{year}_snr_gr_2_extra_output.nc"
    )
    
    # Save version 2 (masknan)
    xr.Dataset(
        {
            "extinction_coefficient": (["latitude", "longitude", "height"], regridded_data_masknan),
            "standard_deviation": (["latitude", "longitude", "height"], regridded_std_masknan),
            "valid_count": (["latitude", "longitude", "height"], regridded_count_masknan),
            "total_count": (["latitude", "longitude", "height"], regridded_total_count),
            "elevation": (["latitude","longitude"], elevation)
        },
        coords={
            "latitude": lat_centers,
            "longitude": lon_centers,
            "height": height
        }
    ).to_netcdf(outfile_masknan)
   
    print("Saved BOTH files:")
    print(" →", outfile_masknan)
 
