import numpy as np
import cartopy.crs as ccrs
import xarray as xr
from pylab import *
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import sys
import os
script_path = '/home/nld6854/earthcare_scripts/scripts/april_2025'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), script_path)))

month='october'
fmonth='October'
vname = 'extinction_coefficient'
figname = 'extinction_coefficient'

year = '2024' if month == 'december' else '2025'
cams_dir = '/scratch/nld6854/earthcare/cams_data/'+month+'_'+year+'/'
file_atlid = cams_dir+'regridded_satellite_total_extinction_coe_2deg_masknan_mean_single_alt_'+month+'_'+year+'_snr_gr_2.nc'
file_cams = file_atlid.replace('satellite','CAMS')#'regridded_CAMS_total_extinction_coe_2deg_masknan_mean_single_alt_'+month+'_2025.nc'

print(file_cams)

# Define region(s) as dictionary:
# Name : [lat_min, lat_max, lon_min, lon_max]
regions = {
    "North_America"  : [ 26, 60, -120, -76],
    "North_Atlantic" : [ 13, 37,  -60, -19],
    "North_Africa"   : [ 13, 37,   -4,  40],
    "East_China"     : [ 20, 40,  105, 122],
    "South_Africa"   : [-29, -3,   10,  40],
    "Southern_Ocean" : [-70,-40, -180, 180],
}

# ============================================================
# LOAD DATA
# ============================================================

ds_cams  = xr.open_dataset(file_cams)
ds_atlid = xr.open_dataset(file_atlid)

ext_cams  = ds_cams["extinction_coefficient"]
ext_atlid = ds_atlid["extinction_coefficient"]

lat = ds_cams["latitude"]
lon = ds_cams["longitude"]
height = ds_cams["height"]

# ============================================================
# HELPER FUNCTION: regional mean extinction profile
# ============================================================

def regional_mean_profile_weighted(ext, lat, lon,
                                   lat_min, lat_max, lon_min, lon_max):
    """
    Compute area-weighted mean extinction profile over a region.
    ext: (lat, lon, height)
    """

    # Select region
    region = ext.sel(
        latitude=slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max)
    )

    # Latitude weights
    lat_rad = np.deg2rad(region.latitude.values)
    weights = np.cos(lat_rad)

    # Normalize
    weights = weights / np.nansum(weights)

    # Expand to 3D: (lat, 1, 1) so it broadcasts over lon & height
    w3d = weights[:, np.newaxis, np.newaxis]

    # Weighted average:
    # Multiply by weights, mean over lon, sum over lat
    weighted = region * w3d
    prof = weighted.mean(dim="longitude", skipna=True).sum(dim="latitude", skipna=True)

    return prof

# ============================================================
# LOOP OVER REGIONS AND COMPUTE MEAN PROFILES
# ============================================================

profiles = {}

for name, (lat_min, lat_max, lon_min, lon_max) in regions.items():
    print(f"Processing region: {name}")

    prof_cams  = regional_mean_profile_weighted(ext_cams,  lat, lon, lat_min, lat_max, lon_min, lon_max)
    prof_atlid = regional_mean_profile_weighted(ext_atlid, lat, lon, lat_min, lat_max, lon_min, lon_max)
    diff = ext_cams.copy()
    diff.values = ext_cams.values-ext_atlid.values 
    prof_diff  = regional_mean_profile_weighted(diff,      lat, lon, lat_min, lat_max, lon_min, lon_max)

    # Compute regional mean AOD (simple vertical integral)
    # ext is m^-1, height is meters ⇒ integrate along height dimension
    aod_cams  = np.trapezoid(prof_cams,  x=height)
    aod_atlid = np.trapezoid(prof_atlid, x=height)

    profiles[name] = {
        "cams_profile"  : prof_cams,
        "atlid_profile" : prof_atlid,
        "cams_aod"      : float(aod_cams),
        "atlid_aod"     : float(aod_atlid),
    }

    fig,ax = plt.subplots(1,figsize=(4,5))
    ax.plot(prof_cams,height/1000,'r-',label='CAMS')
    ax.plot(prof_atlid,height/1000,'k-',label='ATLID')
    ax.plot(prof_diff.values,height/1000,'g--',label='CAMS-ATLID')
    ax.axvline(x=0,color='lightgray',alpha=0.5)
    ax.legend(frameon=False, fontsize=15)
    ax.set_title(name,fontsize=15)
    ax.set_xlabel('Extinction coefficient / m$^{-1}$',fontsize=15)
    ax.set_ylabel('Altitude / km', fontsize=15)
    ax.set_ylim(0,20)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    fig.tight_layout()
    fig.savefig(name+'_domain_mean_ext_prof.jpg')

# ============================================================
# PRINT SUMMARY
# ============================================================

for name, data in profiles.items():
    print("\n===== ", name, " =====")
    print("CAMS AOD :", data["cams_aod"])
    print("ATLID AOD:", data["atlid_aod"])
    print("Profile shapes:", data["cams_profile"].shape, data["atlid_profile"].shape)


