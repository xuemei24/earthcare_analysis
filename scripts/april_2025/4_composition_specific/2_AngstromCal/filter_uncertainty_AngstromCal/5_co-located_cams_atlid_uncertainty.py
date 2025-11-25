import h5py
import concurrent.futures
import glob
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
from scipy.stats import binned_statistic_2d
from pylab import *
import pandas as pd
import matplotlib.colors as colors
import sys
import os
from scipy.signal import find_peaks

script_path = '/home/nld6854/earthcare_scripts/scripts/april_2025'
sys.path.append(script_path)
from plotting_tools import statistics

# Load your dataset (assuming it's a CSV with 'lat', 'lon', 'aod')
month = 'december'
fmonth = 'December'
print('Month=',month)
which_aerosol = 'dust'
#which_aerosol = 'ssa'
fname = 'sea_salt' if which_aerosol == 'ssa' else 'dust'
figname = 'Sea salt'  if which_aerosol=='ssa' else 'Dust+BC+OM'
if which_aerosol == 'dust':
    fname = 'dust_om'
print(fname)

mean_or_std = 'median'

year = '2024' if month == 'december' else '2025'
file_dir = '/scratch/nld6854/earthcare/cams_data/'+month+'_'+year+'/TTcal/'
dfaod = pd.read_csv(file_dir+'AngstromCal_'+fname+'_aod355nm_per_composition_'+month+'_'+year+'_cams_atlid_co-located_snr_gr_2.txt', delimiter=",")
dfaod_var = pd.read_csv(file_dir+fname+'_aod355nm_per_composition_'+month+'_'+year+'_cams_atlid_co-located_aod_variance_snr_gr_2.txt', delimiter=",")


AOD = dfaod['aod355nm_atlid'].values
all_lat = dfaod['# latitude'].values
all_lon = dfaod['longitude'].values

AOD_var = dfaod_var['aod_355nm_var_atlid'].values



reso = 2
lat_bins = np.arange(-90., 90.+reso, reso)
lon_bins = np.arange(-180, 180.+reso, reso)
lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2



###old aod regridding
a_aod = dfaod['aod355nm_atlid'].values
c_aod = dfaod['aod355nm_cams'].values

print('max and min of latitude before mask',np.nanmax(all_lat),np.nanmin(all_lat))
mask = ~np.isnan(a_aod) & ~np.isnan(c_aod)
a_aod = a_aod[mask]
all_lat = all_lat[mask]
all_lon = all_lon[mask]

# Compute 2D histogram for the mean wind
aod_atlid, x_edge, y_edge, _ = binned_statistic_2d(
    all_lat, all_lon, a_aod, statistic=mean_or_std, bins=[lat_bins, lon_bins])

vmax = 1

clons,clats = np.meshgrid(lon_centers,lat_centers)
###old aod regridding

AOD = AOD[mask]
AOD_var = AOD_var[mask]

def binned_median_with_total_uncertainty(
    lons, lats, values, variances, xbins, ybins,
    n_meas_draws=300, n_bootstrap=300, min_count=5, compute_combined=True
):
    """
    Compute gridded median and uncertainties (measurement + sampling + total).

    Parameters
    ----------
    lons, lats, values : 1D arrays
        Longitude, latitude, and AOD values.
    variances : 1D array
        Measurement variances corresponding to `values`.
    xbins, ybins : arrays
        Longitude and latitude bin edges.
    n_meas_draws : int
        Number of Monte Carlo draws for measurement uncertainty.
    n_bootstrap : int
        Number of bootstrap iterations for sampling uncertainty.
    min_count : int
        Minimum samples per grid cell.
    compute_combined : bool
        If True, return combined uncertainty.

    Returns
    -------
    median_map : 2D array
        Gridded median values.
    sigma_meas_map : 2D array
        Measurement uncertainty of median.
    sigma_sample_map : 2D array
        Sampling uncertainty of median.
    sigma_total_map : 2D array
        Total combined uncertainty (optional).
    count_map : 2D array
        Number of samples per bin.
    """

    # Step 1: get bin assignments
    stat, x_edge, y_edge, binnumber = binned_statistic_2d(
        lons, lats, values, statistic='median', bins=[xbins, ybins]
    )

    median_map = stat.T
    sigma_meas_map = np.full_like(median_map, np.nan, dtype=float)
    sigma_sample_map = np.full_like(median_map, np.nan, dtype=float)
    sigma_total_map = np.full_like(median_map, np.nan, dtype=float)
    count_map = np.zeros_like(median_map, dtype=int)

    # Step 2: iterate through bins
    for i in range(len(xbins) - 1):
        for j in range(len(ybins) - 1):
            mask = (
                (lons >= xbins[i]) & (lons < xbins[i+1]) &
                (lats >= ybins[j]) & (lats < ybins[j+1])
            )
            vals = values[mask]
            vars_ = variances[mask]
            count_map[j, i] = len(vals)

            if len(vals) < min_count:
                continue

            sigs = np.sqrt(vars_)

            # --- A. Measurement uncertainty via Monte Carlo ---
            draws = np.random.normal(loc=vals, scale=sigs, size=(n_meas_draws, len(vals)))
            med_meas = np.nanmedian(draws, axis=1)
            sigma_meas = np.nanstd(med_meas)

            # --- B. Sampling uncertainty via bootstrapping ---
            med_samp = [np.nanmedian(np.random.choice(vals, len(vals), replace=True))
                        for _ in range(n_bootstrap)]
            sigma_sample = np.nanstd(med_samp)

            # --- Combine results ---
            sigma_meas_map[j, i] = sigma_meas
            sigma_sample_map[j, i] = sigma_sample
            if compute_combined:
                sigma_total_map[j, i] = np.sqrt(sigma_meas**2 + sigma_sample**2)

    if compute_combined:
        return median_map, sigma_meas_map, sigma_sample_map, sigma_total_map, count_map
    else:
        return median_map, sigma_meas_map, sigma_sample_map, count_map

#AOD_var should be calculated =  AOD_var = np.sum(var_prof * (delta_z ** 2), axis=1)

median_map, sigma_meas, sigma_samp, sigma_total, count_map = binned_median_with_total_uncertainty(
    all_lon, all_lat, AOD, AOD_var,
    lon_bins, lat_bins,
    n_meas_draws=500,
    n_bootstrap=500
)

ds = xr.Dataset(
    {
        "median_aod": (("latitude", "longitude"), median_map),
        "sigma_meas": (("latitude", "longitude"), sigma_meas),
        "sigma_samp": (("latitude", "longitude"), sigma_samp),
        "sigma_total": (("latitude", "longitude"), sigma_total),
        "count": (("latitude", "longitude"), count_map),
    },
    coords={
        "latitude": lat_centers,
        "longitude": lon_centers
    }
)

ds.to_netcdf(file_dir+'AngstromCal_'+fname+'_aod355nm_per_composition_'+month+'_atlid_uncertainty_snr_gr_2.nc', format="NETCDF4")

#fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(20,30),subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0)))
fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(20,30),subplot_kw=dict(projection=ccrs.PlateCarree()))
cmap = plt.colormaps['plasma']
#bounds = np.logspace(-2,0,num=10)
#norm = colors.BoundaryNorm(bounds, ncolors=cmap.N, clip=True)
ax1.set_extent([-180, 180, -89.9, 89.9])#,crs=ccrs.PlateCarree(central_longitude=180))
all_lon = np.where(clons>0,clons,clons+360)
im=ax1.pcolormesh(all_lon,clats,sigma_total,cmap='viridis',transform=ccrs.PlateCarree())#,norm=norm)
#cs = ax1.contour(all_lon,clats,aod_atlid,levels=[0.1],colors='white')
#plt.clabel(cs, inline=1, fontsize=10)

gl = ax1.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
             linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False

ax1.coastlines(resolution='110m')#,color='white')
ax1.gridlines()
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

bar = plt.colorbar(im, orientation='vertical',ax=ax1,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('ATLID total std / -',fontsize=15)
bar.ax.tick_params(labelsize=15)

ax1.set_title('ATLID total std '+fmonth+' '+year,fontsize=15)



ax2.set_extent([-180, 180, -89.9, 89.9])
im=ax2.pcolormesh(clons,clats,sigma_meas,cmap='viridis',transform=ccrs.PlateCarree())#,norm=norm)
#cs = ax2.contour(all_lon,clats,aod_atlid,levels=[0.1],colors='white')
#plt.clabel(cs, inline=1, fontsize=10)

gl = ax2.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
             linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False

ax2.coastlines(resolution='110m')
ax2.gridlines()
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

bar = plt.colorbar(im, orientation='vertical',ax=ax2,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('ATLID AOD measurement std / -',fontsize=15)
bar.ax.tick_params(labelsize=15)
ax2.set_title('ATLID AOD measurement std',fontsize=15)

ax3.set_extent([-180, 180, -89.9, 89.9])
im=ax3.pcolormesh(clons,clats,sigma_samp,cmap='viridis',transform=ccrs.PlateCarree())#,norm=norm)
gl = ax3.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
             linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False

ax3.coastlines(resolution='110m')
ax3.gridlines()
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

bar = plt.colorbar(im, orientation='vertical',ax=ax3,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('ATLID AOD sampling std / -',fontsize=15)
bar.ax.tick_params(labelsize=15)
ax3.set_title('ATLID AOD sampling std',fontsize=15)

plt.tight_layout()
fig.savefig('global_'+which_aerosol+'aod_uncertainty_'+str(reso)+'deg_binned_'+mean_or_std+'_'+month+'_'+year+'_co-located.jpg',bbox_inches='tight')


