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

def filter_uncertainty(uncertainty,count,median):
    with np.errstate(divide='ignore', invalid='ignore'):
        condition = (uncertainty / median <= 1) & (count > 10)
    aod_filtered = np.where(condition, median, np.nan)
    return aod_filtered

# Load your dataset (assuming it's a CSV with 'lat', 'lon', 'aod')
file_dir = '/scratch/nld6854/earthcare/cams_data/'
day_or_night = 'day_'
day_or_night = ''

#simple_classification
which_aerosol='total'

mean_or_std = 'median'
months = [
    ('december', 'December', '2024'),
    ('january',  'January',  '2025'),
    ('february', 'February', '2025'),
    ('march',    'March',    '2025'),
    ('april',    'April',    '2025'),
    ('may',      'May',      '2025'),
    ('june',     'June',     '2025'),
    ('july',     'July',     '2025'),
    ('august',   'August',   '2025'),
    ('september','September','2025'),
    ('october',  'October',  '2025'),
    ('november', 'November', '2025'),
]


atlid_all = []
cams_all  = []
diff_all  = []
count_all = []
uncertainty_all = []
lat_all,lon_all = [],[]
month_labels = []

for month, fmonth, year in months:

    print('Processing', fmonth, year)

    df = pd.read_csv(
        file_dir + month + '_' + year + '/' +
        year + "_" + month + "_cams_atlid_co-located_" +
        day_or_night + "aod_snr_gr_2.txt",
        delimiter=","
    )

    dfuncer = xr.open_dataset(
        file_dir + month + '_' + year + '/' +
        year + "_" + month + '_atlid_' +
        day_or_night + 'uncertainty_snr_gr_2.nc'
    )

    uncertainty = dfuncer['sigma_total']
    count = dfuncer['count']

    #2 deg resolution
    a_aod = df['aod355nm_atlid'].values
    c_aod = df['aod355nm_cams'].values
    all_lat = df['# latitude'].values
    all_lon = df['longitude'].values

    atlid_all.extend(a_aod)
    cams_all.extend(c_aod)
    count_all.extend(count)
    uncertainty_all.extend(uncertainty)
    lat_all.extend(all_lat)
    lon_all.extend(all_lon)
   
print(len(atlid_all))

atlid_all = np.asarray(atlid_all)
c_aod = np.asarray(cams_all)
count = np.asarray(count_all)
uncertainty = np.asarray(uncertainty_all)
all_lat = np.asarray(lat_all)
all_lon = np.asarray(lon_all)

a_aod = np.where(atlid_all>=0.02, atlid_all, np.nan)

print('max and min of latitude before mask',np.nanmax(all_lat),np.nanmin(all_lat))
mask = ~np.isnan(a_aod) & ~np.isnan(c_aod)
a_aod = a_aod[mask]
c_aod = c_aod[mask]
all_lat = all_lat[mask]
all_lon = all_lon[mask]

reso = 2
lat_bins = np.arange(-90., 90.+reso, reso)
lon_bins = np.arange(-180, 180.+reso, reso)

# Compute 2D histogram for the mean wind
aod_atlid, x_edge, y_edge, _ = binned_statistic_2d(
    all_lat, all_lon, a_aod, statistic=mean_or_std, bins=[lat_bins, lon_bins])

aod_cams, x_edge, y_edge, _ = binned_statistic_2d(
    all_lat, all_lon, c_aod, statistic=mean_or_std, bins=[lat_bins, lon_bins])

diff = aod_cams - aod_atlid

lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2


ds = xr.Dataset(
    {
        "atlid_aod": (("latitude", "longitude"), aod_atlid),
        "cams_aod": (("latitude", "longitude"), aod_cams)
    },
    coords={
        "latitude": lat_centers,
        "longitude": lon_centers
    }
)
cams_dir = '/scratch/nld6854/earthcare/cams_data/'
ds.to_netcdf(cams_dir+'122024-112025_atlid_'+day_or_night+'aod_filtered_with_uncertainty_snr_gr_2.nc', format="NETCDF4")

clons,clats = np.meshgrid(lon_centers,lat_centers)
nan_percentage = np.isnan(aod_cams).sum() / aod_cams.size * 100
print(f"Percentage of NaNs in regridded data: {nan_percentage:.2f}%")

vmax = aod_cams.max()
vmax = 1
print('aod_cams.max()=',aod_cams.max())

def landsea_mean(var):
    cams_lsm = Dataset('/scratch/nld6854/earthcare/cams_data/landsea_mask.nc')
    lsm0 = cams_lsm.variables['lsm'][0,0]
    lat_cams_1 = cams_lsm.variables['latitude']
    lon_cams_1 = cams_lsm.variables['longitude']
    lon_cams_1,lat_cams_1=np.meshgrid(lon_cams_1,lat_cams_1)
    print(lon_cams_1.shape)

    stat, x_edge, y_edge, _ = binned_statistic_2d(
    lat_cams_1.flatten(),lon_cams_1.flatten(), lsm0.flatten(), statistic=mean_or_std, bins=[lat_bins, lon_bins])

    lsm = np.round(stat)
    print(lsm.shape,var.shape)

    land = np.nanmedian(var[np.where(lsm==1)])
    sea = np.nanmedian(var[np.where(lsm==0)])
    print('not weighted by area')
    return land,sea

mask = (~np.isnan(aod_cams)) & (~np.isnan(aod_atlid))
aland,asea = landsea_mean(np.where(mask,aod_atlid,np.nan))
print('ATLID land=',round(aland,4),'sea=',round(asea,4))
cland,csea = landsea_mean(np.where(mask,aod_cams,np.nan))
print('CAMS land=',round(cland,4),'sea=',round(csea,4))
acland,acsea = landsea_mean(np.where(mask,aod_cams-aod_atlid,np.nan))
print('CAMS-ATLID land=',round(acland,4),'sea=',round(acsea,4))

aod_atlid_temp = aod_atlid.copy()
if len(aod_atlid_temp[aod_atlid_temp==0]) > 0:
    aod_atlid_temp[aod_atlid_temp==0] = np.nan
    print('excludes 0')
acpland,acpsea = landsea_mean(np.where(mask,(aod_cams-aod_atlid)/aod_atlid_temp,np.nan))
print('CAMS-ATLID/ATLID land=',round(acpland,4),'sea=',round(acpsea,4))


if mean_or_std == 'mean':
    print('CAMS AOD mean=',round(np.nanmean(aod_cams[mask]),4))
    print('ATLID AOD mean=',round(np.nanmean(aod_atlid[mask]),4))
    print('CAMS-ATLID mean=',round(np.nanmean(aod_cams[mask]-aod_atlid[mask]),4))
    print('CAMS-ATLID/ATLID mean=',round(np.nanmean((aod_cams[mask]-aod_atlid[mask])/aod_atlid[mask]),4))
else: 
    print('CAMS AOD median=',round(np.nanmedian(aod_cams[mask]),4))
    print('ATLID AOD median=',round(np.nanmedian(aod_atlid[mask]),4))
    print('CAMS-ATLID median=',round(np.nanmedian(aod_cams[mask]-aod_atlid[mask]),4))
    aod_atlid_mask = aod_atlid[mask]
    if len(aod_atlid_mask[aod_atlid_mask==0]) > 0:
        aod_atlid_mask[aod_atlid_mask==0] = np.nan
        print('excludes 0')
    print('CAMS-ATLID/ATLID median=',round(np.nanmedian((aod_cams[mask]-aod_atlid[mask])/aod_atlid_mask),4))
    print('Median absolute deviation ATLID',round(np.median(np.abs(aod_atlid[mask] - np.median(aod_atlid[mask]))),4))
    print('Median absolute deviation CAMS',round(np.median(np.abs(aod_cams[mask] - np.median(aod_cams[mask]))),4))
print('CAMS,ATLID NMB=',round(statistics.normalized_mean_bias(aod_cams,aod_atlid),4))

#fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(20,30),subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0)))
fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(8,12),subplot_kw=dict(projection=ccrs.PlateCarree()))
cmap = plt.colormaps['plasma']
bounds = np.logspace(np.log10(0.02),0,num=11)
norm = colors.BoundaryNorm(bounds, ncolors=cmap.N, clip=True)
#plt.figure(figsize=(6,3))
#ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax1.set_extent([-180, 180, -89.9, 89.9])#,crs=ccrs.PlateCarree(central_longitude=180))
all_lon = np.where(clons>0,clons,clons+360)
im=ax1.pcolormesh(all_lon,clats,aod_atlid,cmap='plasma',transform=ccrs.PlateCarree(),norm=norm)
#cs = ax1.contour(all_lon,clats,aod_atlid,levels=[0.1],colors='white')
#plt.clabel(cs, inline=1, fontsize=10)

gl = ax1.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
             linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

ax1.coastlines(resolution='110m')
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

bar = plt.colorbar(im, orientation='vertical',ax=ax1,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('-',fontsize=15)
bar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
bar.ax.tick_params(labelsize=15)

ax1.set_title('ATLID integrated AOD '+day_or_night[:-1]+' Dec 2024 - Nov 2025',fontsize=15)



ax2.set_extent([-180, 180, -89.9, 89.9])
im=ax2.pcolormesh(clons,clats,aod_cams,cmap='plasma',transform=ccrs.PlateCarree(),norm=norm)
#cs = ax2.contour(all_lon,clats,aod_atlid,levels=[0.1],colors='white')
#plt.clabel(cs, inline=1, fontsize=10)

gl = ax2.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
             linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

ax2.coastlines(resolution='110m')
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

bar = plt.colorbar(im, orientation='vertical',ax=ax2,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('-',fontsize=15)
bar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
bar.ax.tick_params(labelsize=15)
ax2.set_title('CAMS total AOD',fontsize=15)

ax3.set_extent([-180, 180, -89.9, 89.9])
bounds2 = np.linspace(-vmax/5,vmax/5,11)
norm2 = colors.BoundaryNorm(bounds2,ncolors=plt.colormaps['RdYlBu_r'].N,clip=True)
im=ax3.pcolormesh(clons,clats,aod_cams-aod_atlid,cmap='RdYlBu_r',transform=ccrs.PlateCarree(),norm=norm2)
gl = ax3.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
             linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

ax3.coastlines(resolution='110m')
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

bar = plt.colorbar(im, orientation='vertical',ax=ax3,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('-',fontsize=15)
bar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
bar.ax.tick_params(labelsize=15)
ax3.set_title('CAMS-ATLID AOD',fontsize=15)

plt.tight_layout()
fig.savefig('figures/global_'+day_or_night+'aod_'+which_aerosol+'_'+str(reso)+'deg_binned_'+mean_or_std+'_122024-112025_co-located_filtered_uncertainty.jpg',bbox_inches='tight')

nbins = 150
binsc = np.linspace(0,np.nanmax(aod_cams.flatten()),nbins)
histc,binsc = np.histogram(aod_cams,bins=binsc,density=False)
bcc = 0.5*(binsc[1:] + binsc[:-1])

binsa = np.linspace(0,np.nanmax(aod_atlid),nbins)
hista,binsa = np.histogram(aod_atlid,bins=binsa,density=False)
bca = 0.5*(binsa[1:] + binsa[:-1])

binsa0 = np.linspace(0,np.nanmax(a_aod),nbins)
hista0,binsa0 = np.histogram(a_aod,bins=binsa0,density=False)
bca0 = 0.5*(binsa0[1:] + binsa0[:-1])

binsa_ = np.linspace(0,np.nanmax(aod_atlid),nbins)
hista_,binsa_ = np.histogram(aod_atlid,bins=binsa_,density=True)
bca_ = 0.5*(binsa_[1:] + binsa_[:-1])

binsa0_ = np.linspace(0,np.nanmax(a_aod),nbins)
hista0_,binsa0_ = np.histogram(a_aod,bins=binsa0_,density=True)
bca0_ = 0.5*(binsa0_[1:] + binsa0_[:-1])

a_errors = np.sqrt(hista)
c_errors = np.sqrt(histc)
#fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5),gridspec_kw={'width_ratios': [2, 1, 1]},sharey=False)
fig,ax1 = plt.subplots(1,figsize=(6,5))
ax1.plot(bcc,histc,color='blue',label='CAMS')
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
ax1.set_title(day_or_night[:-1]+' Dec 2024 - Nov 2025',fontsize=15)
ax1.legend(frameon=False,fontsize=15)

fig.savefig('figures/histograms_CAMS_ATLID_'+str(reso)+'deg_binned_'+mean_or_std+'_122024-112025_co-located_filtered_'+day_or_night+'uncertainty_with_modes.jpg')

