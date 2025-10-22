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
file_dir = '/scratch/nld6854/earthcare/cams_data/'
month = 'september'
fmonth = 'September'
print('Month=',month)
df = pd.read_csv(file_dir+month+'_2025/'+"2025_"+month+"_cams_atlid_co-located_aod_snr_gr_2.txt", delimiter=",")
dfuncer = xr.open_dataset(file_dir+month+'_2025/'+"2025_"+month+'_atlid_uncertainty_snr_gr_2.nc')
uncertainty = dfuncer['sigma_total']
count = dfuncer['count']

#simple_classification
which_aerosol='total'
#which_aerosol='sea_salt'
#which_aerosol='dust'
# Define a function that reads data from an HDF5 file

mean_or_std = 'median'

#2 deg resolution
a_aod = df['aod355nm_atlid'].values
c_aod = df['aod355nm_cams'].values
all_lat = df['# latitude'].values
all_lon = df['longitude'].values

a_aod = np.where(a_aod>=0.02,a_aod,np.nan)

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

def filter_uncertainty(uncertainty,count,median):
    with np.errstate(divide='ignore', invalid='ignore'):
        condition = (uncertainty / median <= 1) & (count > 10)
    aod_filtered = np.where(condition, median, np.nan)
    return aod_filtered

#aod_atlid, aod_cams = filter_uncertainty(uncertainty,count,aod_atlid),filter_uncertainty(uncertainty,count,aod_cams)
aod_atlid = filter_uncertainty(uncertainty,count,aod_atlid)
# Replace NaN with zeros (or another value if necessary)
#stat = np.nan_to_num(stat)

lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

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
print('ATLID land=',aland,'sea=',asea)
cland,csea = landsea_mean(np.where(mask,aod_cams,np.nan))
print('CAMS land=',cland,'sea=',csea)
acland,acsea = landsea_mean(np.where(mask,aod_cams-aod_atlid,np.nan))
print('CAMS-ATLID land=',acland,'sea=',acsea)

aod_atlid_temp = aod_atlid.copy()
if len(aod_atlid_temp[aod_atlid_temp==0]) > 0:
    aod_atlid_temp[aod_atlid_temp==0] = np.nan
    print('excludes 0')
acpland,acpsea = landsea_mean(np.where(mask,(aod_cams-aod_atlid)/aod_atlid_temp,np.nan))
print('CAMS-ATLID/ATLID land=',acpland,'sea=',acpsea)


if mean_or_std == 'mean':
    print('CAMS AOD mean=',np.nanmean(aod_cams[mask]))
    print('ATLID AOD mean=',np.nanmean(aod_atlid[mask]))
    print('CAMS-ATLID mean=',np.nanmean(aod_cams[mask]-aod_atlid[mask]))
    print('CAMS-ATLID/ATLID mean=',np.nanmean((aod_cams[mask]-aod_atlid[mask])/aod_atlid[mask]))
else: 
    print('CAMS AOD median=',np.nanmedian(aod_cams[mask]))
    print('ATLID AOD median=',np.nanmedian(aod_atlid[mask]))
    print('CAMS-ATLID median=',np.nanmedian(aod_cams[mask]-aod_atlid[mask]))
    aod_atlid_mask = aod_atlid[mask]
    if len(aod_atlid_mask[aod_atlid_mask==0]) > 0:
        aod_atlid_mask[aod_atlid_mask==0] = np.nan
        print('excludes 0')
    print('CAMS-ATLID/ATLID median=',np.nanmedian((aod_cams[mask]-aod_atlid[mask])/aod_atlid_mask))
print('CAMS,ATLID NMB=',statistics.normalized_mean_bias(aod_cams,aod_atlid))

#fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(20,30),subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0)))
fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(20,30),subplot_kw=dict(projection=ccrs.PlateCarree()))
cmap = plt.colormaps['plasma']
bounds = np.logspace(np.log10(0.02),0,num=10)
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
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False

ax1.coastlines(resolution='110m')#,color='white')
ax1.gridlines()
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

bar = plt.colorbar(im, orientation='vertical',ax=ax1,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('ATLID Particle optical depth / -',fontsize=15)
bar.ax.tick_params(labelsize=15)

ax1.set_title('ATLID integrated AOD '+fmonth+' 2025',fontsize=15)



ax2.set_extent([-180, 180, -89.9, 89.9])
im=ax2.pcolormesh(clons,clats,aod_cams,cmap='plasma',transform=ccrs.PlateCarree(),norm=norm)
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
bar.ax.set_ylabel('CAMS Particle optical depth / -',fontsize=15)
bar.ax.tick_params(labelsize=15)
ax2.set_title('CAMS total AOD',fontsize=15)

ax3.set_extent([-180, 180, -89.9, 89.9])
bounds2 = np.linspace(-vmax,vmax,10)
norm2 = colors.BoundaryNorm(bounds2,ncolors=plt.colormaps['RdYlBu_r'].N,clip=True)
im=ax3.pcolormesh(clons,clats,aod_cams-aod_atlid,cmap='RdYlBu_r',transform=ccrs.PlateCarree(),norm=norm2)
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
bar.ax.set_ylabel('CAMS-ATLID Particle optical depth / -',fontsize=15)
bar.ax.tick_params(labelsize=15)
ax3.set_title('CAMS-ATLID AOD',fontsize=15)

plt.tight_layout()
fig.savefig('figures/global_aod_'+which_aerosol+'_'+str(reso)+'deg_binned_'+mean_or_std+'_'+month+'_2025_co-located_filtered_uncertainty.jpg',bbox_inches='tight')

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

fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5),sharey=False)
ax1.plot(bcc,histc,label='CAMS')
ax1.plot(bca,hista,label='regridded ATLID')

ax2.plot(bca0,hista0,label='original ATLID')
ax3.plot(bca0_,hista0_,label='original ATLID')
ax3.plot(bca_,hista_,label='regridded ATLID')
ax2.set_ylim(1e1,1e7)
ax2.set_yscale('log')
#ax3.set_ylim(1e1,1e7)
#ax3.set_yscale('log')

ax1.set_xlabel('AOD',fontsize=15) 
ax1.set_ylabel('Counts',fontsize=15)
ax2.set_xlabel('AOD',fontsize=15)
ax2.set_ylabel('Counts',fontsize=15)
ax3.set_xlabel('AOD',fontsize=15)
ax3.set_ylabel('Density',fontsize=15)

ax1.tick_params(labelsize=12)
ax2.tick_params(labelsize=12)
ax3.tick_params(labelsize=12)

ax1.set_title(fmonth+' 2025',fontsize=15)
ax1.legend(frameon=False,fontsize=15)
ax2.legend(frameon=False,fontsize=15)
ax3.legend(frameon=False,fontsize=15)
fig.savefig('figures/histograms_CAMS_ATLID_'+str(reso)+'deg_binned_'+mean_or_std+'_'+month+'_2025_co-located_filtered_uncertainty.jpg')

fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5),gridspec_kw={'width_ratios': [2, 1, 1]},sharey=False)
ax1.plot(bcc,histc,label='CAMS')
ax1.plot(bca,hista,label='regridded ATLID')

#CAMS mode
def find_mode(counts,bin_edges):
    # Find the maximum frequency
    peaks, properties = find_peaks(counts)
    sorted_indices = np.argsort(counts[peaks])[::-1]
    nsel = 4 if len(sorted_indices) >= 4 else len(sorted_indices)
        
    top_peaks = peaks[sorted_indices[:nsel]]

    modes = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in top_peaks]
    print("Estimated modes from histogram:", modes)
    return modes,counts[top_peaks]

cmodes,chpeaks = find_mode(histc[:],binsc[:])
amodes,ahpeaks = find_mode(hista[:],binsa[:])

for i,m in enumerate(cmodes):
    ax1.scatter(m,chpeaks[i], color='blue', zorder=5, label=f'Mode ~ {m:.2f}')

for i,m in enumerate(amodes):
    ax1.scatter(m,ahpeaks[i], color='orange', zorder=5, label=f'Mode ~ {m:.2f}')

'''
Cmax_index = np.argmax(histc[1:])
Cmode_estimate = (binsc[1:][Cmax_index] + binsc[1:][Cmax_index + 1]) / 2
print("Estimated mode from histogram:", Cmode_estimate)
ax1.axvline(Cmode_estimate,color='blue',linestyle='dashed', linewidth=2, label=f'Mode ~ {Cmode_estimate:.2f}')

#ATLID mode
Amax_index = np.argmax(hista[1:])
Amode_estimate = (binsa[1:][Amax_index] + binsa[1:][Amax_index + 1]) / 2
print("Estimated mode from histogram:", Amode_estimate)
ax1.axvline(Amode_estimate,color='orange',linestyle='dashed', linewidth=2, label=f'Mode ~ {Amode_estimate:.2f}')
'''

ax2.plot(bca0,hista0,label='original ATLID')
ax3.plot(bca0_,hista0_,label='original ATLID')
ax3.plot(bca_,hista_,label='regridded ATLID')
ax2.set_ylim(1e1,1e7)
ax2.set_yscale('log')
ax1.set_xlim(1e-3,2)
ax1.set_xscale('log')

ax1.set_xlabel('AOD',fontsize=15) 
ax1.set_ylabel('Counts',fontsize=15)
ax2.set_xlabel('AOD',fontsize=15)
ax2.set_ylabel('Counts',fontsize=15)
ax3.set_xlabel('AOD',fontsize=15)
ax3.set_ylabel('Density',fontsize=15)

ax1.tick_params(labelsize=12)
ax2.tick_params(labelsize=12)
ax3.tick_params(labelsize=12)

ax1.set_title(fmonth+' 2025',fontsize=15)
ax1.legend(frameon=False,fontsize=15)
ax2.legend(frameon=False,fontsize=15)
ax3.legend(frameon=False,fontsize=15)
fig.savefig('figures/histograms_CAMS_ATLID_'+str(reso)+'deg_binned_'+mean_or_std+'_'+month+'_2025_co-located_filtered_uncertainty_with_modes.jpg')

