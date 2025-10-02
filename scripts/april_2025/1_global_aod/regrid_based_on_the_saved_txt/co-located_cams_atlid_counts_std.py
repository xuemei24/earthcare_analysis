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

script_path = '/usr/people/wangxu/Desktop/earthcare_scripts/scripts/april_2025/'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), script_path)))
from plotting_tools import statistics

# Load your dataset (assuming it's a CSV with 'lat', 'lon', 'aod')
file_dir = '/net/pc190625/nobackup_1/users/wangxu/cams_data/'
month = 'april'
fmonth = 'April'
print('Month=',month)
df = pd.read_csv(file_dir+"2025_"+month+"_cams_atlid_co-located_aod.txt", delimiter=",")

#simple_classification
which_aerosol='total'
#which_aerosol='sea_salt'
#which_aerosol='dust'
# Define a function that reads data from an HDF5 file

mean_or_std = 'std'

#2 deg resolution
a_aod = df['aod355nm_atlid'].values
c_aod = df['aod355nm_cams'].values
all_lat = df['# latitude'].values
all_lon = df['longitude'].values

print('max and min of latitude before mask',np.nanmax(all_lat),np.nanmin(all_lat))
mask = ~np.isnan(a_aod) & ~np.isnan(c_aod)
a_aod = a_aod[mask]
c_aod = c_aod[mask]
all_lat = all_lat[mask]
all_lon = all_lon[mask]
all_lon2 = all_lon.copy()

reso = 2
lat_bins = np.arange(-90., 90.+reso, reso)
lon_bins = np.arange(-180, 180.+reso, reso)

# Compute 2D histogram for the mean wind
aod_atlid, x_edge, y_edge, _ = binned_statistic_2d(
    all_lat, all_lon, a_aod, statistic=mean_or_std, bins=[lat_bins, lon_bins])

aod_cams, x_edge, y_edge, _ = binned_statistic_2d(
    all_lat, all_lon, c_aod, statistic=mean_or_std, bins=[lat_bins, lon_bins])

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
    cams_lsm = Dataset('/net/pc190625/nobackup_1/users/wangxu/cams_data/landsea_mask.nc')
    lsm0 = cams_lsm.variables['lsm'][0,0]
    lat_cams_1 = cams_lsm.variables['latitude']
    lon_cams_1 = cams_lsm.variables['longitude']
    lon_cams_1,lat_cams_1=np.meshgrid(lon_cams_1,lat_cams_1)
    print(lon_cams_1.shape)

    stat, x_edge, y_edge, _ = binned_statistic_2d(
    lat_cams_1.flatten(),lon_cams_1.flatten(), lsm0.flatten(), statistic='mean', bins=[lat_bins, lon_bins])

    lsm = np.round(stat)
    print(lsm.shape,var.shape)

    land = np.nanmean(var[np.where(lsm==1)])
    sea = np.nanmean(var[np.where(lsm==0)])
    print('not weighted by area')
    return land,sea

aland,asea = landsea_mean(aod_atlid)
print('ATLID land=',aland,'sea=',asea)
cland,csea = landsea_mean(aod_cams)
print('CAMS land=',cland,'sea=',csea)


mask = (~np.isnan(aod_cams)) & (~np.isnan(aod_atlid))
print('CAMS AOD mean=',np.nanmean(aod_cams[mask]))
print('ATLID AOD mean=',np.nanmean(aod_atlid[mask]))
print('CAMS-ATLID mean=',np.nanmean(aod_cams[mask]-aod_atlid[mask]))
print('CAMS-ATLID/ATLID mean=',np.nanmean((aod_cams[mask]-aod_atlid[mask])/aod_atlid[mask]))
print('CAMS,ATLID NMB=',statistics.normalized_mean_bias(aod_cams,aod_atlid))

print('CAMS AOD median=',np.nanmedian(aod_cams[mask]))
print('ATLID AOD median=',np.nanmedian(aod_atlid[mask]))
print('CAMS-ATLID median=',np.nanmedian(aod_cams[mask]-aod_atlid[mask]))
print('CAMS-ATLID/ATLID median=',np.nanmedian((aod_cams[mask]-aod_atlid[mask])/aod_atlid[mask]))

#fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(20,30),subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0)))
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(20,20),subplot_kw=dict(projection=ccrs.PlateCarree()))
cmap = plt.colormaps['plasma']
bounds = np.logspace(-2,0,num=10)
norm = colors.BoundaryNorm(bounds, ncolors=cmap.N, clip=True)
#plt.figure(figsize=(6,3))
#ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax1.set_extent([-180, 180, -89.9, 89.9])#,crs=ccrs.PlateCarree(central_longitude=180))
all_lon = np.where(clons>0,clons,clons+360)
im=ax1.pcolormesh(all_lon,clats,aod_atlid,cmap='plasma',transform=ccrs.PlateCarree())#,norm=norm)
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
im=ax2.pcolormesh(clons,clats,aod_cams,cmap='plasma',transform=ccrs.PlateCarree())#,norm=norm)
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

plt.tight_layout()
fig.savefig('global_aod_'+which_aerosol+'_'+str(reso)+'deg_binned_counts_'+mean_or_std+'_'+month+'_2025_co-located.jpg',bbox_inches='tight')


################# Histograms ##################
def plot_hist(data1,data2,reso,mean_or_std,month): #data1 = aod_cams, data2 = aod_atlid
    nbins = 150
    binsc = np.linspace(0,np.nanmax(data1.flatten()),nbins)
    histc,binsc = np.histogram(data1,bins=binsc,density=False)
    bcc = 0.5*(binsc[1:] + binsc[:-1])
    
    binsa = np.linspace(0,np.nanmax(data2),nbins)
    hista,binsa = np.histogram(data2,bins=binsa,density=False)
    bca = 0.5*(binsa[1:] + binsa[:-1])
    
    fig,ax1 = plt.subplots(1,figsize=(7.5,5),sharey=False)
    ax1.plot(bcc,histc,label='CAMS')
    ax1.plot(bca,hista,label='regridded ATLID')
    
    ax1.set_xlabel('AOD '+mean_or_std,fontsize=15)
    ax1.set_ylabel('Counts',fontsize=15)
   
    ax1.tick_params(labelsize=12)
    
    ax1.set_title(fmonth+' 2025',fontsize=15)
    ax1.legend(frameon=False,fontsize=15)
    fig.savefig('histograms_CAMS_ATLID_'+str(reso)+'deg_binned_'+mean_or_std+'_'+month+'_2025_co-located.jpg')
    
    fig,ax1 = plt.subplots(1,figsize=(7.5,5),sharey=False)
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
   
    ax1.set_xscale('log')
    
    ax1.set_xlabel('AOD '+mean_or_std,fontsize=15) 
    ax1.set_ylabel('Counts',fontsize=15)
    ax1.tick_params(labelsize=12)
    
    ax1.set_title(fmonth+' 2025',fontsize=15)
    ax1.legend(frameon=False,fontsize=15)
    fig.savefig('histograms_CAMS_ATLID_'+str(reso)+'deg_binned_'+mean_or_std+'_'+month+'_2025_co-located_with_modes.jpg')

plot_hist(aod_cams,aod_atlid,reso,mean_or_std,month)



def binned_values_2d(x, y, values, bins):
    """
    Similar to scipy.stats.binned_statistic_2d but returns
    all original values that fall into each grid.

    Parameters
    ----------
    x, y : array-like
        Coordinates of the points (e.g., latitude, longitude).
    values : array-like
        Values to bin (e.g., AOD).
    bins : [x_bins, y_bins]
        Bin edges along x and y (like in binned_statistic_2d).

    Returns
    -------
    values_in_bin : dict
        Dictionary with keys (i, j) for grid indices, values are lists of original points.
    x_edges, y_edges : arrays
        Bin edges along x and y.
    binnumber : array
        Bin index for each input value (like binned_statistic_2d).
    """
    # Call binned_statistic_2d to get bin numbers
    print(x, y, values,bins)
    _, x_edges, y_edges, binnumber = binned_statistic_2d(
        x, y, values, statistic='count', bins=bins
    )

    n_x = len(x_edges) - 1
    n_y = len(y_edges) - 1

    # Initialize dict of lists
    values_in_bin = {(i, j): [] for i in range(n_x) for j in range(n_y)}

    # Convert binnumber → (i, j)
    x_idx = (binnumber - 1) // n_y
    y_idx = (binnumber - 1) % n_y

    # Assign values into bins
    for val, i, j in zip(values, x_idx, y_idx):
        if 0 <= i < n_x and 0 <= j < n_y:
            values_in_bin[(i, j)].append(val)

    return values_in_bin, x_edges, y_edges, binnumber

Aaod_bin_bins,x_edges,y_edges,binnumber = binned_values_2d(
        all_lat, all_lon2, a_aod, bins=[lat_bins, lon_bins])

Caod_bin_bins,x_edges,y_edges,binnumber = binned_values_2d(
        all_lat, all_lon2, c_aod, bins=[lat_bins, lon_bins])

def fraction_within_1std(original_data,aod_per_bin): #data = a_aod or c_aod
    mean, x_edge, y_edge, _ = binned_statistic_2d(
        all_lat, all_lon2, original_data, statistic='mean', bins=[lat_bins, lon_bins])
   
    std, x_edge, y_edge, _ = binned_statistic_2d(
        all_lat, all_lon2, original_data, statistic='std', bins=[lat_bins, lon_bins])
   
    count, x_edge, y_edge, _ = binned_statistic_2d(
        all_lat, all_lon2, original_data, statistic='count', bins=[lat_bins, lon_bins])
   
    n_lat, n_lon = mean.shape
    frac_within_1sigma = np.full((n_lat, n_lon), np.nan)
    # Loop through grid cells
    for (i, j), vals in aod_per_bin.items():
        if len(vals) > 0 and not np.isnan(mean[i, j]) and not np.isnan(std[i, j]):
            mu = mean[i, j]
            sigma = std[i, j]
            N = count[i, j]
            
            # Convert list -> array for easier masking
            arr = np.array(vals)
 
            # Count how many fall within [mu - sigma, mu + sigma]
            inside = np.sum((arr >= mu - sigma) & (arr <= mu + sigma))
            
            # Fraction (percentage if you want multiply by 100)
            frac_within_1sigma[i, j] = inside / N

    # Example: fraction for grid (10, 50)
    print("Fraction in (10,50):", frac_within_1sigma)
    print(len(frac_within_1sigma[frac_within_1sigma>0]))
    return frac_within_1sigma

Afrac = fraction_within_1std(a_aod,Aaod_bin_bins)
Cfrac = fraction_within_1std(c_aod,Caod_bin_bins)
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(20,20),subplot_kw=dict(projection=ccrs.PlateCarree()))
cmap = plt.colormaps['jet']
bounds = np.linspace(0,1,num=11)
norm = colors.BoundaryNorm(bounds, ncolors=cmap.N, clip=True)
#plt.figure(figsize=(6,3))
#ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax1.set_extent([-180, 180, -90, 90])#,crs=ccrs.PlateCarree(central_longitude=180))
all_lon = np.where(clons>0,clons,clons+360)
im=ax1.pcolormesh(all_lon,clats,Afrac,cmap=cmap,transform=ccrs.PlateCarree(),norm=norm)
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
bar.ax.set_ylabel('Fraction / -',fontsize=15)
bar.ax.tick_params(labelsize=15)

ax1.set_title('Original fraction of ATLID within 1std per grid '+fmonth+' 2025',fontsize=15)

im=ax2.pcolormesh(all_lon,clats,Cfrac,cmap=cmap,transform=ccrs.PlateCarree(),norm=norm)
gl = ax2.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
             linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False

ax2.coastlines(resolution='110m')#,color='white')
ax2.gridlines()
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

bar = plt.colorbar(im, orientation='vertical',ax=ax2,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('Fraction / -',fontsize=15)
bar.ax.tick_params(labelsize=15)

ax2.set_title('Original fraction of CAMS within 1std per grid '+fmonth+' 2025',fontsize=15)

fig.savefig('global_aod_fraction_of_data_within_1sigma_per_grid_'+str(reso)+'deg_'+month+'_2025.jpg',bbox_inches='tight')

