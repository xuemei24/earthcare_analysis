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
from pylab import *
import sys
import pandas as pd
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, BoundaryNorm


mean_or_std = 'mean'

cams_dir = '/net/pc190625/nobackup_1/users/wangxu/cams_data/'

f = xr.open_dataset(cams_dir+'aerosol_mmr/percentage_mmr_april_2025.nc')
ssa  = f['sea_salt_0.03-0.5']+f['sea_salt_0.5-5']+f['sea_salt_5-20']            #1
dust = f['dust_0.3-0.55']+f['dust_0.55-0.9']+f['dust_0.9-20']
dust_alike = dust+f['hydrophilic_black_carbon']+f['hydrophobic_black_carbon']   #2

ammonium = f['ammonium']                                                        #3
organic = f['anthropogenic_secondary_organic']+f['biogenic_secondary']+f['hydrophobic_organic_matter']+f['hydrophilic_organic_matter']  #4
nitrate = f['nitrate_fine_mode']+f['nitrate_coarse_mode']                       #5
sulfate = f['sulfate']                                                          #6

lat = f['latitude']
lon = f['longitude']
lon = np.where(lon<=180,lon,lon-360)
clon,clat = np.meshgrid(lon,lat)

final_composition = np.empty((4,60,451,900))
final_composition[:] = np.nan
final_composition = np.where(ssa>0.95,1,final_composition)
final_composition = np.where(dust_alike>0.95,2,final_composition)
final_composition = np.where(ammonium>0.95,3,final_composition)
final_composition = np.where(organic>0.95,4,final_composition)
final_composition = np.where(nitrate>0.95,5,final_composition)
final_composition = np.where(sulfate>0.95,6,final_composition)

flags = [1,2,3,4,5,6]
labels = ['Sea Salt','Dust & Smoke','Ammonium','Organic','Nitrate','Sulfate']
colors = ['cyan','brown','blue','green','orange','purple'] 
cmap = ListedColormap(colors)
boundaries = np.arange(0.5, 7.5, 1)  # [0.5, 1.5, ..., 6.5]
norm = BoundaryNorm(boundaries, cmap.N)

for i in range(ssa.shape[0]):
    for j in range(ssa.shape[1]):
        fig,ax1=plt.subplots(1,figsize=(20,12),subplot_kw=dict(projection=ccrs.PlateCarree()))
        #plt.figure(figsize=(6,3))
        #ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        ax1.set_extent([-180, 180, -89.9, 89.9])#,crs=ccrs.PlateCarree(central_longitude=180))
        im=ax1.pcolormesh(clon,clat,final_composition[i,j],cmap=cmap,transform=ccrs.PlateCarree(),norm=norm)
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
        bar.ax.set_ylabel('Aerosol species',fontsize=15)
        bar.ax.tick_params(labelsize=15)
        
        ax1.set_title('Dominant aerosol species per grid',fontsize=15)
        
        
        plt.tight_layout()
        fig.savefig('dominant_species/global_dominant_aerosol_species_per_grid_'+str(i)+'_'+str(j)+'_april_2025.jpg',bbox_inches='tight')



final_composition = np.empty((451,900))
final_composition[:] = np.nan
final_composition = np.where(np.nanmean(np.nanmean(ssa,axis=0),axis=0)>0.95,1,final_composition)
final_composition = np.where(np.nanmean(np.nanmean(dust_alike,axis=0),axis=0)>0.95,2,final_composition)
final_composition = np.where(np.nanmean(np.nanmean(ammonium,axis=0),axis=0)>0.95,3,final_composition)
final_composition = np.where(np.nanmean(np.nanmean(organic,axis=0),axis=0)>0.95,4,final_composition)
final_composition = np.where(np.nanmean(np.nanmean(nitrate,axis=0),axis=0)>0.95,5,final_composition)
final_composition = np.where(np.nanmean(np.nanmean(sulfate,axis=0),axis=0)>0.95,6,final_composition)

fig,ax1=plt.subplots(1,figsize=(20,12),subplot_kw=dict(projection=ccrs.PlateCarree()))
#plt.figure(figsize=(6,3))
#ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax1.set_extent([-180, 180, -89.9, 89.9])#,crs=ccrs.PlateCarree(central_longitude=180))
im=ax1.pcolormesh(clon,clat,final_composition,cmap=cmap,transform=ccrs.PlateCarree(),norm=norm)
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
bar.ax.set_ylabel('Aerosol species',fontsize=15)
bar.ax.tick_params(labelsize=15)

ax1.set_title('Dominant aerosol species per grid',fontsize=15)

plt.tight_layout()
fig.savefig('dominant_species/global_dominant_aerosol_species_per_grid_monthly_mean_april_2025.jpg',bbox_inches='tight')

