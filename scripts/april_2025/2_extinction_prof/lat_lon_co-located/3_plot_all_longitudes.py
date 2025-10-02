import h5py
import concurrent.futures
import glob
import numpy as np
import cartopy.crs as ccrs
import xarray as xr
from netCDF4 import Dataset
#from pyresample import geometry, kd_tree
from netCDF4 import Dataset
from pylab import *
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import sys
import os
script_path = '/usr/people/wangxu/Desktop/earthcare_scripts/scripts/april_2025/'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), script_path)))

from ectools import ecio
from ectools import ecplot as ecplt
from ectools import colormaps
from plotting_tools import read_h5,ATC_category_colors


def landsea_mean(var):
    cams_lsm = Dataset('/net/pc190625/nobackup_1/users/wangxu/cams_data/landsea_mask.nc')
    lsm = cams_lsm.variables['lsm'][0,0]
    land = np.mean(var[np.where(lsm==1)])
    sea = np.mean(var[np.where(lsm==0)])
    print('not weighted by area')
    return land,sea

month='july'
fmonth='July'
vname = 'extinction_coefficient'
figname = 'extinction_coefficient'
file_name = 'regridded_satellite_total_extinction_coe_2deg_masknan_mean_single_alt_'+month+'_2025.nc'
cams_file = 'regridded_cams_extinction_coe_2deg_mean_'+month+'_2025.nc'

cams_ha = Dataset(cams_file)
cams_h = cams_ha.variables['actual_height'][:]

cams_dir = '/net/pc190625/nobackup_1/users/wangxu/cams_data/'
if figname == 'extinction_coefficient':
    atlid = Dataset(file_name)
    var_atlid = atlid.variables[vname][:,:,:]
    var_zonal_atlid = np.mean(var_atlid,axis=1) #zonal mean
    lon_atlid = atlid.variables['longitude'][:]
    lat_atlid = atlid.variables['latitude'][:]
    atlid_h = atlid.variables['height'][:]
    print(var_zonal_atlid.shape) #(451, 900, 254)

    cams = Dataset(cams_file)
    var_cams = cams.variables[vname+'_cams'][:,:,:]
    var_zonal_cams = np.mean(var_cams,axis=1) #zonal mean
    lon_cams = cams.variables['longitude'][:]
    lat_cams = cams.variables['latitude'][:]
    cams_h = cams.variables['actual_height'][:]
    print(var_cams.shape) #(451, 900, 254)

elif figname == 'ssa_aod':
    atlid = Dataset(file_name)
    aod_atlid = atlid.variables[vname][:]

    cams1 = Dataset('/net/pc190625/nobackup_1/users/wangxu/cams_data/total_aerosol_optical_depth_355nm_dec_2024.nc')
    taod1 = cams1.variables['aod355'][:]
    
    cams2 = Dataset('/net/pc190625/nobackup_1/users/wangxu/cams_data/total_aerosol_optical_depth_550nm_dec_2024.nc')
    taod2 = cams2.variables['aod550'][:]
    aod_550nm = cams2.variables['ssaod550'][:] # Sea salt aerosol
    
    lambda1 = 355
    lambda2 = 550
    angstrom = -(np.log(taod1/taod2)/np.log(lambda2/lambda1))
    aod_355nm = aod_550nm*(355/550)**(angstrom)
    
    aod_cams = np.mean(np.mean(aod_355nm,axis=0),axis=0)
    cams = cams1

    land,sea = landsea_mean(aod_cams)
    print('CAMS land mean=',land,'ocean mean=',sea)
    land,sea = landsea_mean(aod_atlid)
    print('ATLID land mean=',land,'ocean mean=',sea)
    land,sea = landsea_mean(aod_atlid-aod_cams)
    print('ATLID-CAMS land mean=',land,'ocean mean=',sea)

elif figname == 'dust_aod':
    atlid = Dataset(file_name)
    aod_atlid = atlid.variables[vname][:]

    cams1 = Dataset('/net/pc190625/nobackup_1/users/wangxu/cams_data/total_aerosol_optical_depth_355nm_dec_2024.nc')
    taod1 = cams1.variables['aod355'][:]

    cams2 = Dataset('/net/pc190625/nobackup_1/users/wangxu/cams_data/total_aerosol_optical_depth_550nm_dec_2024.nc')
    taod2 = cams2.variables['aod550'][:]

    lambda1 = 355
    lambda2 = 550

    angstrom = -(np.log(taod1/taod2)/np.log(lambda2/lambda1))

    cams3 = Dataset('/net/pc190625/nobackup_1/users/wangxu/cams_data/dust_sulfate_aerosol_optical_depth_dec_2024.nc')
    dust_cams = cams3.variables['duaod550'][:]
    dust_cams = dust_cams*(355/550)**(angstrom)
    aod_cams = np.mean(np.mean(dust_cams,axis=0),axis=0)
    cams = cams1

    land,sea = landsea_mean(aod_cams)
    print('CAMS land mean=',land,'ocean mean=',sea)
    land,sea = landsea_mean(aod_atlid)
    print('ATLID land mean=',land,'ocean mean=',sea)
    land,sea = landsea_mean(aod_atlid-aod_cams)
    print('ATLID-CAMS land mean=',land,'ocean mean=',sea)


#interpolating CAMS to ATLID height
var_cams_interp = np.zeros((var_zonal_atlid.shape[0],var_zonal_atlid.shape[1]))
for i in range(var_zonal_atlid.shape[0]):
    var_cams_interp[i,:] = np.interp(atlid_h,cams_h[i,0,:],var_zonal_cams[i,:])
vmax = 1.e-4

atlid_hpx,lat_atlidpa = np.meshgrid(atlid_h,lat_atlid)
cams_hpx,lat_atlidpc  = np.meshgrid(cams_h[0,0,:],lat_atlid)

fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(8,12))
im=ax1.pcolormesh(lat_atlidpa,atlid_hpx//1000,var_zonal_atlid,cmap=ecplt.colormaps.calipso,norm=matplotlib.colors.LogNorm(vmax=vmax,vmin=vmax/100.))
bar = plt.colorbar(im, orientation='vertical',ax=ax1,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('Extinction coefficient',fontsize=15)
bar.ax.tick_params(labelsize=15)
ax1.set_title('Zonal mean extinction coefficient ATLID '+fmonth+' 2025',fontsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax1.set_ylim(0,20)
ax1.set_ylabel('Altitude / km',fontsize=15)

im=ax2.pcolormesh(lat_atlidpc,cams_h[:,0,:]//1000,var_zonal_cams,cmap=ecplt.colormaps.calipso,norm=matplotlib.colors.LogNorm(vmax=vmax,vmin=vmax/100.))
bar = plt.colorbar(im, orientation='vertical',ax=ax2,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('Extinction coefficient',fontsize=15)
bar.ax.tick_params(labelsize=15)
ax2.set_title('Zonal mean extinction coefficient CAMS '+fmonth+' 2025',fontsize=15)
ax2.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)
ax2.set_ylim(0,20)
ax2.set_ylabel('Altitude / km',fontsize=15)

linthresh = 1e-6
#im=ax3.pcolormesh(lat_atlidpa,atlid_hpx//1000,var_cams_interp-var_zonal_atlid,cmap='RdBu_r',norm=colors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax))
cmap = plt.colormaps['RdBu_r']
bounds = -np.logspace(-6,-4,num=5)
bounds = np.sort(np.append(bounds,-bounds))
print(bounds)
norm = colors.BoundaryNorm(bounds, ncolors=cmap.N, clip=True)
im=ax3.pcolormesh(lat_atlidpa,atlid_hpx//1000,var_cams_interp-var_zonal_atlid,cmap='RdBu_r',norm=norm)
bar = plt.colorbar(im, orientation='vertical',ax=ax3,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('diff Extinction coefficient',fontsize=15)
bar.ax.tick_params(labelsize=15)
ax3.set_title('CAMS-ATLID',fontsize=15)
ax3.tick_params(axis='x', labelsize=15)
ax3.tick_params(axis='y', labelsize=15)
ax3.set_ylim(0,20)
ax3.set_ylabel('Altitude / km',fontsize=15)

plt.tight_layout()
fig.savefig('zonal_nanmean_extinction_coefficient_'+figname+'_2deg_masknan_mean_'+month+'_2025.jpg',bbox_inches='tight')

lon_bins = np.arange(-180.1, 180.1, 2)
lat_cams,cams_hp = np.meshgrid(lat_cams,cams_h[0,0,:])

for ilon in range(var_atlid.shape[1]):
    #interpolating CAMS to ATLID height
    var_cams_interp2 = np.zeros((var_zonal_atlid.shape[1],var_zonal_atlid.shape[0]))
    for ilat in range(var_zonal_atlid.shape[0]):
        var_cams_interp2[:,ilat] = np.interp(atlid_h,cams_h[ilat,ilon,:],var_cams[ilat,ilon,:])

    vmax = 1.e-4

    fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(8,12))
    im=ax1.pcolormesh(lat_atlidpa.transpose(),atlid_hpx.transpose()//1000,var_atlid[:,ilon,:].transpose(),cmap=ecplt.colormaps.calipso,norm=matplotlib.colors.LogNorm(vmax=vmax,vmin=vmax/100.))    

    bar = plt.colorbar(im, orientation='vertical',ax=ax1,shrink=0.7, pad=0.1)
    bar.ax.set_ylabel('Extinction coefficient',fontsize=15)
    bar.ax.tick_params(labelsize=15)
    ax1.set_title('Extinction coefficient ATLID at '+str(lon_atlid[ilon])+' '+fmonth+' 2025',fontsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.set_ylim(0,20)

    im=ax2.pcolormesh(lat_cams,cams_h[:,ilon,:].transpose()//1000,var_cams[:,ilon,:].transpose(),cmap=ecplt.colormaps.calipso,norm=matplotlib.colors.LogNorm(vmax=vmax,vmin=vmax/100.))
    bar = plt.colorbar(im, orientation='vertical',ax=ax2,shrink=0.7, pad=0.1)
    bar.ax.set_ylabel('Extinction coefficient',fontsize=15)
    bar.ax.tick_params(labelsize=15)
    ax2.set_title('Extinction coefficient CAMS at '+str(lon_cams[ilon])+' '+fmonth+' 2025',fontsize=15)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.set_ylim(0,20)

    #im=ax3.pcolormesh(lat_atlidpa.transpose(),atlid_hpx.transpose()//1000,var_cams_interp2-var_atlid[:,ilon,:].transpose(),cmap='RdBu_r',norm=colors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax))
    cmap = plt.colormaps['RdBu_r']
    im=ax3.pcolormesh(lat_atlidpa.transpose(),atlid_hpx.transpose()//1000,var_cams_interp2-var_atlid[:,ilon,:].transpose(),cmap='RdBu_r',norm=norm)
    bar = plt.colorbar(im, orientation='vertical',ax=ax3,shrink=0.7, pad=0.1)
    bar.ax.set_ylabel('diff Extinction coefficient interpolated',fontsize=15)
    bar.ax.tick_params(labelsize=15)
    ax3.set_title('CAMS-ATLID',fontsize=15)
    ax3.tick_params(axis='x', labelsize=15)
    ax3.tick_params(axis='y', labelsize=15)
    ax3.set_ylim(0,20)

    plt.tight_layout()
    fig.savefig('all_lons/extinction_coefficient_'+figname+'_at_'+str(lon_bins[ilon])+'_0.4deg_masknan_'+month+'_2025.jpg',bbox_inches='tight')

