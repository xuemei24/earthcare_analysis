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


month = 'july'
mean_or_std = 'mean'

cams_dir = '/net/pc190625/nobackup_1/users/wangxu/cams_data/'
cams_mmr_dir = '/net/pc200254/nobackup/users/wangxu/cams_data/aerosol_mmr/'+month+'/'
#cams_mmr_dir = '/net/pc190625/nobackup_1/users/wangxu/cams_data/aerosol_mmr/'

f = xr.open_dataset(cams_mmr_dir+'percentage_mmr_'+month+'_2025.nc')
dust = f['dust_0.03-0.55']+f['dust_0.55-0.9']+f['dust_0.9-20']
dust_alike = dust+f['hydrophilic_black_carbon']+f['hydrophobic_black_carbon']
ssa  = f['sea_salt_0.03-0.5']+f['sea_salt_0.5-5']+f['sea_salt_5-20']

cams = Dataset(cams_dir+'total_aerosol_optical_depth_355nm_'+month+'_2025.nc')
#aod_cams = np.mean(np.mean(cams.variables['aod355'][:],axis=0),axis=0)
aod_o_cams = cams.variables['aod355'][::3]
aod_cams = np.zeros((aod_o_cams.shape))

clwc0 = Dataset(cams_dir+'lwc/specific_cloud_liquid_water_content_'+month+'_2025_0.nc')
clwc3 = Dataset(cams_dir+'lwc/specific_cloud_liquid_water_content_'+month+'_2025_3.nc')
clwc6 = Dataset(cams_dir+'lwc/specific_cloud_liquid_water_content_'+month+'_2025_6.nc')
clwc9 = Dataset(cams_dir+'lwc/specific_cloud_liquid_water_content_'+month+'_2025_9.nc')

#delete columns where there are liquid clouds
def where_cloudy(cloud,data):
    print('before',data.shape)
    mask = (cloud>=0.0001).any(axis=1)
    print('mask_expanded.shape',mask.shape)
    data = np.where(mask, np.nan, data)
    print('after',data.shape)
    return data

aod_cams[0] = where_cloudy(clwc0.variables['clwc'][0,:],aod_o_cams[0])
aod_cams[1] = where_cloudy(clwc3.variables['clwc'][0,:],aod_o_cams[1])
aod_cams[2] = where_cloudy(clwc6.variables['clwc'][0,:],aod_o_cams[2])
aod_cams[3] = where_cloudy(clwc9.variables['clwc'][0,:],aod_o_cams[3])

print(aod_cams[aod_cams>0])
aod_cams_ssa = np.nanmean(np.nanmean(np.where(ssa>0.95,aod_cams,np.nan),axis=0),axis=0)
aod_cams_dust = np.nanmean(np.nanmean(np.where(dust_alike>0.95,aod_cams,np.nan),axis=0),axis=0)

print('aod_cams_ssa.shape',aod_cams_ssa.shape)
lat_bins = np.arange(-90.1, 90.1, 2)
lon_bins = np.arange(-180.1, 180.1, 2)

lat_cams_0 = cams.variables['latitude'][:]
lon_cams_0 = cams.variables['longitude'][:]
ilon = np.where(lon_cams_0>=180)
lon_cams_0[ilon] = lon_cams_0[ilon]-360.
lon_cams_1,lat_cams_1 = np.meshgrid(lon_cams_0,lat_cams_0)
lon_cams_1 = lon_cams_1.flatten()
lat_cams_1 = lat_cams_1.flatten()

def regrid(aod_to_regrid):
    aod_to_regrid_1 = aod_to_regrid.flatten()

    mask = ~np.isnan(aod_to_regrid_1)
    aod_to_regrid2 = aod_to_regrid_1[mask]
    lat_cams = lat_cams_1[mask]
    lon_cams = lon_cams_1[mask]

    print(aod_to_regrid2.shape,lat_cams.shape,lon_cams.shape)
    stat, x_edge, y_edge, _ = binned_statistic_2d(
        lat_cams, lon_cams, aod_to_regrid2, statistic=mean_or_std, bins=[lat_bins, lon_bins])
    return stat

# Replace NaN with zeros (or another value if necessary)
#stat = np.nan_to_num(stat)

aod_cams_ssa3 = regrid(aod_cams_ssa)
aod_cams_dust3 = regrid(aod_cams_dust)
#aod_cams = np.where(aod_cams>0,aod_cams,np.nan)
print(np.isnan(aod_cams_ssa3).sum(), "grid cells have missing data")
print(np.isnan(aod_cams_dust3).sum(), "grid cells have missing data")

vmax = aod_cams_ssa3.max()
vmax = 1
print('aod_cams.max()=',aod_cams_ssa3.max())

aero_types = ['sea_salt','dust_alike']
fig_names = ['Sea Salt','Dust & Smoke']
faods = ['2025_'+month+'_aod_ssa.txt','2025_'+month+'_aod_dust_alike.txt']
caods = [aod_cams_ssa3,aod_cams_dust3]
for aero_type,fig_name, faod,caod in zip(aero_types,fig_names,faods,caods):
    tdf = pd.read_csv("/usr/people/wangxu/Desktop/earthcare_scripts/scripts/april_2025/1_global_aod/2025_"+month+"_aod.txt", delimiter=",")
    all_taod = tdf['aod355nm'].values
    all_tlat = tdf['# latitude'].values
    all_tlon = tdf['longitude'].values

    print('max and min of latitude before mask',np.nanmax(all_tlat),np.nanmin(all_tlat))
    mask = ~np.isnan(all_taod)
    all_taod = all_taod[mask]
    all_tlat = all_tlat[mask]
    all_tlon = all_tlon[mask]

    # Compute 2D histogram for the mean wind
    stat, x_edge, y_edge, _ = binned_statistic_2d(
        all_tlat, all_tlon, all_taod, statistic=mean_or_std, bins=[lat_bins, lon_bins])
    taod = stat

    df = pd.read_csv("/usr/people/wangxu/Desktop/earthcare_scripts/scripts/april_2025/1_global_aod/"+faod, delimiter=",")

    #2 deg resolution
    all_aod = df['aod355nm'].values
    all_lat = df['# latitude'].values
    all_lon = df['longitude'].values
   
    print('max and min of latitude before mask',np.nanmax(all_lat),np.nanmin(all_lat))
    mask = ~np.isnan(all_aod)
    all_aod = all_aod[mask]
    all_lat = all_lat[mask]
    all_lon = all_lon[mask]
   
   
    # Compute 2D histogram for the mean wind
    stat, x_edge, y_edge, _ = binned_statistic_2d(
        all_lat, all_lon, all_aod, statistic=mean_or_std, bins=[lat_bins, lon_bins])
   
    # Replace NaN with zeros (or another value if necessary)
    #stat = np.nan_to_num(stat)
    
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
   
    clons,clats = np.meshgrid(lon_centers,lat_centers)
    nan_percentage = np.isnan(stat).sum() / stat.size * 100
    print(f"Percentage of NaNs in regridded data: {nan_percentage:.2f}%")
   
    #plot the figures
    aod_atlid = stat
   
    aod_atlid = np.where(aod_atlid/taod>0.95,taod,np.nan)
    def landsea_mean(var):
        cams_lsm = Dataset('/net/pc190625/nobackup_1/users/wangxu/cams_data/landsea_mask.nc')
        lsm0 = cams_lsm.variables['lsm'][0,0]
 
        stat, x_edge, y_edge, _ = binned_statistic_2d(
        lat_cams_1, lon_cams_1, lsm0.flatten(), statistic='mean', bins=[lat_bins, lon_bins])
 
        lsm = np.round(stat)
        print(lsm.shape,var.shape)
 
        land = np.nanmean(var[np.where(lsm==1)])
        sea = np.nanmean(var[np.where(lsm==0)])
        print('not weighted by area')
        return land,sea

    aland,asea = landsea_mean(aod_atlid)
    print('ATLID land=',aland,'sea=',asea)
    cland,csea = landsea_mean(caod)
    print('CAMS land=',cland,'sea=',csea)
    acland,acsea = landsea_mean(caod-aod_atlid)
    print('CAMS-ATLID land=',acland,'sea=',acsea)
   
    print("Satellite Data Lat Range:", np.min(all_lat), np.max(all_lat))
    print("Satellite Data Lon Range:", np.min(all_lon), np.max(all_lon))
    print("Grid Lat Range:", np.min(clats), np.max(clats))
    print("Grid Lon Range:", np.min(clons), np.max(clons))
    
    print('CAMS AOD.shape',caod.shape,'ATLID AOD.shape',aod_atlid.shape)
    mask = ~np.isnan(caod) & ~np.isnan(aod_atlid)
    print('CAMS AOD mean=',np.nanmean(caod[mask]))
    print('ATLID AOD mean=',np.nanmean(aod_atlid[mask]))
    print('CAMS-ATLID mean=',np.nanmean(caod[mask]-aod_atlid[mask]))

    #fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(20,30),subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=0)))
    fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(20,30),subplot_kw=dict(projection=ccrs.PlateCarree()))
    #plt.figure(figsize=(6,3))
    #ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax1.set_extent([-180, 180, -89.9, 89.9])#,crs=ccrs.PlateCarree(central_longitude=180))
    all_lon = np.where(clons>0,clons,clons+360)
    im=ax1.pcolormesh(all_lon,clats,aod_atlid,cmap='plasma',transform=ccrs.PlateCarree(),norm=colors.LogNorm(vmin=1e-2, vmax=1))
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
    bar.ax.set_ylabel('ATLID '+fig_name+' Particle optical depth / -',fontsize=15)
    bar.ax.tick_params(labelsize=15)
    
    ax1.set_title('ATLID integrated '+fig_name+' AOD',fontsize=15)
   
   
   
    ax2.set_extent([-180, 180, -89.9, 89.9])
    im=ax2.pcolormesh(clons,clats,caod,cmap='plasma',transform=ccrs.PlateCarree(),norm=colors.LogNorm(vmin=1e-2, vmax=1))
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
    bar.ax.set_ylabel('CAMS '+fig_name+' Particle optical depth / -',fontsize=15)
    bar.ax.tick_params(labelsize=15)
    ax2.set_title('CAMS '+fig_name+' AOD',fontsize=15)
   
    ax3.set_extent([-180, 180, -89.9, 89.9])
    im=ax3.pcolormesh(clons,clats,caod-aod_atlid,cmap='RdBu_r',transform=ccrs.PlateCarree(),vmax=vmax,vmin=-vmax)
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
    bar.ax.set_ylabel('CAMS-ATLID '+fig_name+' Particle optical depth / -',fontsize=15)
    bar.ax.tick_params(labelsize=15)
    ax3.set_title('CAMS-ATLID '+fig_name+' AOD',fontsize=15)
   
    plt.tight_layout()
    fig.savefig('global_aod_'+aero_type+'_2deg_binned_'+mean_or_std+'_assign_totalAOD_'+month+'_2025_trimedges_filterCAMScloud_TC.jpg',bbox_inches='tight')
   
    nbins = 150
    binsc = np.linspace(0,np.nanmax(caod),nbins)
    caod_clean = caod[~np.isnan(caod)]
    histc,binsc = np.histogram(caod_clean,bins=binsc,density=False)
    bcc = 0.5*(binsc[1:] + binsc[:-1])
   
    binsa = np.linspace(0,np.nanmax(aod_atlid),nbins)
    hista,binsa = np.histogram(aod_atlid,bins=binsa,density=False)
    bca = 0.5*(binsa[1:] + binsa[:-1])
   
    binsa0 = np.linspace(0,np.nanmax(all_aod),nbins)
    hista0,binsa0 = np.histogram(all_aod,bins=binsa0,density=False)
    bca0 = 0.5*(binsa0[1:] + binsa0[:-1])
   
    binsa_ = np.linspace(0,np.nanmax(aod_atlid),nbins)
    hista_,binsa_ = np.histogram(aod_atlid,bins=binsa_,density=True)
    bca_ = 0.5*(binsa_[1:] + binsa_[:-1])
   
    binsa0_ = np.linspace(0,np.nanmax(all_aod),nbins)
    hista0_,binsa0_ = np.histogram(all_aod,bins=binsa0_,density=True)
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
   
    ax1.set_xlabel(fig_name+' AOD',fontsize=15) 
    ax1.set_ylabel('Counts',fontsize=15)
    ax2.set_xlabel(fig_name+' AOD',fontsize=15)
    ax2.set_ylabel('Counts',fontsize=15)
    ax3.set_xlabel(fig_name+' AOD',fontsize=15)
    ax3.set_ylabel('Density',fontsize=15)
   
    ax1.tick_params(labelsize=12)
    ax2.tick_params(labelsize=12)
    ax3.tick_params(labelsize=12)
   
    ax1.legend(frameon=False,fontsize=15)
    ax2.legend(frameon=False,fontsize=15)
    ax3.legend(frameon=False,fontsize=15)
    fig.savefig('global_'+aero_type+'_histograms_CAMS_ATLID_2deg_binned_'+mean_or_std+'_assign_totalAOD_'+month+'_2025_trimedges_filterCAMScloud_TC.jpg')
