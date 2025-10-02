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
import sys

def landsea_mean(var):
    cams_lsm = Dataset('/net/pc190625/nobackup_1/users/wangxu/cams_data/landsea_mask.nc')
    lsm = cams_lsm.variables['lsm'][0,0]
    land = np.mean(var[np.where(lsm==1)])
    sea = np.mean(var[np.where(lsm==0)])
    print('not weighted by area')
    return land,sea

month='june'
fmonth = 'June'
vname = 'extinction_coefficient'
file_name = 'regridded_satellite_total_extinction_coe_2deg_masknan_mean_single_alt_'+month+'_2025.nc'
figname = 'extinction_coefficient'

cams_dir = '/net/pc190625/nobackup_1/users/wangxu/cams_data/'
if figname == 'extinction_coefficient':
    atlid = Dataset(file_name)
    var_atlid = atlid.variables[vname][:,:,:]
    #var_atlid = np.where(var_atlid<1.e-3,var_atlid,np.nan)
    var_zonal_atlid = np.mean(var_atlid,axis=1) #zonal mean
    lon_atlid = atlid.variables['longitude'][:]
    lat_atlid = atlid.variables['latitude'][:]
    atlid_h = atlid.variables['height'][:] 
    print(var_atlid.shape) #(451, 900, 254)
    print(var_zonal_atlid.shape) #(451,253)

    cams0 = Dataset(cams_dir+'extinction_355nm_multi/aerosol_extinction_coe_355nm_'+month+'2025_0.nc')
    cams3 = Dataset(cams_dir+'extinction_355nm_multi/aerosol_extinction_coe_355nm_'+month+'2025_3.nc')
    cams6 = Dataset(cams_dir+'extinction_355nm_multi/aerosol_extinction_coe_355nm_'+month+'2025_6.nc')
    cams9 = Dataset(cams_dir+'extinction_355nm_multi/aerosol_extinction_coe_355nm_'+month+'2025_9.nc')

    clwc0 = Dataset(cams_dir+'lwc/specific_cloud_liquid_water_content_'+month+'_2025_0.nc')
    clwc3 = Dataset(cams_dir+'lwc/specific_cloud_liquid_water_content_'+month+'_2025_3.nc')
    clwc6 = Dataset(cams_dir+'lwc/specific_cloud_liquid_water_content_'+month+'_2025_6.nc')
    clwc9 = Dataset(cams_dir+'lwc/specific_cloud_liquid_water_content_'+month+'_2025_9.nc')

    #delete columns where there are liquid clouds
    def where_cloudy(cloud,data):
        print('before',data.shape)
        mask = (cloud>=0.0001).any(axis=1)
        mask_expanded = np.expand_dims(mask, axis=1) #Expand mask to match 'data' shape for height (axis=1)
        mask_final = np.repeat(mask_expanded, repeats=137, axis=1)
        print('mask_expanded.shape',mask_expanded.shape)
        data[:, :, :, :] = np.where(mask_final, np.nan, data)
        print('after',data.shape)
        return data

    print(clwc3.variables['clwc'][0].shape,cams3.variables['aerext355'][0].shape)
    v_cams0 = np.nanmean(where_cloudy(clwc0.variables['clwc'][0,:],cams0.variables['aerext355'][0]),axis=0)
    print('v_cams0')
    v_cams3 = np.nanmean(where_cloudy(clwc3.variables['clwc'][0,:],cams3.variables['aerext355'][0]),axis=0)
    print('v_cams0.1')
    v_cams6 = np.nanmean(where_cloudy(clwc6.variables['clwc'][0,:],cams6.variables['aerext355'][0]),axis=0)
    print('v_cams0.2')
    v_cams9 = np.nanmean(where_cloudy(clwc9.variables['clwc'][0,:],cams9.variables['aerext355'][0]),axis=0)
    print('v_cams3')

    var_cams = (v_cams0+v_cams3+v_cams6+v_cams9)/4.
    print('v_cams6')

    #now start regridding to 2 deg
    lat_bins = np.arange(-90.1, 90.1, 2)
    lon_bins = np.arange(-180.1, 180.1, 2)

    regridded_data = np.zeros((len(lat_bins)-1,len(lon_bins)-1, var_cams.shape[0]))
    regridded_height = np.zeros((len(lat_bins)-1,len(lon_bins)-1, var_cams.shape[0]))
    print(regridded_data.shape)

    cams_lat = cams0.variables['latitude'][:]
    cams_lon = cams0.variables['longitude'][:]
    ilon = np.where(cams_lon>=180)
    cams_lon[ilon] = cams_lon[ilon]-360


    cams_lon,cams_lat = np.meshgrid(cams_lon,cams_lat)
    cams_lon0 = cams_lon.flatten()
    cams_lat0 = cams_lat.flatten()
    print(cams_lat.shape,cams_lon.shape,lat_bins.shape,lon_bins.shape)
    for i in range(var_cams.shape[0]): #254
        cams_extinction = var_cams[i,:,:].flatten()
        # Compute 2D histogram for the mean wind
        mask = ~np.isnan(cams_extinction)
        cams_extinction = cams_extinction[mask]
        cams_lat = cams_lat0[mask]
        cams_lon = cams_lon0[mask]

        stat, x_edge, y_edge, _ = binned_statistic_2d(
            cams_lat, cams_lon, cams_extinction, statistic='mean', bins=[lat_bins, lon_bins])
        regridded_data[:,:,i]=stat
        # Replace NaN with zeros (or another value if necessary)
        #stat = np.nan_to_num(stat)

    print(~np.isnan(cams_extinction))
    var_zonal_cams = np.nanmean(regridded_data,axis=1)#[:,48:] #zonal mean
    print(var_zonal_cams[var_zonal_cams>0])
    print('v_cams9')

    cams_altitude = xr.open_dataset('/scratch/nld6854/earthcare/earthcare_scripts/scripts/cams_altitude.nc')
    cams_h = cams_altitude['altitude'].values[::-1,:,:]
    for i in range(cams_h.shape[0]): #254
        cams_height = cams_h[i,:,:].flatten()
        # Compute 2D histogram for the mean wind
        mask = ~np.isnan(cams_height)
        cams_height = cams_height[mask]
        cams_lat = cams_lat0[mask]
        cams_lon = cams_lon0[mask]

        stat, x_edge, y_edge, _ = binned_statistic_2d(
            cams_lat, cams_lon, cams_height, statistic='mean', bins=[lat_bins, lon_bins])
        regridded_height[:,:,i]=stat

    var_zonal_cams = var_zonal_cams[:,::-1]
    print('var_zonal_cams',var_zonal_cams)
    print('var_zonal_cams.max(),var_zonal_cams.min(),len(var_zonal_cams[var_zonal_cams>0])',var_zonal_cams.max(),var_zonal_cams.min(),len(var_zonal_cams[var_zonal_cams>0]))
    print(var_cams.shape) #(137, 451, 900)
    print('var_zonal_cams.shape',var_zonal_cams.shape) #(137,451)

    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

    #regridded_data = regridded_data[:,:,48:]
    regridded_data = regridded_data[:,:,::-1]

    regridded_data_xr = xr.Dataset(
        {
            "extinction_coefficient_cams": (["latitude", "longitude", "height"], regridded_data),
            "actual_height": (["latitude", "longitude", "height"], regridded_height)
        },
        coords={
            "latitude": lat_centers,
            "longitude": lon_centers,
            "height": cams_h[:,0,0]}) #cams_h[:89,0,0]
    output_filename = "regridded_cams_extinction_coe_2deg_mean_"+month+"_2025.nc"
    regridded_data_xr.to_netcdf(output_filename)


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

lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

clons,clats = np.meshgrid(lon_centers,lat_centers)
nan_percentage = np.isnan(stat).sum() / stat.size * 100


print('cams_h=',cams_h.shape,'atlid_h=',atlid_h.shape,'var_zonal_atlid=',var_zonal_atlid.shape)
#interpolating ATLID to CAMS height
#var_atlid_interp = np.zeros((var_zonal_cams.shape[0], var_zonal_cams.shape[1]))
#for i in range(var_zonal_atlid.shape[0]):
#    var_atlid_interp[i,:] = np.interp(cams_h,atlid_h,var_zonal_atlid[i,:])

#interpolating CAMS to ATLID height
var_cams_interp = np.zeros((var_zonal_atlid.shape[0],var_zonal_atlid.shape[1]))
for i in range(var_zonal_atlid.shape[0]):
    var_cams_interp[i,:] = np.interp(atlid_h,regridded_height[i,0,:],var_zonal_cams[i,:]) #cams_h[i,0,:89]
vmax = var_cams_interp.max()/1.5
vmax = 1.e-4

#print('CAMS mean=',np.nanmean(aod_cams))
#print('ATLID mean=',np.nanmean(var_zonal_atlid),np.nanmean(var_atlid_interp))
#print(np.nanmean(aod_cams-aod_atlid))
cams_h,lat_p = np.meshgrid(cams_h,lat_centers)
fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(10,15))
fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(8,12))
im=ax1.pcolormesh(lat_atlid,atlid_h,var_zonal_atlid.transpose(),cmap='viridis',norm=matplotlib.colors.LogNorm(vmax=vmax,vmin=vmax/1000.))

bar = plt.colorbar(im, orientation='vertical',ax=ax1,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('Extinction coefficient',fontsize=15)
bar.ax.tick_params(labelsize=15)
ax1.set_title('Zonal mean extinction coefficient ATLID '+fmonth+' 2025',fontsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax1.set_ylim(0,20000)

im=ax2.pcolormesh(lat_p,cams_height[:,0,:],var_zonal_cams,cmap='viridis',norm=matplotlib.colors.LogNorm(vmax=vmax,vmin=vmax/1000.))
bar = plt.colorbar(im, orientation='vertical',ax=ax2,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('Extinction coefficient',fontsize=15)
bar.ax.tick_params(labelsize=15)
ax2.set_title('Zonal mean extinction coefficient CAMS '+fmonth+' 2025',fontsize=15)
ax2.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)

im=ax3.pcolormesh(lat_p,atlid_h,var_cams_interp-var_zonal_atlid,cmap='RdBu_r',vmax=vmax,vmin=-vmax)
bar = plt.colorbar(im, orientation='vertical',ax=ax3,shrink=0.7, pad=0.1)
bar.ax.set_ylabel('diff Extinction coefficient',fontsize=15)
bar.ax.tick_params(labelsize=15)
ax3.set_title('CAMS-ATLID',fontsize=15)
ax3.tick_params(axis='x', labelsize=15)
ax3.tick_params(axis='y', labelsize=15)

plt.tight_layout()
fig.savefig('zonal_nanmean_extinction_coefficient_'+figname+'_2deg_masknan_mean_'+month+'_2025.jpg',bbox_inches='tight')
sys.exit()


aod_cams = aod_cams.reshape(-1)
aod_atlid = aod_atlid.reshape(-1)
nbins = 100

binsc = np.linspace(0,aod_cams.max(),nbins)
histc,binsc = np.histogram(aod_cams,bins=binsc,density=False)
bcc = 0.5*(binsc[1:] + binsc[:-1])

binsa = np.linspace(0,aod_atlid.max(),nbins)
hista,binsa = np.histogram(aod_atlid,bins=binsa,density=False)
bca = 0.5*(binsa[1:] + binsa[:-1])


fig,ax1 = plt.subplots(1)
ax1.plot(bcc,histc,label='CAMS')
ax1.plot(bca,hista,label='ATLID')
#ax1.set_ylim(1e-8,0.1)
#ax1.set_yscale('log')
ax1.set_xlabel('AOD',fontsize=15)
ax1.set_ylabel('Counts',fontsize=15)
ax1.legend(frameon=False,fontsize=15)
fig.savefig('histograms_CAMS_ATLID'+figname+'_2deg.jpg')
