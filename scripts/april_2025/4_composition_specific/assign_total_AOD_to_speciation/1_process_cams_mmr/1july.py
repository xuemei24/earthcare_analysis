import numpy as np
import xarray as xr
import sys
from pylab import *

month = 'july'
#cams_h = loadtxt('/usr/people/wangxu/Desktop/earthcare_scripts/grib_height/geometric_height.csv',skiprows=2)
#cams_h = cams_h[::-1]
cams_altitude=xr.open_dataset('/scratch/nld6854/earthcare/earthcare_scripts/scripts/cams_altitude.nc')
lat = cams_altitude['latitude']
lon = cams_altitude['longitude']
cams_h = cams_altitude['altitude'][::-1,:,:]
print(cams_h.shape)
cams_orography=xr.open_dataset('/scratch/nld6854/earthcare/earthcare_scripts/scripts/cams_orography.nc')
cams_o = cams_orography['orography']
print(cams_o.shape)
cams_dir = '/net/pc190625/nobackup_1/users/wangxu/cams_data/aerosol_mmr/'
cams_dir = '/net/pc200254/nobackup/users/wangxu/cams_data/aerosol_mmr/'+month+'/'
suffix = '_mmr_'+month+'_2025.nc'

cams_data = ['ammonium', 'anthropogenic_secondary_organic', 'biogenic_secondary_organic', 'dust_0.03-0.55', 'dust_0.55-0.9', 'dust_0.9-20', 'hydrophilic_black_carbon', 'hydrophilic_organic_matter','hydrophobic_black_carbon', 'hydrophobic_organic_matter', 'nitrate_coarse_mode', 'nitrate_fine_mode', 'sea_salt_0.03-0.5', 'sea_salt_0.5-5', 'sea_salt_5-20', 'sulfate'] 
var_names = ['aermr18','aermr20','aermr19','aermr04','aermr05','aermr06','aermr09','aermr07','aermr10','aermr08','aermr17','aermr16','aermr01','aermr02','aermr03','aermr11']

#for data,var_name in zip(cams_data,var_names):
#    print(data)
#    a = xr.open_dataset(cams_dir+data+suffix,chunks={})
#    var = a[var_name][:,:,::-1,:,:].values
#    a.close()
#    print(var)
#    var_int = np.zeros((var.shape[0],var.shape[1],var.shape[3],var.shape[4]))
#    for i in range(var.shape[0]):
#        for j in range(var.shape[1]):
#            var_int[i,j] = np.trapz(var[i,j],x=cams_h,axis=0)

#    print('finish looping')
#    regridded_data_xr = xr.Dataset(
#        {
#            var_name+'_mmr': (["latitude", "longitude"], var_int)
#        },
#        coords={
#            "latitude": lat,
#            "longitude": lon
#        }
#    )
#    output_filename = cams_dir+'integrated_'+var_name+'_'+data+suffix
#    regridded_data_xr.to_netcdf(output_filename)

#suffix1 = '_mmr_april_2025_'
#suffix2 = '.nc'
#forecast_periods = ['0','3','6','9']
#for data,var_name in zip(cams_data,var_names):
#    print(data)
#    var_int = np.zeros((4,60,451,900))
#    for itime,forecast_period in enumerate(forecast_periods):
#        a = xr.open_dataset(cams_dir+data+suffix1+forecast_period+suffix2,chunks={})
#        var = a[var_name][0,:,::-1,:,:]#.values
#        a.close()
#        print(var)
#        for j in range(var.shape[0]):
#            print(j)
#            var_int[itime,j] = np.trapz(var[j],x=cams_h,axis=0)
#
#    print('finish looping')
#    regridded_data_xr = xr.Dataset(
#        {
#            var_name+'_mmr': (["latitude", "longitude"], var_int)
#        },
#        coords={
#            "latitude": lat,
#            "longitude": lon
#        }
#    )
#    output_filename = cams_dir+'integrated_'+var_name+'_'+data+suffix1+'_simpleH'+suffix2
#    regridded_data_xr.to_netcdf(output_filename)

suffix1 = '_mmr_'+month+'_2025_'
suffix2 = '.nc'
forecast_periods = ['12']
for data,var_name in zip(cams_data,var_names):
    print(data)
    for forecast_period in forecast_periods:
        print(cams_dir+data+suffix1+forecast_period+suffix2)
        a = xr.open_dataset(cams_dir+data+suffix1+forecast_period+suffix2,chunks={})
        var = a[var_name][0,:,::-1,:,:]#.values
        forecast_reference_time = a['forecast_reference_time'].values
        a.close()
        print(var)
        var_int = np.zeros((var.shape[0],451,900))
        for j in range(var.shape[0]):
            print(j)
            var_int[j] = np.trapz(var[j],x=cams_h,axis=0)

        print('finish looping')
        regridded_data_xr = xr.Dataset(
            {
                var_name+'_mmr': (["forecast_reference_time","latitude", "longitude"], var_int)
            },
            coords={
                "forecast_reference_time":forecast_reference_time,
                "latitude": lat,
                "longitude": lon
            }
        )
        output_filename = cams_dir+'integ/integrated_'+var_name+'_'+data+suffix1+forecast_period+suffix2
        regridded_data_xr.to_netcdf(output_filename)
