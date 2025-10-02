import numpy as np
import xarray as xr
import sys
from pylab import *

month = 'july'
cams_dir = '/net/pc190625/nobackup_1/users/wangxu/cams_data/aerosol_mmr/'
cams_dir = '/net/pc200254/nobackup/users/wangxu/cams_data/aerosol_mmr/'+month+'/'
suffix = '_mmr_'+month+'_2025.nc'

cams_data = ['ammonium', 'anthropogenic_secondary_organic', 'biogenic_secondary_organic', 'dust_0.03-0.55', 'dust_0.55-0.9', 'dust_0.9-20', 'hydrophilic_black_carbon', 'hydrophilic_organic_matter','hydrophobic_black_carbon', 'hydrophobic_organic_matter', 'nitrate_coarse_mode', 'nitrate_fine_mode', 'sea_salt_0.03-0.5', 'sea_salt_0.5-5', 'sea_salt_5-20', 'sulfate'] 
var_names = ['aermr18','aermr20','aermr19','aermr04','aermr05','aermr06','aermr09','aermr07','aermr10','aermr08','aermr17','aermr16','aermr01','aermr02','aermr03','aermr11']


suffix1 = '_mmr_'+month+'_2025_'
suffix2 = '.nc'
forecast_periods = ['0','3','6','9']
forecast_periods = ['3','6','9','12']
for fforecast_period in forecast_periods:
    print(fforecast_period)
    for itime in range(62):
        total_mmr = np.zeros((1,1,137,451,900))
        for data,var_name in zip(cams_data,var_names):
            print(cams_dir+'temp/'+data+suffix1+'slice_'+str(itime)+'_'+fforecast_period+suffix2)
            a = xr.open_dataset(cams_dir+'temp/'+data+suffix1+'slice_'+str(itime)+'_'+fforecast_period+suffix2)
            model_level = a['model_level']#.values
            forecast_reference_time = a['forecast_reference_time']
            forecast_period_0 = a['forecast_period']
            lat = a['latitude']
            lon = a['longitude']
            total_mmr = total_mmr+a[var_name].values
            a.close()
 
            print('finish looping')
            regridded_data_xr = xr.Dataset(
                {
                    'total_mmr_'+fforecast_period: (["forecast_period","forecast_reference_time","model_level","latitude", "longitude"], total_mmr)
                },
                coords={
                    "forecast_period":forecast_period_0,
                    "forecast_reference_time":forecast_reference_time,
                    "model_level":model_level,
                    "latitude": lat,
                    "longitude": lon
                }
            )
            output_filename = cams_dir+'temp/'+'total_mmr_'+suffix1+'slice_'+str(itime)+'_'+fforecast_period+suffix2
            regridded_data_xr.to_netcdf(output_filename)
