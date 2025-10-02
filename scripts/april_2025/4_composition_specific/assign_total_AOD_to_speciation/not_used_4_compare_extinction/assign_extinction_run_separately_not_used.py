import numpy as np
import xarray as xr
import sys
from pylab import *

cams_dir = '/net/pc190625/nobackup_1/users/wangxu/cams_data/'#aerosol_mmr/'
cams_dir_200254 = '/net/pc200254/nobackup/users/wangxu/cams_data/'
month = 'april'
suffix2 = '.nc'

cams_data = ['ammonium', 'anthropogenic_secondary_organic', 'biogenic_secondary', 'dust_0.3-0.55', 'dust_0.55-0.9', 'dust_0.9-20', 'hydrophilic_black_carbon', 'hydrophilic_organic_matter','hydrophobic_black_carbon', 'hydrophobic_organic_matter', 'nitrate_coarse_mode', 'nitrate_fine_mode', 'sea_salt_0.03-0.5', 'sea_salt_0.5-5', 'sea_salt_5-20', 'sulfate'] 
var_names = ['aermr18','aermr20','aermr19','aermr04','aermr05','aermr06','aermr09','aermr07','aermr10','aermr08','aermr17','aermr16','aermr01','aermr02','aermr03','aermr11']


forecast_periods = ['0','3','6','9']
for fforecast_period in forecast_periods:
    fext = xr.open_dataset(cams_dir+'extinction_355nm_multi/aerosol_extinction_coe_355nm_'+month+'2025_'+fforecast_period+'.nc')
    cext = fext['aerext355'].values
    model_level = fext['model_level']#.values
    forecast_reference_time = fext['forecast_reference_time']
    forecast_period_0 = fext['forecast_period']
    lat = fext['latitude']
    lon = fext['longitude']

    print(cext.shape)
    fext.close()
    for data,var_name in zip(cams_data,var_names):
        idata_nc = np.zeros((cext.shape))
        for itime in range(60):
            ftmmr = xr.open_dataset(cams_dir+'aerosol_mmr/temp/'+'total_mmr__mmr_'+month+'_2025_slice_'+str(itime)+'_'+fforecast_period+suffix2)
            tmmr  = ftmmr['total_mmr_'+fforecast_period]
            fimmr = xr.open_dataset(cams_dir+'aerosol_mmr/temp/'+data+'_mmr_'+month+'_2025_slice_'+str(itime)+'_'+fforecast_period+suffix2)
            immr  = fimmr[var_name]
            idata_nc[0,itime] = np.where(immr/tmmr>=0.95,cext[0,itime],np.nan)
        

        print('finish looping')
        regridded_data_xr = xr.Dataset(
            {
                data+'extinciton_coe'+fforecast_period: (["forecast_period","forecast_reference_time","model_level","latitude", "longitude"], idata_nc)
            },
            coords={
                "forecast_period":forecast_period_0,
                "forecast_reference_time":forecast_reference_time,
                "model_level":model_level,
                "latitude": lat,
                "longitude": lon
            }
        )
        #output_filename = cams_dir+'aerosol_mmr/extinc_per_composition/'+data+'_extinction_coe_'+month+'_2025_'+fforecast_period+suffix2
        output_filename = cams_dir_200254+'aerosol_mmr/extinc_per_composition/'+data+'_extinction_coe_'+month+'_2025_'+fforecast_period+suffix2
        regridded_data_xr.to_netcdf(output_filename)
