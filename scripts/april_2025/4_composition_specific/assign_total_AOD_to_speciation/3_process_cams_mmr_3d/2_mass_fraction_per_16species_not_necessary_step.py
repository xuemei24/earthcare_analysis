import numpy as np
import xarray as xr
import sys
from pylab import *

month = 'july'
cams_dir = '/net/pc190625/nobackup_1/users/wangxu/cams_data/aerosol_mmr/'
suffix = '_mmr_'+month+'_2025.nc'

cams_data = ['ammonium', 'anthropogenic_secondary_organic', 'biogenic_secondary', 'dust_0.03-0.55', 'dust_0.55-0.9', 'dust_0.9-20', 'hydrophilic_black_carbon', 'hydrophilic_organic_matter','hydrophobic_black_carbon', 'hydrophobic_organic_matter', 'nitrate_coarse_mode', 'nitrate_fine_mode', 'sea_salt_0.03-0.5', 'sea_salt_0.5-5', 'sea_salt_5-20', 'sulfate'] 
var_names = ['aermr18','aermr20','aermr19','aermr04','aermr05','aermr06','aermr09','aermr07','aermr10','aermr08','aermr17','aermr16','aermr01','aermr02','aermr03','aermr11']

aero_types = ['dust','organic_matter','black_carbon','sulfate','nitrate','ammonium']
aero_names = [['aermr04','aermr05','aermr06'],['aermr07','aermr08','aermr19','aermr20'],['aermr09','aermr10'],['aermr11'],['aermr16','aermr17'],['aermr18']]
file_names = [['dust_0.03-0.55', 'dust_0.55-0.9', 'dust_0.9-20'],['hydrophilic_organic_matter','hydrophobic_organic_matter','biogenic_secondary','anthropogenic_secondary_organic'],['hydrophilic_black_carbon','hydrophobic_black_carbon'],['sulfate'],['nitrate_fine_mode','nitrate_coarse_mode'],['ammonium']]

suffix1 = '_mmr_'+month+'_2025_'
suffix2 = '.nc'
forecast_periods = ['0','3','6','9']
for fforecast_period in forecast_periods:
    print(fforecast_period)
    for itime in range(60):
        ftmmr = xr.open_dataset(cams_dir+'temp/'+'total_mmr_'+suffix1+'slice_'+str(itime)+'_'+fforecast_period+suffix2)
        total_mmr = ftmmr['total_mmr_'+fforecast_period].values  # 1,1,137,451,900
        ftmmr.close()

        for aero_type,aero_name0,file_name0 in zip(aero_types,aero_names,file_names):
            mass_per_species = np.zeros((total_mmr.shape))
            for aero_name1,file_name1 in zip(aero_name0,file_name0):
                print(aero_type,aero_name1,file_name1)
                faero = xr.open_dataset(cams_dir+'temp/'+file_name1+suffix1+'slice_'+str(itime)+'_'+fforecast_period+suffix2)
                aerommr = faero[aero_name1].values
                model_level = faero['model_level']#.values
                forecast_reference_time = faero['forecast_reference_time']
                forecast_period_0 = faero['forecast_period']
                lat = faero['latitude']
                lon = faero['longitude']

                faero.close()

                mass_per_species = mass_per_species + aerommr

            fraction = mass_per_species/total_mmr


            print('finish looping')
            regridded_data_xr = xr.Dataset(
                {
                    aero_type+'_mass_fraction_'+fforecast_period: (["forecast_period","forecast_reference_time","model_level","latitude", "longitude"], fraction)
                },
                coords={
                    "forecast_period":forecast_period_0,
                    "forecast_reference_time":forecast_reference_time,
                    "model_level":model_level,
                    "latitude": lat,
                    "longitude": lon
                }
            )
            output_filename = cams_dir+'temp/'+aero_type+'_mass_fraction_'+suffix1+'slice_'+str(itime)+'_'+fforecast_period+suffix2
            regridded_data_xr.to_netcdf(output_filename)
