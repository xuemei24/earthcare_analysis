import xarray as xr

month = 'july'
cams_dir = '/net/pc200254/nobackup/users/wangxu/cams_data/aerosol_mmr/'+month+'/'
# List of NetCDF files
cams_data = ['ammonium', 'anthropogenic_secondary_organic', 'biogenic_secondary_organic', 'dust_0.03-0.55', 'dust_0.55-0.9', 'dust_0.9-20', 'hydrophilic_black_carbon', 'hydrophilic_organic_matter','hydrophobic_black_carbon', 'hydrophobic_organic_matter', 'nitrate_coarse_mode', 'nitrate_fine_mode', 'sea_salt_0.03-0.5', 'sea_salt_0.5-5', 'sea_salt_5-20', 'sulfate'] 
var_names = ['aermr18','aermr20','aermr19','aermr04','aermr05','aermr06','aermr09','aermr07','aermr10','aermr08','aermr17','aermr16','aermr01','aermr02','aermr03','aermr11']


suffix1 = '_mmr_'+month+'_2025_'
suffix2 = '.nc'
forecast_periods = ['0','3','6','9']
forecast_periods = ['3','6','9','12']

for data,var_name in zip(cams_data,var_names):
    print(data)
    file_list = [cams_dir+'integ/integrated_'+var_name+'_'+data+suffix1+forecast_period+suffix2 for forecast_period in forecast_periods]
    # Open and concatenate along time dimension
    ds = xr.open_mfdataset(
        file_list,
        combine="nested",
        concat_dim="forecast_period",
        parallel=False,  # set to True if Dask is installed
        engine="netcdf4"     # optional: ensures classic NetCDF4 engine
    )

    # Optionally save to a new file
    ds.to_netcdf(cams_dir+'integ/integrated_'+var_name+'_'+data+suffix1[:-1]+suffix2)

    print(data,'finished')

#individual integrated files can be deleted (not needed anymore)
