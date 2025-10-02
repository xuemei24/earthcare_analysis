import xarray as xr

cams_dir = '/net/pc200254/nobackup/users/wangxu/cams_data/aerosol_mmr/july/'
# List of NetCDF files
cams_data = ['hydrophobic_organic_matter', 'nitrate_coarse_mode', 'nitrate_fine_mode', 'sea_salt_0.03-0.5', 'sea_salt_0.5-5', 'sea_salt_5-20', 'sulfate'] 
var_names = ['aermr08','aermr17','aermr16','aermr01','aermr02','aermr03','aermr11']


suffix1 = '_mmr_july_2025_'
suffix2 = '.nc'
forecast_periods = ['0','3','6','9']

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
