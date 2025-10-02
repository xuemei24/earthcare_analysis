import xarray as xr

month = 'july'
cams_dir = '/net/pc200254/nobackup/users/wangxu/cams_data/aerosol_mmr/july/cams_original_aerosol_mmr/'

aero_types = ["dust_0.03-0.55_mmr"]
for aero_type in aero_types:
    #for forecast_reference_time in [0,3,6,9]:
    file_list = [cams_dir+aero_type+"_"+month+"_2025_"+str(fforecast_period)+".nc" for fforecast_period in [0,3,6,9]]
    print('file_list done')
    # Open and concatenate along time dimension
    ds = xr.open_mfdataset(
        file_list,
        combine="nested",
        concat_dim="forecast_period",
        parallel=False,  # set to True if Dask is installed
        engine="netcdf4"     # optional: ensures classic NetCDF4 engine
    )
    print('forecast_period=',ds['forecast_period']) 
    print('forecast_reference_time',ds['forecast_reference_time'])
