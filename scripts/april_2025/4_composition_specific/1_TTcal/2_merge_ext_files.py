import xarray as xr

month = 'december'
year = '2024' if month == 'december' else '2025'
cams_dir = '/scratch/nld6854/earthcare/cams_data/'+month+'_'+year+'/TTcal/'

aero_types = ["TTcal_aod355nm_per_composition"]
for aero_type in aero_types:
    #for forecast_reference_time in [0,3,6,9]:
    file_list = [cams_dir+aero_type+'_'+month+"_"+year+"_"+str(forecast_period)+".nc" for forecast_period in [0,3,6,9]]
    print(file_list)
    print('file_list done')
 
    # Open and concatenate along time dimension
    ds = xr.open_mfdataset(
        file_list,
        combine="nested",
        concat_dim="forecast_period",
        parallel=False,  # set to True if Dask is installed
        engine="netcdf4"     # optional: ensures classic NetCDF4 engine
    )
 
    ds.to_netcdf(cams_dir+aero_type+'_'+month+"_"+year+".nc")

#'temp/'+aero_type+'_extinction_coefficient_'+suffix1+'slice_'+str(itime)+'_'+fforecast_period+s can be deleted, only the merged 4 files are used later
