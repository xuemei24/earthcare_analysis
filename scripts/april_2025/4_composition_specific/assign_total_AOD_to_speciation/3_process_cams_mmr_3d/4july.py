import xarray as xr

month = 'july'
cams_dir = '/net/pc190625/nobackup_1/users/wangxu/cams_data/aerosol_mmr/temp/'
cams_dir = '/net/pc200254/nobackup/users/wangxu/cams_data/aerosol_mmr/'+month+'/not_used/'
# List of NetCDF files
#file_list = ["ammonium_mmr_april_2025_0.nc","ammonium_mmr_april_2025_3.nc", "ammonium_mmr_april_2025_6.nc", "ammonium_mmr_april_2025_9.nc"]
#aero_types = ["dust","sulfate","ammonium","nitrate","organic_matter","black_carbon","sea_salt","dust_bc"]
aero_types = ["dust_bc","sea_salt","organic_matter_sulfate_nitrate_ammonium"]
for aero_type in aero_types:
    for forecast_reference_time in [3,6,9,12]:
        print('forecast_reference_time=',forecast_reference_time)
        file_list = [cams_dir+aero_type+"_extinction_coefficient__mmr_"+month+"_2025_slice_"+str(itime)+"_"+str(forecast_reference_time)+".nc" for itime in range(62)]
        print('file_list done')
 
        # Open and concatenate along time dimension
        ds = xr.open_mfdataset(
            file_list,
            combine="nested",
            concat_dim="forecast_reference_time",
            parallel=False,  # set to True if Dask is installed
            engine="netcdf4"     # optional: ensures classic NetCDF4 engine
        )
 
        # Optionally save to a new file
        ds.to_netcdf(cams_dir+aero_type+"_extinction_coefficient_"+month+"_2025_"+str(forecast_reference_time)+".nc")

#'temp/'+aero_type+'_extinction_coefficient_'+suffix1+'slice_'+str(itime)+'_'+fforecast_period+s can be deleted, only the merged 4 files are used later
