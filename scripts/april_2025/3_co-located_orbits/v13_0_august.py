import xarray as xr

# List of NetCDF files
month = 'august'
#cams_dir='/net/pc190625/nobackup_1/users/wangxu/cams_data/lwc/'
cams_dir='/net/pc200254/nobackup/users/wangxu/cams_data/aerosol_mmr/august_fire/'
file_list = [cams_dir+"specific_cloud_liquid_water_content_"+month+"_20250801-10_0.nc",cams_dir+"specific_cloud_liquid_water_content_"+month+"_20250801-10_3.nc", cams_dir+"specific_cloud_liquid_water_content_"+month+"_20250801-10_6.nc", cams_dir+"specific_cloud_liquid_water_content_"+month+"_20250801-10_9.nc"]

# Open and concatenate along time dimension
ds = xr.open_mfdataset(
    file_list,
    combine="nested",
    concat_dim="forecast_period",
    parallel=False,  # set to True if Dask is installed
    engine="netcdf4"     # optional: ensures classic NetCDF4 engine
)

# Optionally save to a new file
ds.to_netcdf(cams_dir+"specific_cloud_liquid_water_content_"+month+"_2025.nc")


