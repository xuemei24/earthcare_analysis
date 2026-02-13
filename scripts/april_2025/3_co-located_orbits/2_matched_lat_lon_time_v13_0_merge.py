import xarray as xr

# List of NetCDF files
month = 'november'
year = '2024' if month == 'december' else '2025'
cams_dir='/scratch/nld6854/earthcare/cams_data/'+month+'_'+year+'/'
file_list = [cams_dir+"specific_cloud_liquid_water_content_"+month+"_"+year+"_0.nc",cams_dir+"specific_cloud_liquid_water_content_"+month+"_"+year+"_3.nc", cams_dir+"specific_cloud_liquid_water_content_"+month+"_"+year+"_6.nc", cams_dir+"specific_cloud_liquid_water_content_"+month+"_"+year+"_9.nc"]

# Open and concatenate along time dimension
ds = xr.open_mfdataset(
    file_list,
    combine="nested",
    concat_dim="forecast_period",
    parallel=False,  # set to True if Dask is installed
    engine="netcdf4"     # optional: ensures classic NetCDF4 engine
)

# Optionally save to a new file
ds.to_netcdf(cams_dir+"specific_cloud_liquid_water_content_"+month+"_"+year+".nc")


