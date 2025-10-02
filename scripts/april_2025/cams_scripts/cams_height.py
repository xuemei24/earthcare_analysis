import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
import sys

df = pd.read_csv('/usr/people/wangxu/Desktop/earthcare_scripts/grib_height/l137_model_level_def.csv')
a = df['a[Pa]'][1:]
b = df['b'][1:]
#a_r = np.array(a)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
#b_r = np.array(b)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
#a_br = np.broadcast_to(a_r, (137, 12, 42, 451, 900))
#b_br = np.broadcast_to(b_r, (137, 12, 42, 451, 900))

cams_path = '/net/pc190625/nobackup_1/users/wangxu/cams_data/'
spre = Dataset(cams_path+'surface_pressure.nc')
psurf = spre.variables['sp'][0,0,:]
#psurf_r = psurf[np.newaxis,:]
#psurf_br = np.broadcast_to(psurf_r, (137, 12, 42, 451, 900))

print(psurf.shape)
print('psurf',psurf)
print("spre['sp'][:].shape",spre['sp'][:].shape)
pres = np.zeros((137,451,900))
for i in range(137):
    print(a.values[i],b.values[i])
    pres[i] = a.values[i]+b.values[i]*psurf

print('pres',pres)
R = 287.05  # J/kg/K
g = 9.80665 # m/s²
p0 = 101325. # Pa

stemp = Dataset(cams_path+'temperature/temperature_march_2025_3.nc')
T = stemp.variables['t'][0,1]
lat = stemp.variables['latitude'][:]
lon = stemp.variables['longitude'][:]
print(lat.shape,lon.shape)
model_level = stemp.variables['model_level'][:]
print(T.shape)
z = (R*T/g)*np.log(p0/pres) #downloading temperature and pressure
print(z.shape)
print('z',z)
#z = geopotential/g #download geopotential

regridded_data_xr = xr.DataArray(
    z,
    coords=[model_level,lat, lon],
    dims=['model_level','latitude', 'longitude'],
    name='altitude')
output_filename = "/scratch/nld6854/earthcare/earthcare_scripts/scripts/cams_altitude.nc"
regridded_data_xr.to_netcdf(output_filename)

sgeo = Dataset(cams_path+'surface_geopotential.nc')
surf_geopogential = sgeo.variables['z'][0,0,:]
print(surf_geopogential.shape)
zorog = surf_geopogential/g #download surf_geopotential

regridded_data_xr = xr.DataArray(
    zorog,
    coords=[lat, lon],
    dims=['latitude', 'longitude'],
    name='orography')
output_filename = "/scratch/nld6854/earthcare/earthcare_scripts/scripts/cams_orography.nc"
regridded_data_xr.to_netcdf(output_filename)

np.savetxt('orography.txt',zorog,delimiter=',',header = str(lon.reshape(-1)))

from pylab import *
orog = loadtxt('orography.txt',skiprows=1)
print(orog)
