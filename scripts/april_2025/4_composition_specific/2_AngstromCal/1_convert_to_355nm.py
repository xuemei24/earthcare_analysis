import numpy as np
import xarray as xr
import sys
import os
import pandas as pd
from pylab import *

#################
### FUNCTIONS ###
#################
def get_AngstromExponent(aodW1, aodW2, wave1, wave2):
    return -np.log(aodW1 / aodW2) / np.log(wave1 / wave2)

def get_AerosolOpticalDepth(aod_old, wave_old, wave_new, angstrom):
    print('aod_old>0',aod_old.values[aod_old.values>0], 'wave_old',wave_old, 'wave_new',wave_new, 'angstrom>0',angstrom.values[angstrom.values>0])
    return ((wave_new / wave_old) ** (-angstrom)) * aod_old

month = 'june'
cams_dir = '/scratch/nld6854/earthcare/cams_data/'+month+'_2025/'

ds355 = xr.open_dataset(cams_dir+'total_aerosol_optical_depth_355nm_'+month+'_2025.nc')
aod355 = ds355['aod355']
forecast_period = ds355['forecast_period']
forecast_reference_time = ds355['forecast_reference_time']
latitude = ds355['latitude']
longitude = ds355['longitude']
ds355.close()

ds550 = xr.open_dataset(cams_dir+'multiple_aod_550nm_'+month+'_2025.nc')
aod550 = ds550['aod550']
ssaod550 = ds550['ssaod550']
du_om_aod550 = ds550['duaod550']+ds550['omaod550']+ds550['bcaod550']+ds550['soaod550']
duaod550 = ds550['duaod550']+ds550['bcaod550']
#aod340 = ds550_speciation['aod340']
ds550.close()

AE_550to355 = get_AngstromExponent(aod550, aod355, 550, 355)
#AE_550to340 = get_AngstromExponent(aod550, aod340, 550, 340) 

ssa_aod_550_AE_550to355 = get_AerosolOpticalDepth(ssaod550,550,355,AE_550to355)
du_aod_550_AE_550to355 = get_AerosolOpticalDepth(duaod550,550,355,AE_550to355)
du_om_aod_550_AE_550to355 = get_AerosolOpticalDepth(du_om_aod550,550,355,AE_550to355)

#ssa_aod_550_AE_550to340 = get_AerosolOpticalDepth(ssaod550,550,355,AE_550to340)
#du_aod_550_AE_550to340 = get_AerosolOpticalDepth(duaod550,550,355,AE_550to340)

####################
### WRITE NETCDF ###
####################
ds = xr.Dataset(
     {
        "AE_550to355": (("forecast_period", "forecast_reference_time","latitude","longitude"),AE_550to355.values),
        "ssa_aod_550_AE_550to355": (("forecast_period", "forecast_reference_time","latitude","longitude"), ssa_aod_550_AE_550to355.values),
        "du_aod_550_AE_550to355": (("forecast_period", "forecast_reference_time","latitude","longitude"), du_aod_550_AE_550to355.values),
        "du_om_aod_AE_550to355": (("forecast_period", "forecast_reference_time","latitude","longitude"), du_om_aod_550_AE_550to355.values)
    },
    coords={
        "forecast_period": forecast_period,
        "forecast_reference_time": forecast_reference_time,
        "latitude": latitude,
        "longitude": longitude
    })

filepath = os.path.join(cams_dir+'TTcal/',
f"Angstrom_aod355nm_per_composition_{month}_2025.nc")
ds.to_netcdf(filepath, format="NETCDF4")
