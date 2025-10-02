import h5py
import concurrent.futures
import glob
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
#from pyresample import geometry, kd_tree
from scipy.stats import binned_statistic_2d
from pylab import *
import sys
import pandas as pd
import matplotlib.colors as colors

cams_data = ['ammonium', 'anthropogenic_secondary_organic', 'biogenic_secondary_organic', 'dust_0.03-0.55', 'dust_0.55-0.9', 'dust_0.9-20', 'hydrophilic_black_carbon', 'hydrophilic_organic_matter','hydrophobic_black_carbon', 'hydrophobic_organic_matter', 'nitrate_coarse_mode', 'nitrate_fine_mode', 'sea_salt_0.03-0.5', 'sea_salt_0.5-5', 'sea_salt_5-20', 'sulfate'] 
var_names = ['aermr18','aermr20','aermr19','aermr04','aermr05','aermr06','aermr09','aermr07','aermr10','aermr08','aermr17','aermr16','aermr01','aermr02','aermr03','aermr11']
#cams_data = ['ammonium', 'dust_0.3-0.55', 'dust_0.55-0.9', 'dust_0.9-20', 'hydrophilic_black_carbon', 'hydrophilic_organic_matter','hydrophobic_black_carbon', 'hydrophobic_organic_matter', 'nitrate_coarse_mode', 'nitrate_fine_mode', 'sea_salt_0.03-0.5', 'sea_salt_0.5-5', 'sea_salt_5-20', 'sulfate'] 
#var_names = ['aermr18','aermr04','aermr05','aermr06','aermr09','aermr07','aermr10','aermr08','aermr17','aermr16','aermr01','aermr02','aermr03','aermr11']

month = 'july'
suffix1 = '_mmr_'+month+'_2025.nc'

cams_dir = '/net/pc190625/nobackup_1/users/wangxu/cams_data/aerosol_mmr/'
cams_dir = '/net/pc200254/nobackup/users/wangxu/cams_data/aerosol_mmr/'+month+'/'

forecast_reference_time = 62
mmr = np.zeros([len(cams_data),4,forecast_reference_time,451,900])
for ij,data,var_name in zip(range(len(cams_data)),cams_data,var_names):
    print(data)
    faero = xr.open_dataset(cams_dir+'integ/'+'integrated_'+var_name+'_'+data+suffix1)
    mmr[ij] = faero[var_name+'_mmr']

total_mmr = np.sum(mmr,axis=0)

lat = faero['latitude'].values
lon = faero['longitude'].values
forecast_period = np.arange(4)
forecast_reference_time = np.arange(forecast_reference_time)

out_data_xr = xr.Dataset(
    {
        "ammonium": (["forecast_period","forecast_reference_time","latitude", "longitude"], mmr[0]/total_mmr),
        "anthropogenic_secondary_organic": (["forecast_period","forecast_reference_time","latitude", "longitude"],  mmr[1]/total_mmr),
        "biogenic_secondary": (["forecast_period","forecast_reference_time","latitude", "longitude"],  mmr[2]/total_mmr),
        "dust_0.03-0.55": (["forecast_period","forecast_reference_time","latitude", "longitude"], mmr[3]/total_mmr),
        "dust_0.55-0.9": (["forecast_period","forecast_reference_time","latitude", "longitude"], mmr[4]/total_mmr),
        "dust_0.9-20": (["forecast_period","forecast_reference_time","latitude", "longitude"], mmr[5]/total_mmr),
        "hydrophilic_black_carbon": (["forecast_period","forecast_reference_time","latitude", "longitude"], mmr[6]/total_mmr),
        "hydrophilic_organic_matter": (["forecast_period","forecast_reference_time","latitude", "longitude"], mmr[7]/total_mmr),
        "hydrophobic_black_carbon": (["forecast_period","forecast_reference_time","latitude", "longitude"], mmr[8]/total_mmr),
        "hydrophobic_organic_matter": (["forecast_period","forecast_reference_time","latitude", "longitude"], mmr[9]/total_mmr),
        "nitrate_coarse_mode": (["forecast_period","forecast_reference_time","latitude", "longitude"], mmr[10]/total_mmr),
        "nitrate_fine_mode": (["forecast_period","forecast_reference_time","latitude", "longitude"], mmr[11]/total_mmr),
        "sea_salt_0.03-0.5": (["forecast_period","forecast_reference_time","latitude", "longitude"], mmr[12]/total_mmr),
        "sea_salt_0.5-5": (["forecast_period","forecast_reference_time","latitude", "longitude"], mmr[13]/total_mmr),
        "sea_salt_5-20": (["forecast_period","forecast_reference_time","latitude", "longitude"], mmr[14]/total_mmr),
        "sulfate": (["forecast_period","forecast_reference_time","latitude", "longitude"], mmr[15]/total_mmr)
    },
    coords={
        "forecast_period": forecast_period,
        "forecast_reference_time": forecast_reference_time,
        "latitude": lat,
        "longitude": lon
    }
)

out_data_xr.to_netcdf(cams_dir+'percentage_mmr_'+month+'_2025.nc')

#ingegrated files can be deleted, only percentage_mmr_april_2025.nc is used
