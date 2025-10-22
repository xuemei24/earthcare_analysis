import pandas as pd
import numpy as np
import xarray as xr
import sys
import os
from scipy.stats import binned_statistic_2d,pearsonr
script_path = '/home/nld6854/earthcare_scripts/scripts/april_2025'
sys.path.append(script_path)
from plotting_tools import statistics

def get_AOD(aod_old, wave_old, wave_new, angstrom):
    print('aod_old>0',aod_old[aod_old>0], 'wave_old',wave_old, 'wave_new',wave_new, 'angstrom>0',angstrom[angstrom>0])
    return ((wave_new / wave_old) ** (-angstrom)) * aod_old

#Use this script after ATLID AOD has been processed
month = '05'
month2 = 'May'
month3 = 'may'

#-ATLID-------------------------------------------------------------------------
file_dir = '/net/pc190625/nobackup_1/users/wangxu/cams_data/'
print('Month=',month3)
df = pd.read_csv(file_dir+"2025_"+month3+"_atlid_aeronet_co-located_100km.csv", delimiter=",")

atlid_aod = df['atlid_aod'].values
atlid_lat = df['atlid_lat'].values
atlid_lon = df['atlid_lon'].values

aeronet_aod = df['aeronet_aod'].values
aeronet_lat = df['aeronet_lat'].values
aeronet_lon = df['aeronet_lon'].values

print(aeronet_aod.max())
print(aeronet_aod.min())
mask = ~np.isnan(atlid_aod) & ~np.isnan(aeronet_aod)
atlid_aod = atlid_aod[mask]
atlid_lat = atlid_lat[mask]
atlid_lon = atlid_lon[mask]

aeronet_aod = aeronet_aod[mask]
aeronet_lat = aeronet_lat[mask]
aeronet_lon = aeronet_lon[mask]

lat,lon,aer_aod,atl_aod = [],[],[],[]
for ij in np.unique(aeronet_lat):
    print(aeronet_lon[aeronet_lat==ij])
    lon.append(aeronet_lon[aeronet_lat==ij][0])
    lat.append(ij)
    aer_aod.append(np.nanmean(aeronet_aod[aeronet_lat==ij]))
    atl_aod.append(np.nanmean(atlid_aod[aeronet_lat==ij]))

atl_aod = np.array(atl_aod)
aer_aod = np.array(aer_aod)
print('*********Differences between AERONET & ATLID*********')
mask = ~np.isnan(aer_aod) & ~np.isnan(atl_aod)
print('ATLID mean=',np.nanmean(atl_aod))
print('AERONET mean=',np.nanmean(aer_aod[mask]))
print(aer_aod.max())
print(aer_aod.min())
sys.exit()
print('ATLID-AERONET mean=',np.nanmean(atl_aod[mask]-aer_aod[mask]))
print('AERONET,ATLID NMB=',statistics.normalized_mean_bias(atl_aod[mask],aer_aod[mask]))
print('RMSE=',np.sqrt(np.nanmean((aer_aod[mask]-atl_aod[mask])**2)))


r, p_value = pearsonr(aer_aod[mask],atl_aod[mask])
print('Pearson r=',r,'p-value=',p_value)

print('atl_aod.shape',atl_aod.shape)
lonpr,latpr = np.meshgrid(lon,lat)
mask = (latpr>0) & (latpr<22) & (lonpr>-35) & (lonpr<-14)
a_cams,a_aeronet = np.where(mask,atl_aod,np.nan),np.where(mask,aer_aod,np.nan)
print('West of Africa diffAOD (ATLID-AERONET)=',np.nanmean(a_cams-a_aeronet))
mask = (latpr>10) & (latpr<30) & (lonpr>-15) & (lonpr<32)
a_cams,a_aeronet = np.where(mask,atl_aod,np.nan),np.where(mask,aer_aod,np.nan)
print('North of Africa diffAOD (ATLID-AERONET)=',np.nanmean(a_cams-a_aeronet))
mask = (latpr>21) & (latpr<38) & (lonpr>110) & (lonpr<122)
a_cams,a_aeronet = np.where(mask,atl_aod,np.nan),np.where(mask,aer_aod,np.nan)
print('East China diffAOD (ATLID-AERONET)=',np.nanmean(a_cams-a_aeronet))
mask = (latpr>9) & (latpr<21) & (lonpr>93) & (lonpr<110)
a_cams,a_aeronet = np.where(mask,atl_aod,np.nan),np.where(mask,aer_aod,np.nan)
print('Thailand, Cambodia, Laos, Vietnam diffAOD (ATLID-AERONET)=',np.nanmean(a_cams-a_aeronet))

