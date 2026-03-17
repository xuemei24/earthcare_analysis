import pandas as pd
import numpy as np
import xarray as xr
import sys
import os
from scipy.stats import binned_statistic_2d,pearsonr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors

script_path = '/home/nld6854/earthcare_scripts/scripts/april_2025'
sys.path.append(script_path)

from ectools.ectools_edited import ecio
from ectools.ectools_edited import ecplot as ecplt
from ectools.ectools_edited import colormaps
from plotting_tools import statistics,read_h5,ATC_category_colors,projections

df = pd.read_csv('/scratch/nld6854/earthcare/cams_data/122024-112025_detailed_yearly_atlid_aeronet_detailed_matches_50km.csv', delimiter=",")
cmap = ecplt.colormaps.chiljet2

def read_df(df):
    atlid_aod = df['atlid_aod'].values
    #atlid_lat = df['atlid_lat'].values
    #atlid_lon = df['atlid_lon'].values

    aeronet_aod = df['aeronet_aod'].values
    #aeronet_lat = df['aeronet_lat'].values
    #aeronet_lon = df['aeronet_lon'].values

    #mask = ~np.isnan(atlid_aod) & ~np.isnan(aeronet_aod)
    mask = (atlid_aod>=0.02) & (aeronet_aod>=0.02)
    atlid_aod = atlid_aod[mask]

    aeronet_aod = aeronet_aod[mask]
    '''
    aeronet_lat = aeronet_lat[mask]
    aeronet_lon = aeronet_lon[mask]

    lat,lon,aer_aod,atl_aod = [],[],[],[]
    for ij in np.unique(aeronet_lat):
        lon.append(aeronet_lon[aeronet_lat==ij][0])
        lat.append(ij)
        aer_aod.append(np.nanmean(aeronet_aod[aeronet_lat==ij]))
        atl_aod.append(np.nanmean(atlid_aod[aeronet_lat==ij]))
    return np.asarray(atl_aod),np.asarray(aer_aod)
    '''
    return np.asarray(atlid_aod),np.asarray(aeronet_aod)
Aaod,aero_aod = read_df(df)

x_bins = np.logspace(np.log10(0.001),np.log10(2),100)
y_bins = np.logspace(np.log10(0.001),np.log10(2),100)

#####################################################
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,6))
c1 = ax1.hist2d(aero_aod,Aaod,bins=[x_bins, y_bins],cmap=cmap)#, norm=LogNorm())

x,y = aero_aod,Aaod 
mask = (np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0))
x_fit,y_fit = x[mask],y[mask]
a, b = np.polyfit(x_fit, y_fit, 1)
y_pred = a * x_fit + b
rmse = np.sqrt(np.mean((y_fit - y_pred)**2))
x_line = np.logspace(np.log10(0.001),np.log10(2),200)
y_line = a * x_line + b

ax1.plot(x_line, y_line,color="red",linewidth=2,label=(f"$y = {a:.2f}x + {b:.2e}$\n"f"RMSE = {rmse:.2e}"))
ax1.plot([1e-2, 2],[1e-2, 2],'k-')

cb=fig.colorbar(c1[3], ax=ax1, label='')
cb.ax.tick_params(labelsize=15)
ax1.set_xlabel('AERONET AOD 355 nm',fontsize=15)
ax1.set_ylabel('ATLID AOD 355 nm',fontsize=15)
ax1.tick_params(labelsize=15)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(0.01,2)
ax1.set_ylim(0.01,2)
ax1.set_axisbelow(False)  # grid on top of artists
ax1.grid(True,which="both",color='gray',linestyle="--",linewidth=0.7,alpha=0.7)
ax1.legend(frameon=False)
ax1.set_title('ATLID vs AERONET AOD',fontsize=15)

fig.tight_layout()
fig.savefig('122024-112025_correlation_hist_50km.jpg')
plt.close(fig)


########################
#Monthly mean timeseries
########################
def read_df(df,month):
    df = df[df['month']==month]
    atlid_aod = df['atlid_aod'].values
    #atlid_lat = df['atlid_lat'].values
    #atlid_lon = df['atlid_lon'].values

    aeronet_aod = df['aeronet_aod'].values
    #aeronet_lat = df['aeronet_lat'].values
    #aeronet_lon = df['aeronet_lon'].values

    #mask = ~np.isnan(atlid_aod) & ~np.isnan(aeronet_aod)
    mask = (atlid_aod>=0.02) & (aeronet_aod>=0.02)
    atlid_aod = np.asarray(atlid_aod[mask])

    aeronet_aod = np.asarray(aeronet_aod[mask])
    '''
    aeronet_lat = aeronet_lat[mask]
    aeronet_lon = aeronet_lon[mask]

    lat,lon,aer_aod,atl_aod = [],[],[],[]
    for ij in np.unique(aeronet_lat):
        lon.append(aeronet_lon[aeronet_lat==ij][0])
        lat.append(ij)
        aer_aod.append(np.nanmean(aeronet_aod[aeronet_lat==ij]))
        atl_aod.append(np.nanmean(atlid_aod[aeronet_lat==ij]))
    return np.asarray(atl_aod),np.asarray(aer_aod)
    '''
    return np.nanmean(atlid_aod),np.nanstd(atlid_aod,ddof=1),np.nanmean(aeronet_aod),np.nanstd(aeronet_aod,ddof=1)

months = ['december','january','february','march','april','may','june','july','august','september','october','november']
fmonth = ['12-2024','01-2025','02-2025','03-2025','04-2025','05-2025','06-2025','07-2025','08-2025','09-2025','10-2025','11-2025']
atlid_aod = np.array([])
atlid_std = np.array([])
aeronet_aod = np.array([])
aeronet_std = np.array([])
for month in months:
    atl_aod,atl_std,aer_aod,aer_std = read_df(df,month)
    atlid_aod=np.append(atlid_aod,atl_aod)
    aeronet_aod=np.append(aeronet_aod,aer_aod)
    atlid_std=np.append(atlid_std,atl_std)
    aeronet_std=np.append(aeronet_std,aer_std)

fig,ax = plt.subplots(1,figsize=(12,5))
ax.plot(fmonth, aeronet_aod, marker='o', label='AERONET')
ax.fill_between(fmonth,aeronet_aod-aeronet_std,aeronet_aod+aeronet_std,color='blue',alpha=0.3)
ax.plot(fmonth, atlid_aod, marker='s',label='ATLID')
ax.fill_between(fmonth,atlid_aod-atlid_std,atlid_aod+atlid_std,color='orange',alpha=0.3)
ax.set_ylabel("AOD",fontsize=15)
ax.set_title("Monthly AOD Comparison: ATLID vs AERONET",fontsize=15)
ax.set_ylim(-0.1,0.7)
ax.grid(True, linestyle='--', alpha=0.5)
ax.tick_params(labelsize=12)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig('AERONET_ATLID_monthly_mean_aod_50km.jpg')

