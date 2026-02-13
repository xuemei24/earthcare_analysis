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

from ectools import ecio
from ectools import ecplot as ecplt
from ectools import colormaps
from plotting_tools import statistics,read_h5,ATC_category_colors,projections

def get_AOD(aod_old, wave_old, wave_new, angstrom):
    return ((wave_new / wave_old) ** (-angstrom)) * aod_old

file_dir = '/scratch/nld6854/earthcare/cams_data/'
Aq_dir = '/scratch/nld6854/earthcare/modis/Aqua/hdffiles/monthly_aod/'
VI_dir = '/scratch/nld6854/earthcare/modis/VIIRS/hdffiles/monthly_aod/'
aeronet_path = '/scratch/nld6854/earthcare/aeronet/'

months = [str(nmonth) if nmonth>9 else '0'+str(nmonth) for nmonth in range(1,13)]
month2s = ['January','February','March','April','May','June','July','August','September','October','November','December']
month3s = ['january','february','march','april','may','june','july','august','september','october','november','december']

aqua_aod_all,atl_aodA_all = [],[]
viirs_aod_all,atl_aodV_all = [],[]

cmap = ecplt.colormaps.chiljet2
for month,month2,month3 in zip(months,month2s,month3s):
    year = '2024' if month3 == 'december' else '2025'
    print('year=',year)
    print('month=',month2)
    print('Month=',month3)
    df_atl = pd.read_csv(file_dir+month3+'_'+year+'/'+year+"_"+month3+"_atlid_aeronet_co-located_100km_10atlid_per_aeronet_2nd_method.csv", delimiter=",")
    df_aq = pd.read_csv(Aq_dir+year+"_"+month3+"_modis_aeronet_co-located_100km_30min.csv", delimiter=",")
    df_vi = pd.read_csv(VI_dir+year+"_"+month3+"_modis_aeronet_co-located_100km_30min.csv", delimiter=",")
    df_aer = pd.read_table(aeronet_path+year+month+'_all_sites_aod15_allpoints.txt', delimiter=',', header=[7])
    df_aer = df_aer.replace(-999,np.nan)
    angstrom = df_aer['340-440_Angstrom_Exponent']


    def read_df(df_atl,df_aq):
        modis_aod = get_AOD(df_aq['co_located_modis'].values,550,355,angstrom)
        atlid_aod = df_atl['co_located_atlid'].values

        aeronet_lat = df_atl['lat'].values
        aeronet_lon = df_atl['lon'].values

        mask = ~np.isnan(modis_aod) & ~np.isnan(atlid_aod)
        modis_aod = modis_aod[mask]

        atlid_aod = atlid_aod[mask]
        aeronet_lat = aeronet_lat[mask]
        aeronet_lon = aeronet_lon[mask]

        lat,lon,mod_aod,atl_aod = [],[],[],[]
        for ij in np.unique(aeronet_lat):
            lon.append(aeronet_lon[aeronet_lat==ij][0])
            lat.append(ij)
            mod_aod.append(np.nanmean(modis_aod[aeronet_lat==ij]))
            atl_aod.append(np.nanmean(atlid_aod[aeronet_lat==ij]))
        return np.asarray(mod_aod),np.asarray(atl_aod)

    aqua_aod,atlid_aodA = read_df(df_atl,df_aq)
    maskA = (atlid_aodA>=0) & (aqua_aod>=0)
    atl_aodA,aqua_aod = atlid_aodA[maskA],aqua_aod[maskA]

    viirs_aod,atlid_aodV = read_df(df_atl,df_vi)
    maskV = (atlid_aodV>=0) & (viirs_aod>=0)
    atl_aodV,viirs_aod = atlid_aodV[maskV],viirs_aod[maskV]

    x_bins = np.logspace(np.log10(0.001),np.log10(2),100)
    y_bins = np.logspace(np.log10(0.001),np.log10(2),100)

    #####################################################
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,6))
    c1 = ax1.hist2d(aqua_aod,atl_aodA,bins=[x_bins, y_bins],cmap=cmap)#, norm=LogNorm())

    x,y = aqua_aod,atl_aodA
    mask = (np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0))
    x_fit,y_fit = x[mask],y[mask]
    a, b = np.polyfit(x_fit, y_fit, 1)
    y_pred = a * x_fit + b
    rmse = np.sqrt(np.mean((y_fit - y_pred)**2))
    x_line = np.logspace(np.log10(0.001),np.log10(2),200)
    y_line = a * x_line + b

    ax1.plot(x_line, y_line,color="red",linewidth=2,label=(f"$y = {a:.2f}x + {b:.2e}$\n"f"RMSE = {rmse:.2e}"))

    cb=fig.colorbar(c1[3], ax=ax1, label='')
    cb.ax.tick_params(labelsize=15)
    ax1.set_xlabel('MODIS Aqua AOD 355 nm',fontsize=15)
    ax1.set_ylabel('ATLID AOD 355 nm',fontsize=15)
    ax1.tick_params(labelsize=15)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(0.001,2)
    ax1.set_ylim(0.001,2)
    ax1.set_axisbelow(False)  # grid on top of artists
    ax1.grid(True,which="both",color='gray',linestyle="--",linewidth=0.7,alpha=0.7)
    ax1.legend(frameon=False)
    ax1.set_title(month2,fontsize=15)

    ###########################################################
    c2 = ax2.hist2d(viirs_aod,atl_aodV,bins=[x_bins, y_bins],cmap=cmap)#, norm=LogNorm())

    x,y = viirs_aod,atl_aodV
    mask = (np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0))
    x_fit,y_fit = x[mask],y[mask]
    a, b = np.polyfit(x_fit, y_fit, 1)
    y_pred = a * x_fit + b
    rmse = np.sqrt(np.mean((y_fit - y_pred)**2))
    x_line = np.logspace(np.log10(0.001),np.log10(2),200)
    y_line = a * x_line + b

    ax2.plot(x_line, y_line,color="red",linewidth=2,label=(f"$y = {a:.2f}x + {b:.2e}$\n"f"RMSE = {rmse:.2e}"))

    cb=fig.colorbar(c2[3], ax=ax2, label='')
    cb.ax.tick_params(labelsize=15)
    ax2.set_xlabel('VIIRS Deep Blue AOD 355 nm',fontsize=15)
    ax2.set_ylabel('ATLID AOD 355 nm',fontsize=15)
    ax2.tick_params(labelsize=15)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(0.001,2)
    ax2.set_ylim(0.001,2)
    ax2.set_axisbelow(False)  # grid on top of artists
    ax2.grid(True,which="both",color='gray',linestyle="--",linewidth=0.7,alpha=0.7)
    ax2.legend(frameon=False)
    ax2.set_title(month2,fontsize=15)

    fig.tight_layout()
    fig.savefig(month3+'_correlation_hist.jpg')
    plt.close(fig)


    aqua_aod_all.append(aqua_aod)
    atl_aodA_all.append(atl_aodA)

    viirs_aod_all.append(viirs_aod)
    atl_aodV_all.append(atl_aodV)

aqua_aod_all = np.concatenate(aqua_aod_all)
atl_aodA_all = np.concatenate(atl_aodA_all)

viirs_aod_all = np.concatenate(viirs_aod_all)
atl_aodV_all = np.concatenate(atl_aodV_all)

#####################################################
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,6))
c1 = ax1.hist2d(aqua_aod_all,atl_aodA_all,bins=[x_bins, y_bins],cmap=cmap)#, norm=LogNorm())

x,y = aqua_aod_all,atl_aodA_all
mask = (np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0))
x_fit,y_fit = x[mask],y[mask]
a, b = np.polyfit(x_fit, y_fit, 1)
y_pred = a * x_fit + b
rmse = np.sqrt(np.mean((y_fit - y_pred)**2))
x_line = np.logspace(np.log10(0.001),np.log10(2),200)
y_line = a * x_line + b

ax1.plot(x_line, y_line,color="red",linewidth=2,label=(f"$y = {a:.2f}x + {b:.2e}$\n"f"RMSE = {rmse:.2e}"))

cb=fig.colorbar(c1[3], ax=ax1, label='')
cb.ax.tick_params(labelsize=15)
ax1.set_xlabel('MODIS Aqua AOD 355 nm',fontsize=15)
ax1.set_ylabel('ATLID AOD 355 nm',fontsize=15)
ax1.tick_params(labelsize=15)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(0.001,2)
ax1.set_ylim(0.001,2)
ax1.set_axisbelow(False)  # grid on top of artists
ax1.grid(True,which="both",color='gray',linestyle="--",linewidth=0.7,alpha=0.7)
ax1.legend(frameon=False)
ax1.set_title('Dec 2024 - Nov 2025',fontsize=15)

###########################################################
c2 = ax2.hist2d(viirs_aod_all,atl_aodV_all,bins=[x_bins, y_bins],cmap=cmap)#, norm=LogNorm())

x,y = viirs_aod_all,atl_aodV_all
mask = (np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0))
x_fit,y_fit = x[mask],y[mask]
a, b = np.polyfit(x_fit, y_fit, 1)
y_pred = a * x_fit + b
rmse = np.sqrt(np.mean((y_fit - y_pred)**2))
x_line = np.logspace(np.log10(0.001),np.log10(2),200)
y_line = a * x_line + b

ax2.plot(x_line, y_line,color="red",linewidth=2,label=(f"$y = {a:.2f}x + {b:.2e}$\n"f"RMSE = {rmse:.2e}"))

cb=fig.colorbar(c2[3], ax=ax2, label='')
cb.ax.tick_params(labelsize=15)
ax2.set_xlabel('VIIRS Deep Blue AOD 355 nm',fontsize=15)
ax2.set_ylabel('ATLID AOD 355 nm',fontsize=15)
ax2.tick_params(labelsize=15)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(0.001,2)
ax2.set_ylim(0.001,2)
ax2.set_axisbelow(False)  # grid on top of artists
ax2.grid(True,which="both",color='gray',linestyle="--",linewidth=0.7,alpha=0.7)
ax2.legend(frameon=False)
ax2.set_title('Dec 2024 - Nov 2025',fontsize=15)

fig.tight_layout()
fig.savefig('122024-112025_correlation_hist.jpg')
plt.close(fig)

