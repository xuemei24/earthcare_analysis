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

Aqua_dir = '/scratch/nld6854/earthcare/modis/Aqua/hdffiles/monthly_aod/'
Terra_dir = '/scratch/nld6854/earthcare/modis/Terra/hdffiles/monthly_aod/'

months = [str(nmonth) if nmonth>9 else '0'+str(nmonth) for nmonth in range(1,13)]
month2s = ['January','February','March','April','May','June','July','August','September','October','November','December']
month3s = ['january','february','march','april','may','june','july','august','september','october','november','december']

cmap = ecplt.colormaps.chiljet2
for month,month2,month3 in zip(months,month2s,month3s):

    year = '2024' if month3 == 'december' else '2025'
    print('year=',year)
    print('month=',month2)
    print('Month=',month3)
    df_Aq = pd.read_csv(Aqua_dir+year+"_"+month3+"_modis_aeronet_co-located_100km_30min.csv", delimiter=",")
    df_Te = pd.read_csv(Terra_dir+year+"_"+month3+"_modis_aeronet_co-located_100km_30min.csv", delimiter=",")

    def read_df(df):
        modis_aod = df['co_located_modis'].values

        aeronet_aod = df['aeronet_aod'].values
        aeronet_lat = df['lat'].values
        aeronet_lon = df['lon'].values

        mask = ~np.isnan(modis_aod) & ~np.isnan(aeronet_aod)
        modis_aod = modis_aod[mask]

        aeronet_aod = aeronet_aod[mask]
        aeronet_lat = aeronet_lat[mask]
        aeronet_lon = aeronet_lon[mask]

        lat,lon,aer_aod,atl_aod = [],[],[],[]
        for ij in np.unique(aeronet_lat):
            lon.append(aeronet_lon[aeronet_lat==ij][0])
            lat.append(ij)
            aer_aod.append(np.nanmean(aeronet_aod[aeronet_lat==ij]))
            atl_aod.append(np.nanmean(modis_aod[aeronet_lat==ij]))
        return np.asarray(atl_aod),np.asarray(aer_aod)
    terra_aod,aer_aodT = read_df(df_Te)
    aqua_aod, aer_aodA = read_df(df_Aq)

    maskT = (terra_aod>=0) & (aer_aodT>=0)
    maskA = (aqua_aod>=0) & (aer_aodA>=0)

    Taod,aaodT = terra_aod[maskT],aer_aodT[maskT]
    Aaod,aaodA = aqua_aod[maskA],aer_aodA[maskA]

    x_bins = np.logspace(np.log10(0.001),np.log10(2),100)
    y_bins = np.logspace(np.log10(0.001),np.log10(2),100)

    #####################################################
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,6))
    c1 = ax1.hist2d(aaodT,Taod,bins=[x_bins, y_bins],cmap=cmap)#, norm=LogNorm())

    x,y = aaodT,Taod 
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
    ax1.set_xlabel('AERONET AOD 550 nm',fontsize=15)
    ax1.set_ylabel('MODIS Terra AOD 550 nm',fontsize=15)
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
    c2 = ax2.hist2d(aaodA,Aaod,bins=[x_bins, y_bins],cmap=cmap)
    ax2.plot(x_line, y_line,color="red",linewidth=2,label=(f"$y = {a:.2f}x + {b:.2e}$\n"f"RMSE = {rmse:.2e}"))

    x,y = aaodA,Aaod
    mask = (np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0))
    x_fit,y_fit = x[mask],y[mask]
    a, b = np.polyfit(x_fit, y_fit, 1)
    y_pred = a * x_fit + b
    rmse = np.sqrt(np.mean((y_fit - y_pred)**2))
    x_line = np.logspace(np.log10(0.001),np.log10(2),200)
    y_line = a * x_line + b

    cb=fig.colorbar(c2[3], ax=ax2, label='')
    cb.ax.tick_params(labelsize=15)
    ax2.set_xlabel('AERONET AOD 550 nm',fontsize=15)
    ax2.set_ylabel('MODIS Aqua AOD 550 nm',fontsize=15)
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



