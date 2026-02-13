import pandas as pd
import numpy as np
import xarray as xr
import sys
import os
from scipy.stats import binned_statistic_2d,pearsonr
script_path = '/home/nld6854/earthcare_scripts/scripts/april_2025'
sys.path.append(script_path)
from plotting_tools import statistics
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors

def get_AOD(aod_old, wave_old, wave_new, angstrom):
    return ((wave_new / wave_old) ** (-angstrom)) * aod_old

month = '12'
month2 = 'December'
month3 = 'december'

Te_dir = '/scratch/nld6854/earthcare/modis/Terra/hdffiles/monthly_aod/'
Aq_dir = '/scratch/nld6854/earthcare/modis/Aqua/hdffiles/monthly_aod/'
VIIRS_dir = '/scratch/nld6854/earthcare/modis/VIIRS/hdffiles/monthly_aod/'
atl_dir = '/scratch/nld6854/earthcare/cams_data/'
aeronet_path = '/scratch/nld6854/earthcare/aeronet/'

months = [str(nmonth) if nmonth>9 else '0'+str(nmonth) for nmonth in range(1,13)]
month2s = ['January','February','March','April','May','June','July','August','September','October','November','December']
month3s = ['january','february','march','april','may','june','july','august','september','october','november','december']

for month,month2,month3 in zip(months,month2s,month3s):
    year = '2024' if month3 == 'december' else '2025'
    print('year=',year)
    print('month=',month2)
    print('Month=',month3)

    df_aer = pd.read_table(aeronet_path+year+month+'_all_sites_aod15_allpoints.txt', delimiter=',', header=[7])
    df_aer = df_aer.replace(-999,np.nan)
    angstrom = df_aer['340-440_Angstrom_Exponent']
    
    df_Te = pd.read_csv(Te_dir+year+"_"+month3+"_modis_aeronet_co-located_100km_30min.csv", delimiter=",")
    df_Aq = pd.read_csv(Aq_dir+year+"_"+month3+"_modis_aeronet_co-located_100km_30min.csv", delimiter=",")
    df_VI = pd.read_csv(VIIRS_dir+year+"_"+month3+"_modis_aeronet_co-located_100km_30min.csv", delimiter=",")
    df_atl = pd.read_csv(atl_dir+month3+'_'+year+'/'+year+"_"+month3+"_atlid_aeronet_co-located_100km_10atlid_per_aeronet_2nd_method.csv", delimiter=",")
    
    def read_df(dfmod,dfatl):
        modis_aod = get_AOD(dfmod['co_located_modis'].values,550,355,angstrom)
        aeronet_lat = dfmod['lat'].values
        aeronet_lon = dfmod['lon'].values
    
        atl_aod = dfatl['co_located_atlid'].values
    
        mask = ~np.isnan(modis_aod) & ~np.isnan(atl_aod)
        modis_aod = modis_aod[mask]
        
        atlid_aod = atl_aod[mask]
        aeronet_lat = aeronet_lat[mask]
        aeronet_lon = aeronet_lon[mask]
        
        lat,lon,mod_aod,atl_aods = [],[],[],[]
        for ij in np.unique(aeronet_lat):
            lon.append(aeronet_lon[aeronet_lat==ij][0])
            lat.append(ij)
            atl_aods.append(np.nanmean(atlid_aod[aeronet_lat==ij]))
            mod_aod.append(np.nanmean(modis_aod[aeronet_lat==ij]))
        return mod_aod,atl_aods,lat,lon

    aodA,atlA,lat,lon = read_df(df_Aq,df_atl)
    aodT,atlT,lat_,lon_ = read_df(df_Te,df_atl)
    aodV,atlV,la__,lo__ = read_df(df_VI,df_atl)
    
    bounds = np.linspace(-0.5,0.5,11)
    fig,((ax1,ax4,ax7),(ax2,ax5,ax8),(ax3,ax6,ax9))=plt.subplots(3,3,figsize=(30,18),subplot_kw=dict(projection=ccrs.PlateCarree()))
    ###Terra#######################################################
    ax1.set_extent([-180, 180, -90, 90])#,crs=ccrs.PlateCarree(central_longitude=180))
    gl = ax1.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = False
    
    ax1.coastlines(resolution='110m')#,color='white')
    ax1.gridlines()
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    
    # Plot AOD values
    scatter = ax1.scatter(lon_,lat_,c=aodT,s=60,edgecolor='k',cmap='plasma',
        norm=colors.LogNorm(vmin=1e-2, vmax=1),transform=ccrs.PlateCarree())
    
    bar = plt.colorbar(scatter, orientation='vertical',ax=ax1,shrink=0.7, pad=0.1)
    bar.ax.set_ylabel('MODIS Terra Particle optical depth / -',fontsize=15)
    bar.ax.tick_params(labelsize=15)
    
    ax1.set_title('MODIS Terra AOD '+month2,fontsize=15)
    
    
    ax2.set_extent([-180, 180, -90, 90])#,crs=ccrs.PlateCarree(central_longitude=180))
    gl = ax2.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = False
    
    ax2.coastlines(resolution='110m')#,color='white')
    ax2.gridlines()
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    
    # Plot AOD values
    scatter = ax2.scatter(lon_,lat_,c=atlT,s=60,edgecolor='k',cmap='plasma',
        norm=colors.LogNorm(vmin=1e-2, vmax=1),transform=ccrs.PlateCarree())
    
    bar = plt.colorbar(scatter, orientation='vertical',ax=ax2,shrink=0.7, pad=0.1)
    bar.ax.set_ylabel('ATLID Particle optical depth / -',fontsize=15)
    bar.ax.tick_params(labelsize=15)
    
    ax2.set_title('ATLID AOD '+month2,fontsize=15)
    
    
    gl = ax3.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = False
    
    ax3.coastlines(resolution='110m')#,color='white')

    ax3.gridlines()
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    
    # Plot AOD values
    scatter = ax3.scatter(lon_,lat_,c=np.array(atlT)-np.array(aodT),s=60,edgecolor='k',cmap='RdBu_r',
        norm=colors.BoundaryNorm(boundaries=bounds, ncolors=256),transform=ccrs.PlateCarree())
    
    bar = plt.colorbar(scatter, orientation='vertical',ax=ax3,shrink=0.7, pad=0.1)
    bar.ax.set_ylabel('Differences in Particle optical depth / -',fontsize=15)
    bar.ax.tick_params(labelsize=15)
    
    ax3.set_title('ATLID-MODIS Terra AOD '+month2,fontsize=15)
    
    ###Aqua#######################################################
    ax4.set_extent([-180, 180, -90, 90])#,crs=ccrs.PlateCarree(central_longitude=180))
    gl = ax4.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = False
    
    ax4.coastlines(resolution='110m')#,color='white')
    ax4.gridlines()
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    
    # Plot AOD values
    scatter = ax4.scatter(lon,lat,c=aodA,s=60,edgecolor='k',cmap='plasma',
        norm=colors.LogNorm(vmin=1e-2, vmax=1),transform=ccrs.PlateCarree())
    
    bar = plt.colorbar(scatter, orientation='vertical',ax=ax4,shrink=0.7, pad=0.1)
    bar.ax.set_ylabel('MODIS Aqua Particle optical depth / -',fontsize=15)
    bar.ax.tick_params(labelsize=15)
    
    ax4.set_title('MODIS Aqua AOD '+month2,fontsize=15)
    
    
    ax5.set_extent([-180, 180, -90, 90])#,crs=ccrs.PlateCarree(central_longitude=180))
    gl = ax5.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = False
    
    ax5.coastlines(resolution='110m')#,color='white')
    ax5.gridlines()
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    
    # Plot AOD values
    scatter = ax5.scatter(lon,lat,c=atlA,s=60,edgecolor='k',cmap='plasma',
        norm=colors.LogNorm(vmin=1e-2, vmax=1),transform=ccrs.PlateCarree())

    bar = plt.colorbar(scatter, orientation='vertical',ax=ax5,shrink=0.7, pad=0.1)
    bar.ax.set_ylabel('ATLID Particle optical depth / -',fontsize=15)
    bar.ax.tick_params(labelsize=15)
    
    ax5.set_title('ATLID AOD '+month2,fontsize=15)
    
    
    gl = ax6.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = False
    
    ax6.coastlines(resolution='110m')#,color='white')
    ax6.gridlines()
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    
    # Plot AOD values
    scatter = ax6.scatter(lon,lat,c=np.array(atlA)-np.array(aodA),s=60,edgecolor='k',cmap='RdBu_r',
        norm=colors.BoundaryNorm(boundaries=bounds, ncolors=256),transform=ccrs.PlateCarree())
    
    bar = plt.colorbar(scatter, orientation='vertical',ax=ax6,shrink=0.7, pad=0.1)
    bar.ax.set_ylabel('Differences in Particle optical depth / -',fontsize=15)
    bar.ax.tick_params(labelsize=15)
    
    ax6.set_title('ATLID-MODIS Aqua AOD '+month2,fontsize=15)
    

    ###VIIRS DB#######################################################
    ax7.set_extent([-180, 180, -90, 90])#,crs=ccrs.PlateCarree(central_longitude=180))
    gl = ax7.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = False
    
    ax7.coastlines(resolution='110m')#,color='white')
    ax7.gridlines()
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    
    # Plot AOD values
    scatter = ax7.scatter(lo__,la__,c=aodV,s=60,edgecolor='k',cmap='plasma',
        norm=colors.LogNorm(vmin=1e-2, vmax=1),transform=ccrs.PlateCarree())
    
    bar = plt.colorbar(scatter, orientation='vertical',ax=ax7,shrink=0.7, pad=0.1)
    bar.ax.set_ylabel('MODIS Aqua Particle optical depth / -',fontsize=15)
    bar.ax.tick_params(labelsize=15)
    
    ax7.set_title('VIIRS Deep Blue AOD '+month2,fontsize=15)
    
    
    ax8.set_extent([-180, 180, -90, 90])#,crs=ccrs.PlateCarree(central_longitude=180))
    gl = ax8.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = False
    
    ax8.coastlines(resolution='110m')#,color='white')
    ax8.gridlines()
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    
    # Plot AOD values
    scatter = ax8.scatter(lo__,la__,c=atlV,s=60,edgecolor='k',cmap='plasma',
        norm=colors.LogNorm(vmin=1e-2, vmax=1),transform=ccrs.PlateCarree())

    bar = plt.colorbar(scatter, orientation='vertical',ax=ax8,shrink=0.7, pad=0.1)
    bar.ax.set_ylabel('ATLID Particle optical depth / -',fontsize=15)
    bar.ax.tick_params(labelsize=15)
    
    ax8.set_title('ATLID AOD '+month2,fontsize=15)
    
    gl = ax9.gridlines(crs=ccrs.PlateCarree(central_longitude=0), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = False
    
    ax9.coastlines(resolution='110m')#,color='white')
    ax9.gridlines()
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}
    
    # Plot AOD values
    scatter = ax9.scatter(lo__,la__,c=np.array(atlV)-np.array(aodV),s=60,edgecolor='k',cmap='RdBu_r',
        norm=colors.BoundaryNorm(boundaries=bounds, ncolors=256),transform=ccrs.PlateCarree())
    
    bar = plt.colorbar(scatter, orientation='vertical',ax=ax9,shrink=0.7, pad=0.1)
    bar.ax.set_ylabel('Differences in Particle optical depth / -',fontsize=15)
    bar.ax.tick_params(labelsize=15)
    
    ax9.set_title('ATLID-VIIRS AOD '+month2,fontsize=15)
    


    plt.tight_layout()
    fig.savefig('global_ATLID_MODIS_aod_'+month2+'_'+year+'_100km.jpg',bbox_inches='tight')
    
    atl_aodT = np.array(atlT)
    atl_aodA = np.array(atlA)
    atl_aodV = np.array(atlV)
    terra_aod = np.array(aodT)
    aqua_aod = np.array(aodA)
    viirs_aod = np.array(aodV)
    print('*********Differences between ATLID & Aqua*********')
    mask = ~np.isnan(aqua_aod) & ~np.isnan(atl_aodA)
    print('ATLID mean=',np.nanmean(atl_aodA))
    print('Aqua mean=',np.nanmean(aqua_aod[mask]))
    print('ATLID-Aqua mean=',np.nanmean(atl_aodA[mask]-aqua_aod[mask]))
    print('(ATLID-Aqua)/Aqua mean=',np.nanmean((atl_aodA[mask]-aqua_aod[mask])/aqua_aod[mask]))
    print('Aqua,ATLID NMB=',statistics.normalized_mean_bias(atl_aodA[mask],aqua_aod[mask]))
    print('RMSE=',np.sqrt(np.nanmean((aqua_aod[mask]-atl_aodA[mask])**2)))
    print('ATLID std=',np.std(atl_aodA[mask]))
    print('Aqua std=',np.std(aqua_aod[mask]))
    
    
    r, p_value = pearsonr(aqua_aod[mask],atl_aodA[mask])
    print('Pearson r=',r,'p-value=',p_value)
    
    print('atl_aod.shape',atl_aodA.shape)
    lonpr,latpr = np.meshgrid(lon,lat)
    mask = (latpr>0) & (latpr<22) & (lonpr>-35) & (lonpr<-14)
    a_cams,a_aeronet = np.where(mask,atl_aodA,np.nan),np.where(mask,aqua_aod,np.nan)
    print('West of Africa diffAOD (ATLID-Aqua)=',np.nanmean(a_cams-a_aeronet))
    mask = (latpr>10) & (latpr<30) & (lonpr>-15) & (lonpr<32)
    a_cams,a_aeronet = np.where(mask,atl_aodA,np.nan),np.where(mask,aqua_aod,np.nan)

    print('North of Africa diffAOD (ATLID-Aqua)=',np.nanmean(a_cams-a_aeronet))
    mask = (latpr>21) & (latpr<38) & (lonpr>110) & (lonpr<122)
    a_cams,a_aeronet = np.where(mask,atl_aodA,np.nan),np.where(mask,aqua_aod,np.nan)
    print('East China diffAOD (ATLID-Aqua)=',np.nanmean(a_cams-a_aeronet))
    mask = (latpr>9) & (latpr<21) & (lonpr>93) & (lonpr<110)
    a_cams,a_aeronet = np.where(mask,atl_aodA,np.nan),np.where(mask,aqua_aod,np.nan)
    print('Thailand, Cambodia, Laos, Vietnam diffAOD (ATLID-Aqua)=',np.nanmean(a_cams-a_aeronet))
    print('----------------------------------------------------------------------------------')


    print('*********Differences between ATLID & VIIRS*********')
    mask = ~np.isnan(viirs_aod) & ~np.isnan(atl_aodV)
    print('ATLID mean=',np.nanmean(atl_aodV))
    print('VIIRS mean=',np.nanmean(viirs_aod[mask]))
    print('ATLID-VIIRS mean=',np.nanmean(atl_aodV[mask]-viirs_aod[mask]))
    print('(ATLID-VIIRS)/VIIRS mean=',np.nanmean((atl_aodV[mask]-viirs_aod[mask])/viirs_aod[mask]))
    print('VIIRS,ATLID NMB=',statistics.normalized_mean_bias(atl_aodV[mask],viirs_aod[mask]))
    print('RMSE=',np.sqrt(np.nanmean((viirs_aod[mask]-atl_aodV[mask])**2)))
    print('ATLID std=',np.std(atl_aodV[mask]))
    print('VIIRS std=',np.std(viirs_aod[mask]))
    
    
    r, p_value = pearsonr(viirs_aod[mask],atl_aodV[mask])
    print('Pearson r=',r,'p-value=',p_value)
    
    print('atl_aod.shape',atl_aodV.shape)
    print('viirs_aod.shape',viirs_aod.shape)
    lonpr,latpr = np.meshgrid(lo__,la__)
    mask = (latpr>0) & (latpr<22) & (lonpr>-35) & (lonpr<-14)
    a_cams,a_aeronet = np.where(mask,atl_aodV,np.nan),np.where(mask,viirs_aod,np.nan)
    print('West of Africa diffAOD (ATLID-VIIRS)=',np.nanmean(a_cams-a_aeronet))
    mask = (latpr>10) & (latpr<30) & (lonpr>-15) & (lonpr<32)
    a_cams,a_aeronet = np.where(mask,atl_aodV,np.nan),np.where(mask,viirs_aod,np.nan)

    print('North of Africa diffAOD (ATLID-VIIRS)=',np.nanmean(a_cams-a_aeronet))
    mask = (latpr>21) & (latpr<38) & (lonpr>110) & (lonpr<122)
    a_cams,a_aeronet = np.where(mask,atl_aodV,np.nan),np.where(mask,viirs_aod,np.nan)
    print('East China diffAOD (ATLID-VIIRS)=',np.nanmean(a_cams-a_aeronet))
    mask = (latpr>9) & (latpr<21) & (lonpr>93) & (lonpr<110)
    a_cams,a_aeronet = np.where(mask,atl_aodV,np.nan),np.where(mask,viirs_aod,np.nan)
    print('Thailand, Cambodia, Laos, Vietnam diffAOD (ATLID-VIIRS)=',np.nanmean(a_cams-a_aeronet))
    print('----------------------------------------------------------------------------------')
