import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import glob
import numpy as np

def get_AOD(aod_old, wave_old, wave_new, angstrom):
    return ((wave_new / wave_old) ** (-angstrom)) * aod_old

# -------------------------------------------------
# 1. Load continent boundaries
# -------------------------------------------------
world = gpd.read_file("ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")

# Keep only needed columns
world = world[["CONTINENT", "geometry"]].rename(columns={"CONTINENT": "continent"})
world = world.to_crs("EPSG:4326")

# -------------------------------------------------
# 2. Read all CSV files
# -------------------------------------------------
Aqua_dir = '/scratch/nld6854/earthcare/modis/Aqua/hdffiles/monthly_aod/'
VIIRS_dir = '/scratch/nld6854/earthcare/modis/VIIRS/hdffiles/monthly_aod/'
ATLID_dir = '/scratch/nld6854/earthcare/cams_data/'
aeronet_path = '/scratch/nld6854/earthcare/aeronet/'

months = [str(nmonth) if nmonth>9 else '0'+str(nmonth) for nmonth in range(1,13)]
month2s = ['January','February','March','April','May','June','July','August','September','October','November','December']
month3s = ['january','february','march','april','may','june','july','august','september','october','november','december']

#cmap = ecplt.colormaps.chiljet2
for month,month2,month3 in zip(months,month2s,month3s):

    year = '2024' if month3 == 'december' else '2025'
    print('year=',year)
    print('month=',month2)
    print('Month=',month3)
    df_Aq = pd.read_csv(Aqua_dir+year+"_"+month3+"_modis_aeronet_co-located_100km_30min.csv", delimiter=",")
    df_VI = pd.read_csv(VIIRS_dir+year+"_"+month3+"_modis_aeronet_co-located_100km_30min.csv", delimiter=",")
    df_AT = pd.read_csv(ATLID_dir+month3+'_'+year+'/'+year+"_"+month3+"_atlid_aeronet_co-located_100km_10atlid_per_aeronet_2nd_method.csv", delimiter=",")
    df_aer = pd.read_table(aeronet_path+year+month+'_all_sites_aod15_allpoints.txt', delimiter=',', header=[7])
    df_aer = df_aer.replace(-999,np.nan)
    angstrom = df_aer['340-440_Angstrom_Exponent']

    df_Aq['co_located_modis'] = get_AOD(df_Aq['co_located_modis'].values,550,355,angstrom)
    df_Aq = df_Aq.dropna(subset=["co_located_modis", "aeronet_aod"])

    df_VI['co_located_modis'] = get_AOD(df_VI['co_located_modis'].values,550,355,angstrom)
    df_VI = df_VI.dropna(subset=["co_located_modis", "aeronet_aod"])
    df_AT = df_AT.dropna(subset=["co_located_atlid", "aeronet_aod"])

    # Parse time (optional, but recommended)
    df_Aq["time"] = pd.to_datetime(df_Aq["time"])
    df_VI["time"] = pd.to_datetime(df_VI["time"])
    df_AT["time"] = pd.to_datetime(df_AT["time"])

    # -------------------------------------------------
    # 3. Convert to GeoDataFrame
    # -------------------------------------------------
    df_Aq['diff_ATLID_Aqua'] = df_AT['co_located_atlid']-df_Aq['co_located_modis']
    df_Aq = df_Aq.dropna(subset=["diff_ATLID_Aqua", "lat", "lon"])

    df_VI['diff_ATLID_VIIRS'] = df_AT['co_located_atlid']-df_VI['co_located_modis']
    df_VI = df_VI.dropna(subset=["diff_ATLID_VIIRS", "lat", "lon"])

    def plot_diff(df,varname):
        geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
   
        #-------------------------------------------------
        # 4. Spatial join: assign continent
        # -------------------------------------------------
        #sgdf = gpd.sjoin(gdf, world, how="left", predicate="within")
        gdf = gpd.sjoin(gdf, world, how="left", predicate="intersects")

        
        # Optional: drop Antarctica / missing
        gdf = gdf[gdf["continent"].notna()]
        gdf = gdf[gdf["continent"] != "Antarctica"]
        
        # -------------------------------------------------
        # 5. Plot AOD by continent
        # -------------------------------------------------
        import seaborn as sns
        plt.figure(figsize=(10, 6))
        continent_order = ["Africa","Asia","Europe","North America","South America","Oceania"]

        ax = sns.boxplot(data=gdf, x="continent", y=varname,order=continent_order,fill=False,
                         boxprops=dict(color="black"),
                         whiskerprops=dict(color="black"),
                         capprops=dict(color="black"),
                         medianprops=dict(color="black"),
                         flierprops=dict(marker='o',markerfacecolor='none',markeredgecolor='black',markersize=3))

        # Axis formatting
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title(varname + " AOD by continent: " + month2 + " " + str(year), fontsize=15)
        plt.xlabel("Continent", fontsize=15)
        plt.ylabel(varname + " (AOD)", fontsize=15)
        plt.ylim(-3,3)
                
        # --- Add median annotations ---
        medians = (gdf.groupby("continent")[varname].median().reindex(continent_order))

        y_range = gdf[varname].max() - gdf[varname].min()
        y_offset = 0.5 * y_range

        for i, median_val in enumerate(medians):
            if pd.notna(median_val):
                ax.text(i,                      # x-position (box index)
                        median_val-y_offset,         # y-position
                        f"{median_val:.3f}",    # text
                        ha="center",va="bottom",fontsize=12,color="red",fontweight="bold")
        
        plt.tight_layout()
        plt.savefig(varname + '_aod_by_continent_box_' + month + '_' + str(year) + '.jpg')
        plt.close()
            
        '''
        plt.figure(figsize=(20, 6))
       
        gdf.boxplot(column=varname, by="continent", figsize=(8, 5))
        plt.suptitle(month2+' '+str(year),fontsize=15)
        plt.title(varname+" AOD by continent",fontsize=15)
        plt.ylabel("AOD",fontsize=15)
        plt.tick_params(labelsize=15)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(varname+'_aod_by_continent_box_'+month+'_'+str(year)+'.jpg')
        '''
        mean_aod = gdf.groupby("continent")[varname].mean()
        median_aod = gdf.groupby("continent")[varname].median()
        print(month,year,varname,'Mean=',mean_aod,'Median=',median_aod)
 
    plot_diff(df_Aq,'diff_ATLID_Aqua')
    plot_diff(df_VI,'diff_ATLID_VIIRS')
