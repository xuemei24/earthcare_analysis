import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np
import glob

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
AERONET_dir = '/scratch/nld6854/earthcare/aeronet/'

months = [str(nmonth) if nmonth>9 else '0'+str(nmonth) for nmonth in range(1,13)]
month2s = ['January','February','March','April','May','June','July','August','September','October','November','December']
month3s = ['january','february','march','april','may','june','july','august','september','october','november','december']

df_all = []

#cmap = ecplt.colormaps.chiljet2
for month,month2,month3 in zip(months,month2s,month3s):

    year = '2024' if month3 == 'december' else '2025'
    print('year=',year)
    print('month=',month2)
    print('Month=',month3)
    df = pd.read_csv(AERONET_dir+year+month+'_all_sites_aod15_dailyAVG.txt',delimiter=',',header=[7])

    # Parse time (optional, but recommended)
    df["ratio"] = df['340-440_Angstrom_Exponent']/df['440-675_Angstrom_Exponent']
    df = df.dropna(subset=["ratio"])

    geometryAER = [Point(xy) for xy in zip(df["Site_Longitude(Degrees)"], df["Site_Latitude(Degrees)"])]
    gdfAER = gpd.GeoDataFrame(df, geometry=geometryAER, crs="EPSG:4326")

    gdfAER = gpd.sjoin(gdfAER, world, how="left", predicate="intersects")
    # Optional: drop Antarctica / missing
    gdfAER = gdfAER[gdfAER["continent"].notna()]
    gdfAER = gdfAER[gdfAER["continent"] != "Antarctica"]
    continent_order = ["Africa","Asia","Europe","North America","South America","Oceania"]
    meansAER = (gdfAER.groupby("continent")['ratio'].mean().reindex(continent_order))
    print('meansAER',meansAER)
    def plot_diff(df,varname):
        geometry = [Point(xy) for xy in zip(df["Site_Longitude(Degrees)"], df["Site_Latitude(Degrees)"])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
   
        #-------------------------------------------------
        # 4. Spatial join: assign continent
        # -------------------------------------------------
        #gdf = gpd.sjoin(gdf, world, how="left", predicate="within")
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
                         capprops=dict(color="black"),showfliers=False,
                         medianprops=dict(color="black"), flierprops=dict(marker='o',markerfacecolor='none',markeredgecolor='black',markersize=3))
        
        # Axis formatting
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title(varname + " AOD by continent: " + month2 + " " + str(year), fontsize=15)
        plt.xlabel("Continent", fontsize=15)
        plt.ylabel(varname + " (AOD)", fontsize=15)
                
        # --- Add median annotations ---
        medians = (gdf.groupby("continent")[varname].median().reindex(continent_order))
        y_range = gdf[varname].max() - gdf[varname].min()
        y_offset = 0.5 * y_range
 
        mean_aod = gdf.groupby("continent")[varname].mean()
        print('mean_aod',mean_aod)
        median_aod = gdf.groupby("continent")[varname].median()
        percentage_mean = mean_aod/meansAER.replace(0,np.nan)*100.
        print('percentage_mean',percentage_mean)

        for i, continent in enumerate(continent_order):
            median_val = medians.loc[continent]
            perc_val = percentage_mean.loc[continent]

            if pd.notna(median_val) and pd.notna(perc_val):
                ax.text(
                    i,                      # x-position (box index)
                    median_val-median_val*1.5,           # y-position
                    f"{median_val:.3f} ({perc_val:.1f}%)",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color="red",
                    fontweight="bold"
                )
        
        plt.tight_layout()
        plt.savefig(varname + '_ratio_by_continent_box_' + month + '_' + str(year) + '.jpg')
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
        print(month,year,varname,'Mean=',mean_aod,'Median=',median_aod)
 
    plot_diff(df,'ratio')

    df_all.append(df)

df_all = pd.concat(df_all, ignore_index=True)

geometryAER = [Point(xy) for xy in zip(df_all["Site_Longitude(Degrees)"], df_all["Site_Latitude(Degrees)"])]
gdfAER = gpd.GeoDataFrame(df_all, geometry=geometryAER, crs="EPSG:4326")

gdfAER = gpd.sjoin(gdfAER, world, how="left", predicate="intersects")
gdfAER = gdfAER[gdfAER["continent"].notna()]
gdfAER = gdfAER[gdfAER["continent"] != "Antarctica"]

continent_order = ["Africa","Asia","Europe","North America","South America","Oceania"]
meansAER = gdfAER.groupby("continent")["ratio"].mean().reindex(continent_order)

print("Climatological meansAER:")
print(meansAER)

def plot_diff2(df,varname):
    geometry = [Point(xy) for xy in zip(df["Site_Longitude(Degrees)"], df["Site_Latitude(Degrees)"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    #-------------------------------------------------
    # 4. Spatial join: assign continent
    # -------------------------------------------------
    #gdf = gpd.sjoin(gdf, world, how="left", predicate="within")
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
                     capprops=dict(color="black"),showfliers=False,
                     medianprops=dict(color="black"), flierprops=dict(marker='o',markerfacecolor='none',markeredgecolor='black',markersize=3))

    # Axis formatting
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(varname + " AOD by continent: Dec 2024 - Nov 2025", fontsize=15)
    plt.xlabel("Continent", fontsize=15)
    plt.ylabel(varname + " (AOD)", fontsize=15)

    # --- Add median annotations ---
    medians = (gdf.groupby("continent")[varname].median().reindex(continent_order))
    y_range = gdf[varname].max() - gdf[varname].min()
    y_offset = 0.5 * y_range

    mean_aod = gdf.groupby("continent")[varname].mean()
    print('mean_aod',mean_aod)
    median_aod = gdf.groupby("continent")[varname].median()
    percentage_mean = mean_aod/meansAER.replace(0,np.nan)*100.
    print('percentage_mean',percentage_mean)

    for i, continent in enumerate(continent_order):
        median_val = medians.loc[continent]
        perc_val = percentage_mean.loc[continent]

        if pd.notna(median_val) and pd.notna(perc_val):
            ax.text(
                i,                      # x-position (box index)
                median_val-median_val*1.5,           # y-position
                f"{median_val:.3f} ({perc_val:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=12,
                color="red",
                fontweight="bold"
            )
    
    plt.tight_layout()
    plt.savefig(varname + '_ratio_by_continent_box_122024-112025.jpg')
    plt.close()
          
plot_diff2(df_all,'ratio')

