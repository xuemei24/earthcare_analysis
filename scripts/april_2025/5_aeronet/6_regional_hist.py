import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
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
# 1b. Define Amazon region
# -------------------------------------------------
# Amazon basin approximate bounds
amazon_bounds = box(-80, -20, -45, 5)  # (minx, miny, maxx, maxy)
amazon_region = gpd.GeoDataFrame({'continent': ['Amazon'], 'geometry': [amazon_bounds]}, crs="EPSG:4326")

# -------------------------------------------------
# 2. Read all CSV files
# -------------------------------------------------
Aqua_dir = '/scratch/nld6854/earthcare/modis/Aqua/hdffiles/monthly_aod/'
Terra_dir = '/scratch/nld6854/earthcare/modis/Terra/hdffiles/monthly_aod/'
VIIRS_dir = '/scratch/nld6854/earthcare/modis/VIIRS/hdffiles/monthly_aod/'
ATLID_dir = '/scratch/nld6854/earthcare/cams_data/'

months = [str(nmonth) if nmonth>9 else '0'+str(nmonth) for nmonth in range(1,13)]
month2s = ['January','February','March','April','May','June','July','August','September','October','November','December']
month3s = ['january','february','march','april','may','june','july','august','september','october','november','december']

df_Aq_all = []
df_Te_all = []
df_VI_all = []
df_AT_all = []

# -------------------------------------------------
# Function to assign continents INCLUDING Amazon
# -------------------------------------------------
def assign_continent_with_amazon(df):
    """Assign continent to each point, with Amazon as separate region"""
    geometry = [Point(xy) for xy in zip(df["aeronet_lon"], df["aeronet_lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # First, check if points are in Amazon
    gdf_amazon = gpd.sjoin(gdf, amazon_region, how="left", predicate="intersects")
    
    # Then get general continents
    gdf_continent = gpd.sjoin(gdf, world, how="left", predicate="intersects")
    
    # Combine: Amazon takes precedence over South America
    gdf['continent'] = gdf_continent['continent']
    amazon_mask = gdf_amazon['continent'] == 'Amazon'
    gdf.loc[amazon_mask, 'continent'] = 'Amazon'
    
    # Filter out Antarctica and NaNs
    gdf = gdf[gdf["continent"].notna()]
    gdf = gdf[gdf["continent"] != "Antarctica"]
    
    return gdf

# -------------------------------------------------
# NEW: Dual histogram plotting function (AERONET vs Dataset)
# -------------------------------------------------
def plot_dual_histograms_by_continent(df, aeronet_var, dataset_var, dataset_name, 
                                     filename_suffix="", title_suffix=""):
    """
    Plot side-by-side histograms for AERONET and corresponding dataset
    
    Parameters:
    -----------
    df : DataFrame with 'lat', 'lon', aeronet_var, and dataset_var columns
    aeronet_var : str, name of AERONET variable (e.g., 'aeronet_aod')
    dataset_var : str, name of dataset variable (e.g., 'co_located_modis', 'co_located_atlid')
    dataset_name : str, name for labeling (e.g., 'Aqua', 'Terra', 'ATLID')
    filename_suffix : str, added to output filename
    title_suffix : str, added to plot title
    """
    
    gdf = assign_continent_with_amazon(df)
    
    # Define continent order including Amazon
    continent_order = ["Africa", "Asia", "Europe", "North America", 
                      "South America", "Amazon", "Oceania"]
    
    # Filter to only continents that have data
    available_continents = [c for c in continent_order if c in gdf['continent'].values]
    n_continents = len(available_continents)
    
    if n_continents == 0:
        print(f"No data for {dataset_name}{filename_suffix}")
        return
    
    # Create 4x2 grid (4 rows, 2 columns)
    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    axes = axes.flatten()  # Flatten to 1D array for easy indexing
    
    # Define bins (same for both datasets to allow comparison)
    # Adjust range based on your data
    bins = np.linspace(0, 1.0, 50)  # AOD range from 0 to 1
    
    for i, continent in enumerate(available_continents):
        ax = axes[i]
        
        # Get data for this continent
        continent_data = gdf[gdf['continent'] == continent]
        aeronet_data = continent_data[aeronet_var].dropna()
        dataset_data = continent_data[dataset_var].dropna()
        
        # Plot both histograms
        ax.hist(aeronet_data, bins=bins, alpha=0.6, edgecolor='black', 
                label=f'AERONET (N={len(aeronet_data)})', color='grey', density=True)
        ax.hist(dataset_data, bins=bins, alpha=0.6, edgecolor='black', 
                label=f'{dataset_name} (N={len(dataset_data)})', color='red', density=True)
        
        # Add statistics for both datasets
        aer_mean = aeronet_data.mean()
        aer_median = aeronet_data.median()
        dat_mean = dataset_data.mean()
        dat_median = dataset_data.median()
        
        # Add vertical lines for means
        ax.axvline(aer_mean, color='blue', linestyle='--', linewidth=1.5, alpha=0.8,label='AERONET mean')
        ax.axvline(dat_mean, color='red', linestyle='--', linewidth=1.5, alpha=0.8,label=dataset_name+' mean')
        
        # Labels and title
        ax.set_xlabel('AOD 355 nm' if dataset_name=='ATLID' else 'AOD 550 nm', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{continent}\nAERONET: μ={aer_mean:.3f}, {dataset_name}: μ={dat_mean:.3f}', 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_continents, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(f'AERONET vs {dataset_name} AOD Histograms by Continent{title_suffix}', 
                fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(f'AERONET_vs_{dataset_name}_histograms_by_continent{filename_suffix}.jpg', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved dual histograms for AERONET vs {dataset_name}{filename_suffix}")

   
df_AT = pd.read_csv('/scratch/nld6854/earthcare/cams_data/122024-112025_detailed_yearly_atlid_aeronet_detailed_matches.csv', delimiter=",")
df_AT = df_AT.dropna(subset=["aeronet_aod", "atlid_aod", "aeronet_lat", "aeronet_lon"])

plot_dual_histograms_by_continent(df_AT, 'aeronet_aod', 'atlid_aod', 'ATLID',
                                 '_122024-112025', ': Dec 2024 - Nov 2025')

print("\nDone! All dual histograms generated.")
