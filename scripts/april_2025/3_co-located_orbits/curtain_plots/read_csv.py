import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians,
                                 [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

month = 'december'
for month in ['april','june','august','september','december']:
    f = pd.read_csv('detailed_atlid_matches_cartel_'+month+'.csv',delimiter=',')
    aeronet_aod = np.where(f['aeronet_aod']>=0.02,f['aeronet_aod'],np.nan)
    aeronet_lat = np.where(f['aeronet_aod']>=0.02,f['aeronet_lat'],np.nan)
    aeronet_lon = np.where(f['aeronet_aod']>=0.02,f['aeronet_lon'],np.nan)
    atlid_aod = np.where(f['atlid_aod']>=0.02,f['atlid_aod'],np.nan)
    atlid_lat = np.where(f['atlid_aod']>=0.02,f['atlid_lat'],np.nan)
    atlid_lon = np.where(f['atlid_aod']>=0.02,f['atlid_lon'],np.nan)
    
    distances = haversine(aeronet_lat,aeronet_lon,atlid_lat,atlid_lon)
    fig,ax1 = plt.subplots(1,figsize=(8,4))
    ax1.plot(np.arange(len(aeronet_aod)),aeronet_aod,'k-',label='AERONET')
    ax1.plot(np.arange(len(atlid_aod)),atlid_aod,'k:',label='ATLID')
    ax1_1=ax1.twinx() 
    ax1_1.plot(np.arange(len(distances)),distances)
    ax1.legend(frameon=False)
    ax1.set_title(month)
    ax1.set_xlabel('# of points')
    ax1_1.set_ylabel('Distances in km')
    ax1.set_ylabel('AOD')
    fig.tight_layout()
    fig.savefig(month+'_dist_aod.jpg')
