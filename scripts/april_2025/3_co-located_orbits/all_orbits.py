import h5py
import concurrent.futures
import glob
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
from pyresample import geometry, kd_tree
import sys
import os

script_path = '/usr/people/wangxu/Desktop/earthcare_scripts/scripts/april_2025/'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), script_path)))

from ectools import ecio
from ectools import ecplot as ecplt
from ectools import colormaps
from plotting_tools import read_h5,ATC_category_colors

which_aerosol = 'total'
def read_hdf5(file_path):
    tc_file = file_path.replace('EBD','TC_')

    ATC = ecio.load_ATC(tc_file, prodmod_code="ECA_EXAC")
    AEBD = ecio.load_AEBD(file_path,prodmod_code="ECA_EXAC")
    print(tc_file,file_path)
 
    #Read ATC
    qc = ATC['quality_status']
    tc = ATC['classification_low_resolution']
    #find the -1 in sub-surface
    tct = tc.copy()
    for ni in range(tc.shape[0]):
        for nj in range(1,tc.shape[1]):
            if tc[ni,nj-1]==-2 and tc[ni,nj]==-1:
                tct[ni,nj]=-2
    tc = tct

    tc2 = np.where(tc>=-1,tc,np.nan)
    tc2 = np.where(tc2<=2,tc2,np.nan)
    tc2 = np.where(tc2!=0,tc2,np.nan)
    tc = np.where(qc<2,tc,np.nan)

    if which_aerosol == 'total':
        #tc = np.where(tc>=10,tc,np.nan) # missing, sub-surface, attenuated, clear, liquid, supercooled liquid, ice
        #tc = np.where(tc<=26,tc,np.nan) # not necessarily needed
        #tc = np.where(tc!=20,tc,np.nan) # STS PSC
        #tc = np.where(tc!=21,tc,np.nan) # NAT PSC
        #tc = np.where(tc!=22,tc,np.nan) # stratospheric ice
        #tc = np.where(tc!=25,tc,np.nan) # stratospheric ash
        #tc = np.where(tc<=15,tc,np.nan) # no stratospheric particles
        aer_index = np.array([10,11,12,13,14,15,25,26,27])
        tc = np.where(np.isin(tc,aer_index),tc,np.nan)

    elif which_aerosol == 'sea_salt':
        tc = np.where(tc==11,tc,np.nan)
    elif which_aerosol == 'dust':
        tc = np.where(tc==15,tc,np.nan) #there also dusty smoke =14 & dusty mix =15

    #Read AEBD
    data = AEBD['particle_extinction_coefficient_355nm_low_resolution']
    data2 = AEBD['particle_extinction_coefficient_355nm_low_resolution_error']
    lat = AEBD['latitude']
    lon = AEBD['longitude']
    qc = AEBD['quality_status']
    #height = np.where(height>=0,height,np.nan)
    data = np.where(qc<2,data,0)
    data = np.where(data<1e-3,data,0)
    data = np.where(data>=0,data,0)
    data = np.where(tc>0,data,0)  #only keep aerosol particles

    #mask the columns when there are liquid clouds or fully attenuated
    mask = (tc2>=-1).any(axis=1)
    data[mask,:] = np.nan

    #Trim the rows when height is negative
    height = AEBD['height']
    #mask = (height<0).any(axis=0)
    #data = data[:,~mask]
    #height = height[:,~mask]

    reversed_data = data[:,::-1]
    reversed_height = height[:,::-1]
    aod = np.trapz(reversed_data,x=reversed_height,axis=1)

    #start plotting
 
    nrows=4
    suffix = '_low_resolution'
    hmax=20e3
  
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, gridspec_kw={'hspace':0.75})
    print(AEBD['time'].shape,aod.shape)
    print(AEBD)
    #print(aod[aod>0])
    ecplt.plot_EC_1Dxw(axes[0], AEBD, {'AEBD':{'xdata':AEBD['time'][:],'ydata':aod}},#AEBD['time'], 'ydata':aod[0]}},
                 "Aerosol optical depth", "AOD / -", timevar='time', include_ruler=False)
  
    ecplt.plot_EC_2D(axes[1], AEBD, 'particle_extinction_coefficient_355nm_low_resolution', r"$\alpha_\mathrm{mie}$", cmap=ecplt.colormaps.calipso, plot_scale='log', plot_range=[1e-6,1e-3], units='m$^{-1}$', hmax=hmax, plot_where=AEBD.particle_extinction_coefficient_355nm > 1e-6)
  
    ecplt.plot_EC_2Dxw(axes[2], AEBD, 'particle_extinction_coefficient_355nm_low_resolution','particle_extinction_coefficient_355nm_low_resolution_error', r"-", cmap=ecplt.colormaps.calipso, plot_scale='linear', title='Extinction coefficient SNR', plot_range=[0,30],units='-', hmax=hmax, plot_where=AEBD.particle_extinction_coefficient_355nm > 1e-6)

    ecplt.plot_EC_target_classification(axes[3], ATC, 'classification_low_resolution', ecplt.ATC_category_colors, hmax=hmax)
  
  
    ecplt.add_subfigure_labels(axes)
  
    dstdir = "/usr/people/wangxu/Desktop/earthcare_scripts/scripts/april_2025/3_co-located_orbits/slices_all_orbits/"
    srcfile_string = AEBD.encoding['source'].split("/")[-1].split(".")[0]
    dstfile = f"{srcfile_string}_quicklook{suffix}_aod_EBD_TC.png"
    fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')

# List of file paths you want to read concurrently
file_paths = glob.glob('/net/pc190625/nobackup_1/users/wangxu/earthcare_data/april_2025/EBD/ECA_EXAE_ATL_EBD_2A_*04795A*.h5')
file_paths.sort()

## Use ProcessPoolExecutor to read files concurrently
#with concurrent.futures.ProcessPoolExecutor() as executor:
#    executor.map(read_hdf5, file_paths)

for f_path in file_paths:
    read_hdf5(f_path)
