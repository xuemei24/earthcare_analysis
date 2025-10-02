"""
Copyright 2023- ECMWF

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Shannon Mason, Bernat Puigdomenech Treserras, Anja Hunerbein, Nicole Docter"
__copyright__ = "Copyright 2023- ECMWF"
__license__ = "Apache license version 2.0"
__maintainer__ = "Shannon Mason"
__email__ = "shannon.mason@ecmwf.int"

import xarray as xr
import numpy as np

import os
from glob import glob

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def from_dB(v):
    """Convert v to linear units from dB.
    
    To do:
    - update units
    """
   
    
    if type(v) in [xr.core.dataset.Dataset, xr.core.dataarray.DataArray]:
        v_dB = xr.DataArray(10.**(v/10.), coords=v.coords, attrs=v.attrs)
        return v_dB
    else:
        return 10.**(v/10.)
    
def to_dB(v):
    """Convert v to dB from linear units.
        
    To do:
    - update units
    """
    if type(v) in [xr.core.dataset.Dataset, xr.core.dataarray.DataArray]:
        v_linear = xr.DataArray(10.*np.log10(v), coords=v.coords, attrs=v.attrs)
        return v_linear
    else:
        return 10.*np.log10(v)

###
# direct copy from read_and_plot_l1_msi_rgr.py see PDGS confluence table - except specRad2Refl() ... removed fill_value handeling etc here 
def cosd(a):
    return np.cos(a*np.pi/180.)
def sind(a):
    return np.sin(a*np.pi/180.)

def obs_angle_conv(sea,vea,saa,vaa):
    sza = 90.-sea
    vza = 90.-vea
    raa = vaa -180. - saa 
    notok = np.where(raa < -180.)
    raa[notok]+=360.
    notok = np.where(raa >  180)
    raa[notok]-=360.
    notok = np.where(raa < 0)
    raa[notok]*=-1.
    sca=np.arccos(-1.*cosd(sza)*cosd(vza)+sind(sza)*sind(vza)*cosd(raa))* 180./np.pi
    return sza,vza,raa,sca

def specRad2Refl(pxlval,solirr,sza): #### TODO: needs extra fill_value handling here
    # unit: [1]
    Refl = np.empty_like(pxlval[0:4,:,:])
    dum  = np.pi/cosd(sza[:,:])
    for i in range(4): 
        for k in range(pxlval.shape[2]):
            for kk in range(pxlval.shape[1]):
                Refl[i,kk,k] = (pxlval[i,kk,k]*dum[kk,k])/solirr[i,k]   
    return Refl
###

def drop_dim(ds, dimname):
    return ds.drop([v for v in ds.data_vars if dimname in ds[v].dims])

frame_limits = {'A': [-22.5,  22.5], 'B': [ 22.5,  67.5], 'C':[ 67.5,  67.5], 'D':[ 67.5,  22.5], 
                'E': [ 22.5, -22.5], 'F': [-22.5, -67.5], 'G':[-67.5, -67.5], 'H':[-67.5, -22.5]}

get_frame = lambda ds: ds.encoding['source'].split('/')[-1].split('.')[0].split("_")[-1][-1]

def get_frame_alongtrack(ds, along_track_dim='along_track', latvar='latitude'):    
    lat_framestart, lat_framestop = frame_limits[get_frame(ds)]
    
    i_halfway = len(ds[along_track_dim])//2
    i_framestart = np.argmin(np.abs(ds[latvar].values[:i_halfway] - lat_framestart))
    i_framestop  = i_halfway + np.argmin(np.abs(ds[latvar].values[i_halfway:] - lat_framestop))

    print(f"Selecting frame from {i_framestart} to {i_framestop}")
    
    return i_framestart, i_framestop


def trim_to_frame(ds, along_track_dim='along_track', latvar='latitude', isel_nadir=None, sel_nadir=None):
    if isel_nadir is not None and sel_nadir is not None:
        print("Specify either isel_nadir or sel_nadir, but not both.")

    if isel_nadir:
        assert isinstance(isel_nadir, dict), "isel_nadir should be a dictionary"
        return ds.isel({along_track_dim:slice(*get_frame_alongtrack(ds.isel(isel_nadir), along_track_dim=along_track_dim, latvar=latvar))})
    
    elif sel_nadir:
        assert isinstance(sel_nadir, dict), "sel_nadir should be a dictionary"
        return ds.isel({along_track_dim:slice(*get_frame_alongtrack(ds.sel(sel_nadir), along_track_dim=along_track_dim, latvar=latvar))})
    
    else:
        return ds.isel({along_track_dim:slice(*get_frame_alongtrack(ds, along_track_dim=along_track_dim, latvar=latvar))})


def trim_to_islice(ds, islice, along_track_dim='along_track', timevar='time'):
    if isinstance(islice, dict):
        return ds.isel(islice)
        
    elif isinstance(islice, slice):
        return ds.isel({along_track_dim:islice})
        
    else:
        assert isinstance(islice, dict) or isinstance(islice, slice), "islice should either a slice or a dictionary"
   

def trim_to_timeslice(ds, timeslice, along_track_dim='along_track', timevar='time'):
    assert isinstance(timeslice, dict) or isinstance(timeslice, slice), "timeslice should be a slice or a dictionary"
    _ds = ds.swap_dims({along_track_dim:timevar})
    return _ds.sel(timeslice).swap_dims({timevar:along_track_dim})

                   
def trim_to_latslice(ds, latslice, along_track_dim='along_track', latitudevar='latitude', isel_nadir=None, sel_nadir=None):
    if (isel_nadir is not None) & (sel_nadir is not None):
        print("Specify either isel_nadir or sel_nadir, but not both.")
    
    _ds = ds.swap_dims({along_track_dim:latitudevar})
    if isinstance(latslice, dict):
        return _ds.sel(latslice).swap_dims({latitudevar:along_track_dim})
        
    elif isinstance(latslice, slice):
        return _ds.sel({latitudevar:latslice}).swap_dims({latitudevar:along_track_dim})
        
    else:
        assert isinstance(islice, dict) or isinstance(islice, slice), "islice should either a slice or a dictionary"
    
                   
def load_EC_product(srcglob, trim=True, 
                    group='ScienceData', i=-1, renamer={}, 
                    alongtrackvar='along_track', timevar='time', latitudevar='latitude',
                    isel_nadir=None, sel_nadir=None, preprocess_for_NaT=False,
                    verbose=True):
    srcfiles = sorted(glob(srcglob))
    
    if len(srcfiles) > 1:
        if verbose:
            print(f"{len(srcfiles)} EarthCARE product files match path {srcglob}; selecting {i}")
        srcfile = srcfiles[i]
    elif len(srcfiles) == 1:
        if verbose:
            (f"Only one EarthCARE product files matches path .")
        srcfile = srcfiles[-1]
    else:
        print(f"No EarthCARE product files match path {srcglob}.")
        return None

    if preprocess_for_NaT:
        ds = xr.open_dataset(srcfile, group=group, decode_times=False)
        ds[timevar] = ds[timevar].where(ds[timevar] < 1e36)
        ds = xr.decode_cf(ds)
    else:
        ds = xr.open_dataset(srcfile, group=group).rename(renamer)
        
    if alongtrackvar in ds.dims:
        ds[alongtrackvar] = ds[alongtrackvar]
        
    if isinstance(trim, dict):
        if list(trim.keys())[0] == timevar:
            print("Trimming by time")
            ds = trim_to_timeslice(ds, trim)
        elif list(trim.keys())[0] == latitudevar:
            print("Trimming by latitude")
            ds = trim_to_latslice(ds, trim)
        else:
            print("trim should be a dictionary with { ")
    elif trim:
        ds = trim_to_frame(ds, along_track_dim=alongtrackvar, latvar=latitudevar, isel_nadir=isel_nadir, sel_nadir=sel_nadir)
    return ds


def get_filelist(srcpath, prodmod_code="*", product_code="*", frame_datetime="*", production_datetime="*", frame_code="*", 
                flat_directories=False):

    if flat_directories:
        srcpath = f"{srcpath}/{prodmod_code}_{product_code}_{frame_datetime}_{production_datetime}_{frame_code}.h5"
    else:
        srcpath = f"{srcpath}/{prodmod_code}_{product_code}_{frame_datetime}_{production_datetime}_{frame_code}/" + \
                f"{prodmod_code}_{product_code}_{frame_datetime}_{production_datetime}_{frame_code}.h5"

    return sorted(glob(srcpath))

def get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure):
    import pandas as pd
    
    if not srcpath.endswith(".h5"):
        if nested_directory_structure:
            if frame_datetime != "*":
                frame_dt = pd.to_datetime(frame_datetime.split("T")[0])
                
                srcdir  = f"{srcpath}/{frame_dt:%Y}/{frame_dt:%m}/{frame_dt:%d}"
                srcstr  = f"{prodmod_code}_{product_code}_{frame_datetime}_{production_datetime}_{frame_code}"
                srcpath = f"{srcdir}/{srcstr}/{srcstr}.h5"
                
            else:
                srcdir  = f"{srcpath}/*/*/*"
                srcstr  = f"{prodmod_code}_{product_code}_{frame_datetime}_{production_datetime}_{frame_code}"
                srcpath = f"{srcdir}/{srcstr}/{srcstr}.h5"
            
        else:    
            srcstr  = f"{prodmod_code}_{product_code}_{frame_datetime}_{production_datetime}_{frame_code}"
            srcpath = f"{srcpath}/{srcstr}/{srcstr}.h5"

    return srcpath

########################
##### AUX products #####
########################

def load_XJSG(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="AUX_JSG_1D", 
              frame_datetime="*", production_datetime="*", frame_code="*",
              trim=False, group='ScienceData', i=-1, nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)
    
    ds = load_EC_product(srcpath, trim=trim, group=group, i=i)
    
    if ds:
        ds.time.attrs['units'] = 's since 2000-01-01 00:00:00.00'
        ds = xr.decode_cf(ds)
        ds.time.attrs['units'] = 's since 2000-01-01 00:00:00.00'
    return ds


def load_XMET(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="AUX_MET_1D", 
              frame_datetime="*", production_datetime="*", frame_code="*",
              trim=False, group='ScienceData', i=-1, nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    if trim:
        print("Can't trim X-MET")
        trim = False
        
    ds = load_EC_product(srcpath, trim=trim, group=group, i=i)
    return ds


#######################
##### L1 products #####
#######################

##### ATLID #####

def load_ANOM(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="ATL_NOM_1B", 
              frame_datetime="*", production_datetime="*", frame_code="*",
              trim=False, group='ScienceData', i=-1, latvar='ellipsoid_latitude', nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    ds = load_EC_product(srcpath, trim=False, group=group, i=i, latitudevar=latvar)
    
    if ds:
        ds['latitude'] = ds.ellipsoid_latitude
        ds['longitude'] = ds.ellipsoid_longitude
        
    # Sometimes need to merge/rename some dummy dimensions
    renamer={}
    for d in ['dim_0', 'dim_1']:
        if d in ds.dims:
            if len(ds[d]) == len(ds['along_track']):
                renamer.update({d:'along_track'})
            elif len(ds[d]) == len(ds['height']):
                renamer.update({d:'height'})
    ds = ds.rename(renamer)     

    # Couldn't trim until geo group merged with data group
    if trim:
        ds = trim_to_frame(ds, latvar=latvar)
    

    return ds

##### CPR #####
    
def load_CNOM(srcpath, prodmod_code="ECA_JX[A-Z][A-Z]", product_code="CPR_NOM_1B",
              frame_datetime="*", production_datetime="*", frame_code="*",
              trim=True, i=-1, convert_to_dB=True,
              renamer_datagroup={'phony_dim_10':'along_track', 'phony_dim_11':'CPR_height'},
              renamer_geogroup ={'phony_dim_14':'along_track', 'phony_dim_15':'CPR_height'}, 
              nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)
        
    ds = load_EC_product(srcpath, trim=False, group='ScienceData/Data', renamer=renamer_datagroup, i=i)
    
    if ds:
        # merging in data from the Geo group (for geolocation)
        _geo = load_EC_product(srcpath, trim=False, group='ScienceData/Geo', renamer=renamer_geogroup, i=i, verbose=False)
        ds = xr.merge([ds, _geo])
        ds.encoding['source'] = _geo.encoding['source']
        _geo.close()
        
        # convert some variables to decibels
        if convert_to_dB:
            for v in ['radarReflectivityFactor', 'noiseFloorPower']:
                ds[v] = to_dB(ds[v])
    
        # Couldn't trim until geo group merged with data group
        if trim:
            ds = trim_to_frame(ds, latvar='latitude')
    
    return ds

##### MSI #####

def load_MRGR(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="MSI_RGR_1C", 
              frame_datetime="*", production_datetime="*", frame_code="*", 
              trim=True, i=-1, group='ScienceData', idx_nadir=284, nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    ds = load_EC_product(srcpath, trim=False, group=group, i=i)
                         
    if ds:
        sel_nadir={'across_track':idx_nadir}
        if trim:
            ds = trim_to_frame(ds, latvar='latitude', sel_nadir=sel_nadir)
            
        ds = ds.assign(selected_latitude=ds["latitude" ].isel(sel_nadir), 
             selected_longitude=ds["longitude"].isel(sel_nadir),
               VIS  = ds["pixel_values"].isel({'band':0}),
               NIR  = ds["pixel_values"].isel({'band':1}),
               SWIR1= ds["pixel_values"].isel({'band':2}),
               SWIR2= ds["pixel_values"].isel({'band':3}),
               TIR1 = ds["pixel_values"].isel({'band':4}),
               TIR2 = ds["pixel_values"].isel({'band':5}),
               TIR3 = ds["pixel_values"].isel({'band':6})) 

    return ds


def load_MNOM(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="MSI_NOM_1B", 
              frame_datetime="*", production_datetime="*", frame_code="*", 
              trim=True, i=-1, group='ScienceData', idx_nadir=284, nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    ds = load_EC_product(srcpath, trim=False, group=group, i=i)
                         
    if ds:
        sel_nadir={'across_track':idx_nadir}
        if trim:
            ds = trim_to_frame(ds, latvar='latitude', sel_nadir=sel_nadir)

    return ds


##### BBR #####

def load_BNOM(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="BBR_NOM_1B", 
              frame_datetime="*", production_datetime="*", frame_code="*", 
              trim=True, i=-1, group='ScienceData/full', nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    ds = load_EC_product(srcpath, trim=False, group=group, i=i)
    
    if trim:
        ds = trim_to_frame(ds, latvar='barycentre_latitude')
    return ds


def load_BSNG(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="BBR_SNG_1B", 
              frame_datetime="*", production_datetime="*", frame_code="*", 
              trim=True, i=-1, group='ScienceData', idx_nadir=15, view='NADIR', nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                        frame_datetime, production_datetime, frame_code, 
                        nested_directory_structure)

    ds = load_EC_product(srcpath, trim=False, group=group, i=i, preprocess_for_NaT=True)
    
    if ds:
        ds = ds.sel(view=view)
        if view == "FORE":
            ds['radiance'][:] = ds.radiance
        sel_nadir={'across_track':idx_nadir}
        ds = ds.assign(selected_latitude=ds["latitude"].isel(sel_nadir).sel(band='TW'), 
                       selected_longitude=ds["longitude"].isel(sel_nadir).sel(band='TW')) 
        if trim:
            ds = trim_to_frame(ds, latvar='selected_latitude', sel_nadir=sel_nadir)
            
    return ds

############################
##### ESA L2a products #####
############################


##### A-FM #####

def load_AFM(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="ATL_FM__2A", 
             frame_datetime="*", production_datetime="*", frame_code="*",
             trim=True, group='ScienceData', i=-1, nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    return load_EC_product(srcpath, trim=trim, group=group, i=i)


##### A-PRO #####

def load_AEBD(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="ATL_EBD_2A", 
              frame_datetime="*", production_datetime="*", frame_code="*",
              trim=True, group='ScienceData', i=-1, nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    return load_EC_product(srcpath, trim=trim, group=group, i=i)


def load_AAER(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="ATL_AER_2A", 
              frame_datetime="*", production_datetime="*", frame_code="*",
              trim=True, group='ScienceData', i=-1, nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    return load_EC_product(srcpath, trim=trim, group=group, i=i)
    

def load_AICE(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="ATL_ICE_2A", 
              frame_datetime="*", production_datetime="*", frame_code="*",
              trim=True, group='ScienceData', i=-1, nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    return load_EC_product(srcpath, trim=trim, group=group, i=i)

def load_ATC(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="ATL_TC__2A", 
             frame_datetime="*", production_datetime="*", frame_code="*",
             trim=True, group='ScienceData', i=-1, nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    return load_EC_product(srcpath, trim=trim, group=group, i=i)


##### C-PRO #####

def load_CFMR(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="CPR_FMR_2A", 
              frame_datetime="*", production_datetime="*", frame_code="*",
              trim=True, i=-1, group='ScienceData', nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    return load_EC_product(srcpath, trim=trim, group=group, i=i)

def load_CCD(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="CPR_CD__2A", 
             frame_datetime="*", production_datetime="*", frame_code="*",
             trim=True, i=-1, group='ScienceData', nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    return load_EC_product(srcpath, trim=trim, group=group, i=i)

def load_CTC(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="CPR_TC__2A", 
             frame_datetime="*", production_datetime="*", frame_code="*",
             trim=True, i=-1, group='ScienceData', nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    return load_EC_product(srcpath, trim=trim, group=group, i=i)


##### C-CLD #####

def load_CCLD(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="CPR_CLD_2A", 
              frame_datetime="*", production_datetime="*", frame_code="*",
              trim=True, i=-1, group='ScienceData', nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    return load_EC_product(srcpath, trim=trim, group=group, i=i)

##### M-CLD #####

def load_MCM(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="MSI_CM__2A", 
              frame_datetime="*", production_datetime="*", frame_code="*", 
              trim=True, i=-1, group='ScienceData', idx_nadir=280, nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    ds = load_EC_product(srcpath, trim=False, group=group, i=i)
                         
    sel_nadir={'across_track':idx_nadir}
    
    if ds is not None:
        if trim:
            ds = trim_to_frame(ds, latvar='latitude', isel_nadir=sel_nadir)
        
        return ds.assign(selected_latitude=ds["latitude"].isel(sel_nadir), 
                     selected_longitude=ds["longitude"].isel(sel_nadir)) 
    

def load_MCOP(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="MSI_COP_2A", 
              frame_datetime="*", production_datetime="*", frame_code="*", 
              trim=True, i=-1, group='ScienceData', idx_nadir=280, nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    ds = load_EC_product(srcpath, trim=False, group=group, i=i)
                         
    sel_nadir={'across_track':idx_nadir}

    if ds is not None:
        if trim:
            ds = trim_to_frame(ds, latvar='latitude', isel_nadir=sel_nadir)
            
        return ds.assign(selected_latitude=ds["latitude"].isel(sel_nadir), 
                     selected_longitude=ds["longitude"].isel(sel_nadir)) 

##### M-AOT #####

def load_MAOT(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="MSI_AOT_2A", 
              frame_datetime="*", production_datetime="*", frame_code="*", 
              trim=True, i=-1, group='ScienceData', idx_nadir=280, nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    ds = load_EC_product(srcpath, trim=False, group=group, i=i)
                         
    sel_nadir={'across_track':idx_nadir}
        
    if ds is not None:
        if trim:
            ds = trim_to_frame(ds, latvar='latitude', sel_nadir=sel_nadir)
        return ds.assign(selected_latitude=ds["latitude"].isel(sel_nadir), 
                     selected_longitude=ds["longitude"].isel(sel_nadir)) 


############################
##### ESA L2b products #####
############################

def load_ACTC(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="AC__TC__2B", 
              frame_datetime="*", production_datetime="*", frame_code="*", 
              trim=True, i=-1, group='ScienceData', nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    return load_EC_product(srcpath, trim=trim, group=group, i=i)


def load_ACMCAP(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="ACM_CAP_2B", 
                frame_datetime="*", production_datetime="*", frame_code="*", 
                trim=True, i=-1, group='ScienceData', nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    return load_EC_product(srcpath, trim=trim, group=group, i=i)


def load_ACMCOM(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="ACM_COM_2B", 
                frame_datetime="*", production_datetime="*", frame_code="*", 
                trim=True, i=-1, group='ScienceData', nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    return load_EC_product(srcpath, trim=trim, group=group, i=i, renamer={'latitude_active':'latitude', 'longitude_active':'longitude'})

def load_ACMRT(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="ACM_RT__2B", 
                frame_datetime="*", production_datetime="*", frame_code="*", 
                trim=False, i=-1, group='ScienceData', nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    return load_EC_product(srcpath, trim=trim, group=group, i=i)

def load_ALLDF(srcpath, prodmod_code="ECA_EX[A-Z][A-Z]", product_code="ALL_DF__2B", 
                frame_datetime="*", production_datetime="*", frame_code="*", 
                trim=True, i=-1, group='ScienceData', nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    return load_EC_product(srcpath, trim=trim, group=group, i=i)
    

    

#############################
##### JAXA L2a products #####
#############################

def load_CCLP(srcpath, prodmod_code="ECA_JX[A-Z][A-Z]", product_code="CPR_CLP_2A",
              frame_datetime="*", production_datetime="*", frame_code="*",
              trim=True, i=-1, convert_to_dB=True,
              renamer_datagroup={'phony_dim_3':'along_track', 'phony_dim_4':'CPR_height'},
              renamer_geogroup ={'phony_dim_6':'along_track', 'phony_dim_7':'CPR_height'}, 
              nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    ds = load_EC_product(srcpath, trim=False, group='ScienceData/Data', renamer=renamer_datagroup, i=i)
    
    if ds:
        # merging in data from the Geo group (for geolocation)
        _geo = load_EC_product(srcpath, trim=False, group='ScienceData/Geo', renamer=renamer_geogroup, i=i, verbose=False)
        ds = xr.merge([ds, _geo])
        ds.encoding['source'] = _geo.encoding['source']
        _geo.close()
            
        # Couldn't trim until geo group merged with data group
        if trim:
            ds = trim_to_frame(ds, latvar='latitude')
    return ds


def load_ACLA(srcpath, prodmod_code="ECA_JX[A-Z][A-Z]", product_code="ATL_CLA_2A",
              frame_datetime="*", production_datetime="*", frame_code="*",
              trim=True, i=-1,
              renamer_datagroup={'phony_dim_3':'along_track', 'phony_dim_5':'JSG_height', 'phony_dim_6':'along_track_footprint'},
              renamer_geogroup ={'phony_dim_9':'along_track', 'phony_dim_10':'JSG_height', 'phony_dim_11':'along_track_footprint'}, 
              nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)
        
    ds = load_EC_product(srcpath, trim=False, group='ScienceData/Data', renamer=renamer_datagroup, i=i)
    
    if ds:
        # merging in data from the Geo group (for geolocation)
        _geo = load_EC_product(srcpath, trim=False, group='ScienceData/Geo', renamer=renamer_geogroup, i=i, verbose=False)
        ds = xr.merge([ds, _geo])
        ds.encoding['source'] = _geo.encoding['source']
        _geo.close()
            
        # Couldn't trim until geo group merged with data group
        if trim:
            ds = trim_to_frame(ds, latvar='latitude')
    return ds.squeeze()


def load_MCLP(srcpath, prodmod_code="ECA_JX[A-Z][A-Z]", product_code="MSI_CLP_2A",
              frame_datetime="*", production_datetime="*", frame_code="*",
              trim=True, i=-1, idx_nadir=284,
              renamer_datagroup={'phony_dim_3':'along_track', 'phony_dim_4':'across_track'},
              renamer_geogroup ={'phony_dim_6':'along_track', 'phony_dim_7':'across_track'}, nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)
        
    ds = load_EC_product(srcpath, trim=False, group='ScienceData/Data', renamer=renamer_datagroup, i=i)
    
    if ds:
        # merging in data from the Geo group (for geolocation)
        _geo = load_EC_product(srcpath, trim=False, group='ScienceData/Geo', renamer=renamer_geogroup, i=i, verbose=False)
        ds = xr.merge([ds, _geo])
        ds.encoding['source'] = _geo.encoding['source']
        _geo.close()
            
    sel_nadir={'across_track':idx_nadir}

    if ds is not None:
        if trim:
            ds = trim_to_frame(ds, latvar='latitude', isel_nadir=sel_nadir)
            
        return ds.assign(selected_latitude=ds["latitude"].isel(sel_nadir), 
                         selected_longitude=ds["longitude"].isel(sel_nadir)) 

    
#############################
##### JAXA L2b products #####
#############################

def load_ACCLP(srcpath, prodmod_code="ECA_JX[A-Z][A-Z]", product_code="AC__CLP_2B",
              frame_datetime="*", production_datetime="*", frame_code="*",
              trim=True, i=-1, convert_to_dB=True,
              renamer_datagroup={'phony_dim_3':'along_track', 'phony_dim_4':'CPR_height'},
              renamer_geogroup ={'phony_dim_7':'along_track', 'phony_dim_8':'CPR_height'}, 
               nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

    ds = load_EC_product(srcpath, trim=False, group='ScienceData/Data', renamer=renamer_datagroup, i=i)
    
    if ds:
        print(ds.dims)
        # merging in data from the Geo group (for geolocation)
        _geo = load_EC_product(srcpath, trim=False, group='ScienceData/Geo', renamer=renamer_geogroup, i=i, verbose=False)
        print(_geo.dims)
        ds = xr.merge([ds, _geo])
        ds.encoding['source'] = _geo.encoding['source']
        _geo.close()
            
        # Couldn't trim until geo group merged with data group
        if trim:
            ds = trim_to_frame(ds, latvar='latitude')
    
    return ds


def load_ACMCLP(srcpath, prodmod_code="ECA_JX[A-Z][A-Z]", product_code="ACM_CLP_2B",
                frame_datetime="*", production_datetime="*", frame_code="*",
                trim=True, i=-1, convert_to_dB=True,
                renamer_datagroup={'phony_dim_3':'along_track', 'phony_dim_4':'CPR_height'},
                renamer_geogroup ={'phony_dim_6':'along_track', 'phony_dim_7':'CPR_height'}, 
                nested_directory_structure=False):
    
    srcpath=get_srcpath(srcpath, prodmod_code, product_code, 
                frame_datetime, production_datetime, frame_code, 
                nested_directory_structure)

        
    ds = load_EC_product(srcpath, trim=False, group='ScienceData/Data', renamer=renamer_datagroup, i=i)
    
    if ds:
        print(ds.dims)
        # merging in data from the Geo group (for geolocation)
        _geo = load_EC_product(srcpath, trim=False, group='ScienceData/Geo', renamer=renamer_geogroup, i=i, verbose=False)
        print(_geo.dims)
        ds = xr.merge([ds, _geo])
        ds.encoding['source'] = _geo.encoding['source']
        _geo.close()
            
        # Couldn't trim until geo group merged with data group
        if trim:
            ds = trim_to_frame(ds, latvar='latitude')
    
    return ds









def load_ECL1(srcdir, trim=True):
    ANOM = load_ANOM(srcdir, trim=trim)
    BNOM = load_BNOM(srcdir, trim=trim)
    CNOM = load_CNOM(srcdir, trim=trim)
    MRGR = load_MRGR(srcdir, trim=trim)
    return ANOM, CNOM, MRGR, BNOM 


def load_ECL2(srcdir, trim=True):
    ATC = load_ATC(srcdir, trim=trim)
    CTC = load_CTC(srcdir, trim=trim)
    ACTC = load_ACTC(srcdir, trim=trim)
    
    AEBD = load_AEBD(srcdir, trim=trim)
    AAER = load_AAER(srcdir, trim=trim)
    AICE = load_AICE(srcdir, trim=trim)
    CCLD = load_CCLD(srcdir, trim=trim)
    ACMCOM = load_ACMCOM(srcdir, trim=trim)
    ACMCAP = load_ACMCAP(srcdir, trim=trim)
    
    return ATC, AEBD, AAER, AICE, CTC, CCLD, ACTC, ACMCAP, ACMCOM


def get_XMET(XMET, ds, 
             XMET_1D_variables = ['skin_temperature', 'surface_pressure'],
             #XMET_1D_variables = ['skin_temperature', 'temperature_at_2_metres, 
             #                     'sea_surface_temperature', 'soil_temperature_level1', 'surface_pressure', 
             #                     'eastward_wind_at_10_metres', 'northward_wind_at_10_metres', 
             #                     'snow_depth', 'snow_albedo_surface', 'sea_ice_cover',
             #                     'total_cloud_cover', 'total_column_ozone', 'total_column_water_vapour',
             #                     'leaf_area_index_low_vegetation', 'leaf_area_index_high_vegetation', 'forecast_surface_roughness',
             #                     'boundary_layer_height',  'tropopause_height_wmo', 'tropopause_height_calipso', 
             #                     'near_ir_albedo_for_diffuse_radiation_surface', 'near_ir_albedo_for_direct_radiation_surface',
             #                     'uv_visible_albedo_for_diffuse_radiation_surface', 'uv_visible_albedo_for_direct_radiation_surface'],
             XMET_2D_variables = ['temperature', 'pressure', 'specific_humidity'],
             #XMET_2D_variables=['temperature', 'wet_bulb_temperature, 'pressure', 'specific_humidity', 'relative_humidity', 
             #                    'northward_wind', 'eastward_wind', 'upward_air_velocity', 
             #                    'ozone_mass_mixing_ratio', 'divergence', 'cloud_cover', 
             #                    'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content', 
             #                    'specific_rain_water_content', 'specific_snow_water_content'],
             grid_latvar='latitude', grid_lonvar='longitude', grid_altvar='height', grid_time='time', 
             grid_alongtrackdim='along_track', grid_heightdim='JSG_height',
             grid_surface_elevation='elevation', merge_inputs=True):
    """
    Adds supplementary meteorological information from X-MET, interpolated onto the grid of the dataset provided.

    Note that the default lists of 1D and 2D variables are a subset of those available from the X-MET product.
    Full lists of available variables are commented out above.
    """
    
    from scipy.interpolate import NearestNDInterpolator
    frame = ds.encoding['source'].split('.')[0].split("_")[-1]
    if frame.endswith('C') or frame.endswith('G'):
        #Polar frames have some artefacts of the nearest-neighbour interpolation 
        #over X-MET grid: should find another (still fast) approach 
        #which may include a linear interpolation 
        from scipy.interpolate import NearestNDInterpolator
        NearestIndex = NearestNDInterpolator(np.array([XMET['longitude'].to_numpy().flatten(), 
                                                       XMET['latitude'].to_numpy().flatten()]).T, 
                                         XMET.horizontal_grid.to_numpy().flatten())
    
    else:
        from scipy.interpolate import NearestNDInterpolator
        NearestIndex = NearestNDInterpolator(np.array([XMET['longitude'].to_numpy().flatten(), 
                                                       XMET['latitude'].to_numpy().flatten()]).T, 
                                             XMET.horizontal_grid.to_numpy().flatten())
        
    track_indices = NearestIndex(np.array([ds[grid_lonvar], ds[grid_latvar]]).T).astype('int')
    unique_track_indices, unique_alongtrack_indices = np.unique(track_indices, return_index=True)

    # Preserving indices 
    XMET['horizontal_grid'] = XMET.horizontal_grid
    
    xmet_2D_variables = ['geometrical_height'] + XMET_2D_variables

    print("Interpolating 2D fields...")
    XMET_profiles_unique = []
    for uihz, uiat in zip(unique_track_indices, unique_alongtrack_indices):
        _XMET = XMET[xmet_2D_variables].isel(horizontal_grid=uihz).swap_dims({'height':'geometrical_height'}).interp(geometrical_height=ds.isel(along_track=uiat)[grid_altvar]).drop(['horizontal_grid', 'geometrical_height'])
        XMET_profiles_unique.append(_XMET)
    XMET_interp2D = xr.concat(XMET_profiles_unique, dim=grid_alongtrackdim).sortby(grid_alongtrackdim).reindex_like(ds, method='bfill')#.ffill('along_track')

    print("\t\t\t... done")
    
    #Subset using repeated values 
    XMET_alongtrack = XMET.isel(horizontal_grid=track_indices).rename({'horizontal_grid':grid_alongtrackdim})
    XMET_alongtrack[grid_alongtrackdim] = ds[grid_alongtrackdim]
    
    XMET_on_grid = XMET_alongtrack[XMET_1D_variables]  
    for v in [grid_time, grid_latvar, grid_lonvar]:
        XMET_on_grid[v] = ds[v]
    
    if merge_inputs:
        _encoding = ds.encoding
        ds_merged = xr.merge([ds, XMET_interp2D, XMET_on_grid])
        ds_merged.encoding = _encoding
        return ds_merged
    else:
        return XMET_on_grid
