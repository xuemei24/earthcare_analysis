import numpy as np
import xarray as xr
import sys
import pandas as pd
from pylab import *
import os
import sys

#################
### FUNCTIONS ###
#################
def get_AngstromExponent(aodW1, aodW2, wave1, wave2):
    return -np.log(aodW1 / aodW2) / np.log(wave1 / wave2)

def get_AerosolOpticalDepth(aod_old, wave_old, wave_new, angstrom):
    print('aod_old>0',aod_old[aod_old>0], 'wave_old',wave_old, 'wave_new',wave_new, 'angstrom>0',angstrom[angstrom>0])
    return ((wave_new / wave_old) ** (-angstrom)) * aod_old

#################
### CONSTANTS ###
#################
Rd = 287.058 # Dry Air gas constant(J kg^-1 K^-1)
Rw = 461.52  # Water Vapor gas constant(J kg^-1 K^-1)
g  = 9.80665 # Gravitational acceleration constant (m s^-2)

#################
### READ DATA ###
#################
### Read Mass Extinction Coefficient (kg m^-2)
### units: m2 kg-1, long_name: Shortwave mass-extinction coefficient of hydrophilic aerosols
### units: m2 kg-1, long_name: Shortwave mass-extinction coefficient of hydrophobic aerosols
### wavenumber1_sw[band_sw], units: cm-1 converted to wavelength nm
### wavenumber2_sw[band_sw], units: cm-1 converted to wavelength nm 
fmext = xr.open_dataset('aerosol_ifs_rrtm_49R1_20230119.nc')
mext_sw_hl     = fmext["mass_ext_sw_hydrophilic"]  # mass_ext_sw_hydrophilic[ band_sw (14), relative_humidity (12) , hydrophilic (10) ]
mext_sw_hb     = fmext["mass_ext_sw_hydrophobic"]  # mass_ext_sw_hydrophobic[ band_sw (14), hydrophobic (14) ]
wavelength1_sw = 1e7 * 1 / fmext["wavenumber1_sw"] # Lower bound wavenumber for shortwave band
wavelength2_sw = 1e7 * 1 / fmext["wavenumber2_sw"] # Upper bound wavenumber for shortwave band
wavelength_sw  = (wavelength1_sw + wavelength2_sw) / 2
print('wavelength_sw',wavelength_sw)
fmext.close()

### Variables
MR_names = ["aermr01", # long_name: Sea Salt Aerosol (0.03 - 0.5 um) Mixing Ratio
            "aermr02", # long_name: Sea Salt Aerosol (0.5 - 5 um) Mixing Ratio
            "aermr03", # long_name: Sea Salt Aerosol (5 - 20 um) Mixing Ratio
            "aermr04", # long_name: Dust Aerosol (0.03 - 0.55 um) Mixing Ratio
            "aermr05", # long_name: Dust Aerosol (0.55 - 0.9 um) Mixing Ratio
            "aermr06", # long_name: Dust Aerosol (0.9 - 20 um) Mixing Ratio
            "aermr07", # long_name: Hydrophilic Organic Matter Aerosol Mixing Ratio
            "aermr08", # long_name: Hydrophobic Organic Matter Aerosol Mixing Ratio
            "aermr09", # long_name: Hydrophilic Black Carbon Aerosol Mixing Ratio
            "aermr10", # long_name: Hydrophobic Black Carbon Aerosol Mixing Ratio
            "aermr11", # long_name: Sulphate Aerosol Mixing Ratio
            "aermr16", # long_name: Nitrate fine mode aerosol mass mixing ratio
            "aermr17", # long_name: Nitrate coarse mode aerosol mass mixing ratio
            "aermr18", # long_name: Ammonium aerosol mass mixing ratio
            "aermr19", # long_name: Biogenic secondary organic aerosol mass mixing ratio
            "aermr20"] # long_name: Anthropogenic secondary organic aerosol mass mixing ratio

cams_data = ['sea_salt_0.03-0.5', 'sea_salt_0.5-5', 'sea_salt_5-20','dust_0.03-0.55', 'dust_0.55-0.9', 'dust_0.9-20','hydrophilic_organic_matter','hydrophobic_organic_matter','hydrophilic_black_carbon','hydrophobic_black_carbon','sulfate','nitrate_fine_mode','nitrate_coarse_mode','ammonium','biogenic_secondary_organic','anthropogenic_secondary_organic']



month = 'november'
year = '2024' if month == 'december' else '2025'
### Forecast periods
suffix1 = '_mmr_'+month+'_'+year+'_'
suffix2 = '.nc'
forecast_periods = ['0','3']#,'6','9']

cams_dir = '/scratch/nld6854/earthcare/cams_data/'+month+'_'+year+'/'

#load AOD files
faod355 = xr.open_dataset(cams_dir+'total_aerosol_optical_depth_355nm_'+month+'_'+year+'.nc')
all_aod355  = faod355['aod355']
clat = faod355['latitude'].data
clon = faod355['longitude'].data
all_time = faod355['forecast_reference_time'].data
all_forecast_period_aod = faod355['forecast_period']

length = faod355['aod355'].shape[1]
faod355.close()

faod550 = xr.open_dataset(cams_dir+'multiple_aod_550nm_'+month+'_'+year+'.nc') #12,60,451,900
all_aod550  = faod550['aod550']
faod550.close()

#load a few other files
fl137 = pd.read_csv('L137.csv')
a_param = fl137["a [Pa]"]
b_param = fl137["b"]

fsurfp = xr.open_dataset(cams_dir+'surface_pressure_'+month+'_'+year+'.nc') #12,60,451,900
all_Pressure_s = fsurfp['sp']
fsurfp.close()

fgeop = xr.open_dataset(cams_dir+'surface_geopotential_'+month+'_'+year+'.nc') #12,60,451,900
all_Geopotential_s = fgeop['z']
fgeop.close()

for lforecast_period in forecast_periods:
    print('lforecast_period=',lforecast_period)
    aod550p = np.zeros((1,length,451,900)) #xxx
    aod355p = np.zeros((1,length,451,900)) #xxx
    AE_550to355p = np.zeros((1,length,451,900)) #xxx
    AOD_Dry_355nmp = np.zeros((1,length,451,900,8)) #xxx
    AOD_Wat_355nmp = np.zeros((1,length,451,900)) #xxx
    timep = np.zeros((length))

    for iff in range(length): #forecast reference time
        print('lforecast_period=',lforecast_period,'iff=',iff)
        ### Read Model AOD
        aod355 = all_aod355[int(lforecast_period),iff]
        timep[iff] = all_time[iff]
        forecast_period_aod = all_forecast_period_aod[int(lforecast_period)]
        faod355.close()
 
        aod550  = all_aod550[int(lforecast_period),iff] #xxx
 
        AE_550to355 = get_AngstromExponent(aod550, aod355, 550, 355)
 
 
        ### Derive air density
        #################
        ### CALCULATE ### Pressure_i, Pressure_f, AirDensity_f, TemperatureVirtual_f, Geopotential_i, GridcellHeight
        #################
        ### Calculating Pressure (Pa) by level (narkwtika... anti na ta dinoun...)
        ### More info: Chapter 2.2.1 https://www.ecmwf.int/sites/default/files/elibrary/2015/9210-part-iii-dynamics-and-numerical-procedures.pdf
        ###            https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions
        ###            (1) Pressure_i = A + B * Psurface                        
        ###            (2) Pressure_f = (Pressure_i+1 + Pressure_i-1) / 2
        pressure_s_2d = all_Pressure_s[int(lforecast_period),iff]#xxx
 
        # Initialize Pressure_i
        n_levels = len(a_param)
        lat, lon = pressure_s_2d.shape
        Pressure_i = np.empty((n_levels, lat, lon))
 
        # Fill Pressure_i
        for l in range(n_levels):
            Pressure_i[l,:,:] = a_param[l] + b_param[l] * pressure_s_2d
 
        n_levels, lat, lon = Pressure_i.shape
        Pressure_f = np.empty((n_levels - 1, lat, lon))
 
        for l in range(1, n_levels):
            Pressure_f[l - 1,:, :] = (Pressure_i[l-1, :, :] + Pressure_i[l, :, :]) / 2
 
        ftemp = xr.open_dataset(cams_dir+'temperature_'+month+'_'+year+'_'+lforecast_period+'.nc') #xxx #1,60,137,451,900
        Temperature_f = ftemp['t'][0,iff]#xxx
        ftemp.close()
 
        fqt = xr.open_dataset(cams_dir+'specific_humidity_'+month+'_'+year+'_'+lforecast_period+'.nc') #xxx #1,60,137,451,900
        SpecificHumidity_f = fqt['q'][0,iff] #xxx
        fqt.close()
 
        Geopotential_s = all_Geopotential_s[int(lforecast_period),iff] #xxx
 
        TemperatureVirtual_f = Temperature_f * ( 1 + ( (Rw / Rd) - 1) * SpecificHumidity_f )
 
        ### Defining grids
        nx   = lon
        ny   = lat
        nz_f = 137#len(lev)   # At full levels (137 in total)
        nz_i = n_levels+1 # At interfaces  (138 in total)
  
        # Initialize Geopotential_i
        Geopotential_i = np.zeros((nz_i,ny,nx))
        # Set bottom layer equal to Geopotential_s
        Geopotential_i[nz_i - 1, :, :] = Geopotential_s  # Python is 0-indexed
        # Fill remaining levels from bottom to top (nz_i-1 to 1)
        for l in range(nz_i - 2, -1, -1):  # From nz_i-1 to 1
            Geopotential_i[l - 2,:, :] = Geopotential_i[l-1, :, :] + \
                Rd * TemperatureVirtual_f[l-2, :, :] * np.log(Pressure_i[l-1, :, :] / Pressure_i[l-2, :, :])
 
        GridcellHeight = np.zeros((nz_f, ny, nx))
 
        # Calculate GridcellHeight
        nz = GridcellHeight.shape[0]  # Assuming GridcellHeight is already initialized
        for l in range(nz):
            GridcellHeight[l, :, :] = (Geopotential_i[l, :, :] - Geopotential_i[l+1, :, :]) / g
        # Set top full level to zero
        GridcellHeight[0, :, :] = 0
 
        ### Calculating Air Density (kg m^-3)
        ### More info: https://confluence.ecmwf.int/pages/viewpage.action?pageId=153391710
        ### Air_Density = Pressure / (Specific_Gas_Constant * Temperature)
        AirDensity_f = Pressure_f / (Rd * Temperature_f)
 
 
        #################
        ### CALCULATE ### AOD_Dry using MassExtinctionCoefficientDry at RH between 0% to 10% for soluble particles
        #################
        AOD_Dry_303nm = np.zeros((ny, nx, 8)) # Initialize variable for 7 species + 1 total
        AOD_Dry_393nm = np.zeros((ny, nx, 8)) # Initialize variable for 7 species + 1 total
        AOD_Dry = np.zeros((ny, nx))                        # Initialize variable
        AOD_SpeciesDry = np.zeros((ny, nx, len(MR_names)))  # Initialize variable for 13 tracers
 
        for w in [10,11]: # 10 = 393 nm, 11 = 304 nm
            for i in range(len(MR_names)):  # Loop through tracer names
                file_path = cams_dir+'cams_original_aerosol_mmr/'+cams_data[i]+suffix1+lforecast_period+suffix2
                fMixingRatio_f = xr.open_dataset(file_path)
                MixingRatio_f = fMixingRatio_f[MR_names[i]][0,iff]#xxx
                fMixingRatio_f.close()
      
                ### Manually pick the correct MEC for each species. For soluble get MEC for RH between 0% to 10%
                hl_map = {
                    "aermr01": (mext_sw_hl, (0, 0, w)),
                    "aermr02": (mext_sw_hl, (1, 0, w)),
                    "aermr03": (mext_sw_hl, (2, 0, w)),
                    "aermr07": (mext_sw_hl, (3, 0, w)),
                    "aermr11": (mext_sw_hl, (5, 0, w)),
                    "aermr16": (mext_sw_hl, (11,0, w)),
                    "aermr17": (mext_sw_hl, (12,0, w)),
                    "aermr18": (mext_sw_hl, (10,0, w)),
                    "aermr19": (mext_sw_hl, (8, 0, w)),
                    "aermr20": (mext_sw_hl, (9, 0, w)),}
               
                hb_map = {
                    "aermr04": (mext_sw_hb, (6, w)),
                    "aermr05": (mext_sw_hb, (7, w)),
                    "aermr06": (mext_sw_hb, (8, w)),
                    "aermr08": (mext_sw_hb, (12,w)),
                    "aermr09": (mext_sw_hb, (14,w)),
                    "aermr10": (mext_sw_hb, (14,w)),}
               
                name = MR_names[i]
               
                if name in hl_map:
                    arr, idx = hl_map[name]
                    MassExtinctionCoefficientDry = arr[idx]
                elif name in hb_map:
                    arr, idx = hb_map[name]
                    MassExtinctionCoefficientDry = arr[idx]
                else:
                    raise ValueError(f"Unknown tracer name: {name}")
      
                # Mass Concentration (kg m^-3) = Mixing Ratio (kg kg^-1) * Air Density (kg m^-3)
                MassConcentration_f = MixingRatio_f * AirDensity_f
                # Extinction Coefficients (m^-1) = Mass Concentration (kg m^-3) * Mass Extinction Coefficient (kg m^-2)
                ExtinctionCoefficientsDry_f  = MassConcentration_f * MassExtinctionCoefficientDry
  
                # Extinction (Unitless) = Extinction Coefficients (m^-1) * Height of each model grid cell (m)
                #ExtinctionDry_f = ExtinctionCoefficientsDry_f * GridcellHeight
                #ExtinctionDry_f = np.trapz(ExtinctionCoefficientsDry_f.values,x=GridcellHeight,axis=0)
                ExtinctionDry_f = ExtinctionCoefficientsDry_f * GridcellHeight
                # Sum Extinction (Unitless) over all model levels to get AOD_Dry.
                # Place each AOD species in the 3rd dimension
                AOD_SpeciesDry[:, :, i] = np.nansum(ExtinctionDry_f, axis=0)
            if w == 10:   # 393 nm
                AOD_Dry_393nm[:, :, 0] = np.nansum(AOD_SpeciesDry, axis=2)              # aod
                AOD_Dry_393nm[:, :, 1] = np.nansum(AOD_SpeciesDry[:, :, 0:3], axis=2)   # ssa
                AOD_Dry_393nm[:, :, 2] = np.nansum(AOD_SpeciesDry[:, :, 3:6], axis=2)   # du
                AOD_Dry_393nm[:, :, 3] = np.nansum(AOD_SpeciesDry[:, :, 6:8], axis=2)+np.nansum(AOD_SpeciesDry[:, :, -2:], axis=2)   # om
                AOD_Dry_393nm[:, :, 4] = np.nansum(AOD_SpeciesDry[:, :, 8:10], axis=2)  # bc
                AOD_Dry_393nm[:, :, 5] = AOD_SpeciesDry[:, :, 10]                       # su
                AOD_Dry_393nm[:, :, 6] = np.nansum(AOD_SpeciesDry[:, :, 11:13], axis=2) # ni
                AOD_Dry_393nm[:, :, 7] = AOD_SpeciesDry[:, :, 13]                       # am
  
            elif w == 11: # 303 nm
                AOD_Dry_303nm[:, :, 0] = np.nansum(AOD_SpeciesDry, axis=2)              # aod
                AOD_Dry_303nm[:, :, 1] = np.nansum(AOD_SpeciesDry[:, :, 0:3], axis=2)   # ssa
                AOD_Dry_303nm[:, :, 2] = np.nansum(AOD_SpeciesDry[:, :, 3:6], axis=2)   # du
                AOD_Dry_303nm[:, :, 3] = np.nansum(AOD_SpeciesDry[:, :, 6:8], axis=2)+np.nansum(AOD_SpeciesDry[:, :, -2:], axis=2)   # om
                AOD_Dry_303nm[:, :, 4] = np.nansum(AOD_SpeciesDry[:, :, 8:10], axis=2)  # bc
                AOD_Dry_303nm[:, :, 5] = AOD_SpeciesDry[:, :, 10]                       # su
                AOD_Dry_303nm[:, :, 6] = np.nansum(AOD_SpeciesDry[:, :, 11:13], axis=2) # ni
                AOD_Dry_303nm[:, :, 7] = AOD_SpeciesDry[:, :, 13]                       # am
 
        ####################################################
        ### CALCULATE Dry AOD and AE for 550nm and 865nm ###
        ####################################################
        AE_Dry_303to393 = get_AngstromExponent(AOD_Dry_303nm,AOD_Dry_393nm,wavelength_sw[11].values,wavelength_sw[10].values)
        AOD_Dry_355nm   = get_AerosolOpticalDepth(AOD_Dry_303nm,wavelength_sw[11].values,355,AE_Dry_303to393)
 
        AOD_Wat_355nm = aod355 - AOD_Dry_355nm[:, :, 0]

        aod550p[0,iff] = aod550.values #xxx
        aod355p[0,iff] = aod355.values #xxx
        AE_550to355p[0,iff] = AE_550to355.values #xxx
        AOD_Dry_355nmp[0,iff] = AOD_Dry_355nm #xxx
        AOD_Wat_355nmp[0,iff] = AOD_Wat_355nm #xxx

    ####################
    ### WRITE NETCDF ###
    ####################
    print('Now start writing AOD to netCDF4')
    ds = xr.Dataset(
        {
            "aod550": (("forecast_period", "forecast_reference_time","latitude","longitude"),aod550p),
            "aod355": (("forecast_period", "forecast_reference_time","latitude","longitude"), aod355p),
            "ang550to355": (("forecast_period", "forecast_reference_time","latitude","longitude"), AE_550to355p),
            "dryaod355": (("forecast_period", "forecast_reference_time","latitude","longitude"), AOD_Dry_355nmp[:,:, :, :, 0]),
            "wataod355": (("forecast_period", "forecast_reference_time","latitude","longitude"), AOD_Wat_355nmp),
            "dudryaod355": (("forecast_period", "forecast_reference_time", "latitude", "longitude"), AOD_Dry_355nmp[:, :, :, :, 2]),
            "ssdryaod355": (("forecast_period", "forecast_reference_time", "latitude", "longitude"), AOD_Dry_355nmp[:, :, :, :, 1]),
            "omdryaod355": (("forecast_period", "forecast_reference_time", "latitude", "longitude"), AOD_Dry_355nmp[:, :, :, :, 3]),
            "bcdryaod355": (("forecast_period", "forecast_reference_time", "latitude", "longitude"), AOD_Dry_355nmp[:, :, :, :, 4]),
            "sudryaod355": (("forecast_period", "forecast_reference_time", "latitude", "longitude"), AOD_Dry_355nmp[:, :, :, :, 5]),
            "nidryaod355": (("forecast_period", "forecast_reference_time", "latitude", "longitude"), AOD_Dry_355nmp[:, :, :, :, 6]),
            "amdryaod355": (("forecast_period", "forecast_reference_time", "latitude", "longitude"), AOD_Dry_355nmp[:, :, :, :, 7])

        },
        coords={
            "forecast_period": [forecast_period_aod.values],
            "forecast_reference_time": timep,
            "latitude": clat,
            "longitude": clon
        }

    )

    filepath = os.path.join(cams_dir+'/TTcal/',
    f"TTcal_aod355nm_per_composition_{month}_"+year+"_{lforecast_period}.nc")
    ds.to_netcdf(filepath, format="NETCDF4")
