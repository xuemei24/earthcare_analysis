import numpy as np
import glob
from ectools import ecio

select_height=-241

def get_wind(file_path):
    file_dir = file_path[:-63]
    file_dir = file_dir.replace('EBD','XMET')
    #file_dir = file_dir.replace('nobackup','nobackup_1')
    #file_dir = file_dir.replace('_2A','')
    orbit_sequence=file_path[-9:-3]
    xmet_file = sorted(glob.glob(file_dir+'*'+file_path[-9:]))[0]

    AEBD = ecio.load_AEBD(file_path)
    XMET = ecio.load_XMET(xmet_file)
    AEBD = ecio.get_XMET(XMET,AEBD,XMET_2D_variables=['eastward_wind', 'northward_wind', 'upward_air_velocity'],grid_heightdim='ATLID_height')

    u_wind = AEBD['eastward_wind'][:,select_height:].values
    v_wind = AEBD['northward_wind'][:,select_height:].values
    w_wind = AEBD['upward_air_velocity'][:,select_height:].values
    lat = AEBD['latitude'].values
    lon = AEBD['longitude'].values

    # Calculate heading angle (in radians)
    dlat = np.gradient(lat)
    dlon = np.gradient(lon)

    # Convert to meters (approx.)
    R_earth = 6378e3  # Earth radius in meters
    dx = np.radians(dlon) * R_earth * np.cos(np.radians(lat))
    dy = np.radians(dlat) * R_earth

    # Heading angle from east (zonal direction)
    theta = np.arctan2(dy, dx)  # shape: (n_points,)

    # u, v: shape (height, track_len)
    # theta: shape (track_len,) — broadcast along height
    theta_2d = theta[:,np.newaxis]
    #theta_2d = np.broadcast_to(theta, u_wind.shape)

    u_along = u_wind * np.cos(theta_2d) + v_wind * np.sin(theta_2d)

    return u_along,w_wind

#remove any column with liquid water
def get_ext_col(file_path,include_xmet=0):
    try:
        tc_file = file_path.replace('EBD','TC_')
        with ecio.load_ATC(tc_file) as ATC:
            qc = ATC['quality_status'][:,select_height:].values
            tc = ATC['classification_low_resolution'][:,select_height:].values
            tct = tc.copy()
            for ni in range(tc.shape[0]):
                for nj in range(1,tc.shape[1]):
                    if tc[ni,nj-1]==-2 and tc[ni,nj]==-1:
                        tct[ni,nj]=-2
            tc = tct
            tc_cld = tc.copy()
            cld_index = np.array([-2, 1, 2, 3, 20, 21, 22])
            tc_cld = np.where(np.isin(tc_cld,cld_index),tc_cld,np.nan)
 
            tc2 = np.where(tc>=-1,tc,np.nan)
            tc2 = np.where(tc2<=2,tc2,np.nan)
            tc2 = np.where(tc2!=0,tc2,np.nan)
            tc = np.where(qc<2,tc,np.nan)
 
            aer_index = np.array([10,11,12,13,14,15,25,26,27])
            tc = np.where(np.isin(tc,aer_index),tc,np.nan)
 
        with ecio.load_AEBD(file_path) as AEBD:
            data = AEBD['particle_extinction_coefficient_355nm_low_resolution'][:,select_height:].values
            err  = AEBD['particle_extinction_coefficient_355nm_low_resolution_error'][:,select_height:].values
            lat = AEBD['latitude'].values
            lon = AEBD['longitude'].values
            time = AEBD['time'].values
            def filter_data_col(dt):
                dt = np.where(qc<2,dt,0)
                dt = np.where(tc>0,dt,0)  #only keep aerosol particles
 
                #mask the columns when there are liquid clouds or fully attenuated
                mask = (tc2>=-1).any(axis=1)
                dt[mask,:] = np.nan
                return dt

            data = filter_data_col(data)

            #filter data with deploarisation <=0.2
            depol_err = AEBD['particle_linear_depol_ratio_355nm_low_resolution_error'].values
            data = np.where(depol_err<=0.2,data,0)

            data = np.where(data<1e-3,data,0)
            data = np.where(data>=0,data,0)

            err  = filter_data_col(err) 

            #Trim the rows when height is negative
            geoid = AEBD['geoid_offset'].values
            geoid = geoid[:,np.newaxis]
            height = AEBD['height'][:,select_height:].values-geoid
 
        return data[:,::-1], lat, lon,time, height[:,::-1],tc_cld[:,::-1],tct[:,::-1],err[:,::-1]#,ATC

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

#keep all ext data (including columns with liquid clouds and/or fully attenuated)
def get_ext(file_path,aer_index,include_xmet=0):
    try:
        tc_file = file_path.replace('EBD','TC_')
        with ecio.load_ATC(tc_file) as ATC:
            qc = ATC['quality_status'][:,select_height:].values
            tc = ATC['classification_low_resolution'][:,select_height:].values
            tct = tc.copy()
            for ni in range(tc.shape[0]):
                for nj in range(1,tc.shape[1]):
                    if tc[ni,nj-1]==-2 and tc[ni,nj]==-1:
                        tct[ni,nj]=-2
            tc = tct
            tc_cld = tc.copy()
            cld_index = np.array([-2, 1, 2, 3, 20, 21, 22])
            tc_cld = np.where(np.isin(tc_cld,cld_index),tc_cld,np.nan)
 
            tc = np.where(qc<2,tc,np.nan)
            tc = np.where(np.isin(tc,aer_index),tc,np.nan)
 
        with ecio.load_AEBD(file_path) as AEBD:
            data = AEBD['particle_extinction_coefficient_355nm_low_resolution'][:,select_height:].values
            err  = AEBD['particle_extinction_coefficient_355nm_low_resolution_error'][:,select_height:].values
            lat = AEBD['latitude'].values
            lon = AEBD['longitude'].values
            time = AEBD['time'].values

            ### filter with snr and replace with nan
            snr_threshold = 2                                       
            snr = np.where(err != 0, data / err, snr_threshold+1)   
            data = np.where(snr>=snr_threshold,data,0)

            #filter data with deploarisation <=0.2
            depol_err = AEBD['particle_linear_depol_ratio_355nm_low_resolution_error'][:,select_height:].values
            data = np.where(depol_err<=0.2,data,0)

            def filter_data(dt):
                dt = np.where(qc<2,dt,0)                                 
                dt = np.where(tc>0,dt,0)  #only keep aerosol particles   

                return dt

            data = filter_data(data)
            data = np.where(data<1e-3,data,0)                  
            data = np.where(data>=0,data,0)                    

            err  = filter_data(err)

            #Trim the rows when height is negative
            geoid = AEBD['geoid_offset'].values
            geoid = geoid[:,np.newaxis]
            height = AEBD['height'][:,select_height:].values-geoid

            reversed_data = data[:,::-1]
            reversed_height = height[:,::-1]
            tc_cld = tc_cld[:,::-1]
        return reversed_data, lat, lon,time, reversed_height,tc_cld,tct[:,::-1],err[:,::-1]#,ATC

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

#This function is now replaced by get_aod_snr remove any column if liquid water exists
def get_aod(file_path,aer_index):
    try:
        tc_file = file_path.replace('EBD','TC_')
        #tc_file = tc_file.replace('nobackup','nobackup_1')
        #with ecio.load_ATC(tc_file,prodmod_code="ECA_EXAE") as ATC:
        with ecio.load_ATC(tc_file) as ATC:
            qc = ATC['quality_status'].values
            tc = ATC['classification_low_resolution'].values
       
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
       
            #total aerosol
            aer_index = aer_index#np.array([10,11,12,13,14,15,25,26,27])
            tc = np.where(np.isin(tc,aer_index),tc,np.nan)

        #with ecio.load_AEBD(file_path,prodmod_code="ECA_EXAE") as AEBD:
        with ecio.load_AEBD(file_path) as AEBD:
            data = AEBD['particle_extinction_coefficient_355nm_low_resolution'].values
            lat = AEBD['latitude'].values
            lon = AEBD['longitude'].values
            qc = AEBD['quality_status'].values
            time = AEBD['time'].values

            data = np.where(qc<2,data,0)
            data = np.where(data<1e-3,data,0)
            data = np.where(data>=0,data,0)
            data = np.where(tc>0,data,0)  #only keep aerosol particles

            #mask the columns when there are liquid clouds or fully attenuated
            mask = (tc2>=-1).any(axis=1)
            data[mask,:] = np.nan

            #Trim the rows when height is negative
            height = AEBD['height'].values

            reversed_data = data[:,::-1]
            reversed_height = height[:,::-1]
            aod = np.trapz(reversed_data,x=reversed_height,axis=1)
        return aod, lat, lon,time
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

#keep all ext data (including columns with liquid clouds and/or fully attenuated)
#filter with SNR>=2 and depolarisation err <= 0.2
def get_aod_var(file_path,aer_index,include_xmet=0):
    try:
        tc_file = file_path.replace('EBD','TC_')
        with ecio.load_ATC(tc_file) as ATC:
            qc = ATC['quality_status'].values
            tc = ATC['classification_low_resolution'].values
            tct = tc.copy()
            for ni in range(tc.shape[0]):
                for nj in range(1,tc.shape[1]):
                    if tc[ni,nj-1]==-2 and tc[ni,nj]==-1:
                        tct[ni,nj]=-2
            tc = tct
            tc_cld = tc.copy()
            cld_index = np.array([-2, 1, 2, 3, 20, 21, 22])
            tc_cld = np.where(np.isin(tc_cld,cld_index),tc_cld,np.nan)

            tc = np.where(qc<2,tc,np.nan)

            tc = np.where(np.isin(tc,aer_index),tc,np.nan)

        with ecio.load_AEBD(file_path) as AEBD:
            data = AEBD['particle_extinction_coefficient_355nm_low_resolution'].values
            err  = AEBD['particle_extinction_coefficient_355nm_low_resolution_error'].values
            lat = AEBD['latitude'].values
            lon = AEBD['longitude'].values
            time = AEBD['time'].values
            def filter_data(dt):
                dt = np.where(qc<2,dt,np.nan) #was 0
                dt = np.where(tc>0,dt,np.nan) #was 0 #only keep aerosol particles
                return dt

            err  = filter_data(err)

            ########filtering err with snr >= 2
            snr_threshold = 2
            snr = np.where(err != 0, data / err, snr_threshold+1)
            err = np.where(snr>=snr_threshold,err,np.nan) #was 0

            #filter data with deploarisation <=0.2
            depol_err = AEBD['particle_linear_depol_ratio_355nm_low_resolution_error'].values
            err = np.where(depol_err<=0.2,err,np.nan)

            #Trim the rows when height is negative
            #height = file['ScienceData/height'][:,select_height:]-file['ScienceData/geoid_offset'][:][:,np.newaxis]
            geoid = AEBD['geoid_offset'].values
            geoid = geoid[:,np.newaxis]
            height = AEBD['height'].values-geoid

            #mask = (height<0).any(axis=0)
            #data = data[:,~mask]
            #height = height[:,~mask]

            reversed_err = err[:,::-1]
            reversed_height = height[:,::-1]

            dz = np.diff(reversed_height, axis=1)
            sigma2_avg = 0.5 * (reversed_err[:, :-1]**2 + reversed_err[:, 1:]**2)
            # Mask out any segments with NaN in either dz or variance
            mask = np.isfinite(sigma2_avg) & np.isfinite(dz)

            # Compute σ² × (dz)² where valid
            segment_contrib = np.where(mask, sigma2_avg * (dz ** 2), 0.0)

            # Count how many valid segments each profile has
            valid_counts = np.sum(mask, axis=1)

            # Integrate (sum) over valid segments
            aod_var = np.sum(segment_contrib, axis=1)

            # Optional: flag profiles with too few valid levels
            aod_var[valid_counts < 2] = np.nan

        return aod_var, lat, lon,time

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

#filter aod with snr>=2 and depolarisation err <= 0.2
def get_aod_snr(file_path,aer_index):
    try:
        tc_file = file_path.replace('EBD','TC_')
        with ecio.load_ATC(tc_file) as ATC:
            qc = ATC['quality_status'].values
            tc = ATC['classification_low_resolution'].values

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

            #total aerosol
            aer_index = aer_index#np.array([10,11,12,13,14,15,25,26,27])
            tc = np.where(np.isin(tc,aer_index),tc,np.nan)

        with ecio.load_AEBD(file_path) as AEBD:
            data = AEBD['particle_extinction_coefficient_355nm_low_resolution'].values

            #filter data with SNR>=2
            err  = AEBD['particle_extinction_coefficient_355nm_low_resolution_error'].values           
            snr_threshold = 2
            snr = np.where(err != 0, data / err, snr_threshold+1)
            data = np.where(snr>=snr_threshold,data,0)

            #filter data with deploarisation <=0.2
            depol_err = AEBD['particle_linear_depol_ratio_355nm_low_resolution_error'].values
            data = np.where(depol_err<=0.2,data,0)

            lat = AEBD['latitude'].values
            lon = AEBD['longitude'].values
            qc = AEBD['quality_status'].values
            time = AEBD['time'].values

            data = np.where(qc<2,data,0)
            data = np.where(data<1e-3,data,0)
            data = np.where(data>=0,data,0)
            data = np.where(tc>0,data,0)  #only keep aerosol particles

            #mask the columns when there are liquid clouds or fully attenuated
            mask = (tc2>=-1).any(axis=1)
            data[mask,:] = np.nan

            #Trim the rows when height is negative
            height = AEBD['height'].values

            reversed_data = data[:,::-1]
            reversed_height = height[:,::-1]
            aod = np.trapz(reversed_data,x=reversed_height,axis=1)
        return aod, lat, lon,time
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

