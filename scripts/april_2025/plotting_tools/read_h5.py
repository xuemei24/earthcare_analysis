import numpy as np
import glob
from ectools.ectools_edited import ecio

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
            tcnan = tc.copy()
            tc_clear = tc.copy()
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
            tc = np.where(qc==0,tc,np.nan)
            tc_clear = np.where(qc==0,tc_clear,np.nan)
 
            aer_index = np.array([10,11,12,13,14,15,25,26,27])
            tc = np.where(np.isin(tc,aer_index),tc,np.nan)

            nan_flag = np.array([-3,-2,-1,1,2,3,20,21,22])
            tc_nan = np.where(np.isin(tcnan,nan_flag),tcnan,np.nan)
        with ecio.load_AEBD(file_path) as AEBD:
            data = AEBD['particle_extinction_coefficient_355nm_low_resolution'][:,select_height:].values
            data = np.where(tc>0,data,np.nan)  #only keep aerosol particles
            err  = AEBD['particle_extinction_coefficient_355nm_low_resolution_error'][:,select_height:].values
            qc = AEBD['quality_status'][:,select_height:].values
            lat = AEBD['latitude'].values
            lon = AEBD['longitude'].values
            time = AEBD['time'].values

            data = np.where(data>=0,data,np.nan)# used to be 0

            ### filter with snr and replace with 0
            snr_threshold = 2
            snr = np.where(err != 0, data / err, snr_threshold+1)
            data = np.where(snr>=snr_threshold,data,0)

            #filter data with deploarisation
            depol = AEBD['particle_linear_depol_ratio_355nm_low_resolution'][:,select_height:].values
            depol_err = AEBD['particle_linear_depol_ratio_355nm_low_resolution_error'][:,select_height:].values
            snr = np.where(depol_err != 0, depol / depol_err, snr_threshold+1)
            data = np.where(snr>=snr_threshold,data,0)

            # set clear region as 0
            data = np.where(tc_clear==0,0,data)
            data = np.where(tc_nan>=-3, np.nan, data)


            def filter_data_col(dt):
                dt = np.where(qc==0,dt,np.nan)  # used to be 0
                #mask the columns when there are liquid clouds or fully attenuated
                mask = (tc2>=-1).any(axis=1)
                dt[mask,:] = np.nan
                return dt

            data = filter_data_col(data)

            #data = np.where(data<1e-3,data,np.nan)# used to be 0

            err  = filter_data_col(err) 

            #Trim the rows when height is negative
            geoid = AEBD['geoid_offset'].values
            geoid_2d = geoid[:,np.newaxis]
            height = AEBD['height'][:,select_height:].values
 
            elev = AEBD['elevation'].values
            elev_2d = elev[:, np.newaxis]
            data = np.where(height > elev_2d, data, np.nan)

            height = height-geoid_2d
            elev = elev - geoid

        return data[:,::-1], lat, lon,time, height[:,::-1],elev,tc_cld[:,::-1],tct[:,::-1],err[:,::-1]#,ATC

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

            tc_clear = tc.copy()
            tct = tc.copy()
            tcnan = tc.copy()
            for ni in range(tc.shape[0]):
                for nj in range(1,tc.shape[1]):
                    if tc[ni,nj-1]==-2 and tc[ni,nj]==-1:
                        tct[ni,nj]=-2
            tc = tct
            tc_cld = tc.copy()
            cld_index = np.array([-2, 1, 2, 3, 20, 21, 22])
            tc_cld = np.where(np.isin(tc_cld,cld_index),tc_cld,np.nan)
 
            tc = np.where(qc==0,tc,np.nan)
            tc = np.where(np.isin(tc,aer_index),tc,np.nan)
            tc_clear = np.where(qc==0,tc_clear,np.nan)
 
            nan_flag = np.array([-3,-2,-1,1,2,3,20,21,22])
            tc_nan = np.where(np.isin(tcnan,nan_flag),tcnan,np.nan)
        with ecio.load_AEBD(file_path) as AEBD:
            data = AEBD['particle_extinction_coefficient_355nm_low_resolution'][:,select_height:].values
            data = np.where(tc>0,data,np.nan) # used to be 0  #only keep aerosol particles   
            err  = AEBD['particle_extinction_coefficient_355nm_low_resolution_error'][:,select_height:].values
            qc = AEBD['quality_status'][:,select_height:].values
            lat = AEBD['latitude'].values
            lon = AEBD['longitude'].values
            time = AEBD['time'].values

            data = np.where(data>=0,data,np.nan) # used to be 0                    

            ### filter with snr and replace with 0
            snr_threshold = 2                                       
            snr = np.where(err != 0, data / err, snr_threshold+1)   
            data = np.where(snr>=snr_threshold,data,0)

            #filter data with deploarisation <=0.2
            depol = AEBD['particle_linear_depol_ratio_355nm_low_resolution'][:,select_height:].values
            depol_err = AEBD['particle_linear_depol_ratio_355nm_low_resolution_error'][:,select_height:].values
            snr = np.where(depol_err != 0, depol / depol_err, snr_threshold+1)
            data = np.where(snr>=snr_threshold,data,0)

            # setting ext in clear as 0
            data = np.where(tc_clear==0,0,data)
            data = np.where(tc_nan>=-3, np.nan, data)

            def filter_data(dt):
                dt = np.where(qc==0,dt,np.nan) # used to be 0                                 
                return dt

            data = filter_data(data)
            #data = np.where(data<1e-3,data,np.nan) # used to be 0                  

            err  = filter_data(err)
            
            geoid = AEBD['geoid_offset'].values
            geoid_2d = geoid[:,np.newaxis]
            height = AEBD['height'][:,select_height:].values

            elev = AEBD['elevation'].values
            elev_2d = elev[:, np.newaxis]
            data = np.where(height > elev_2d, data, np.nan)

            height = height-geoid_2d
            elev = elev - geoid

            reversed_data = data[:,::-1]
            reversed_height = height[:,::-1]
            tc_cld = tc_cld[:,::-1]
        return reversed_data, lat, lon,time, reversed_height,elev,tc_cld,tct[:,::-1],err[:,::-1]#,ATC

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

            tc = np.where(qc==0,tc,np.nan)

            tc = np.where(np.isin(tc,aer_index),tc,np.nan)

        with ecio.load_AEBD(file_path) as AEBD:
            data = AEBD['particle_extinction_coefficient_355nm_low_resolution'].values
            err  = AEBD['particle_extinction_coefficient_355nm_low_resolution_error'].values
            lat = AEBD['latitude'].values
            lon = AEBD['longitude'].values
            time = AEBD['time'].values
            def filter_data(dt):
                dt = np.where(qc==0,dt,np.nan) #was 0
                dt = np.where(tc>0,dt,np.nan) #was 0 #only keep aerosol particles
                return dt

            err  = filter_data(err)

            ########filtering err with snr >= 2
            snr_threshold = 2
            snr = np.where(err != 0, data / err, snr_threshold+1)
            err = np.where(snr>=snr_threshold,err,0)

            #filter data with deploarisation <=0.2
            depol = AEBD['particle_linear_depol_ratio_355nm_low_resolution'][:,select_height:].values
            depol_err = AEBD['particle_linear_depol_ratio_355nm_low_resolution_error'][:,select_height:].values
            snr = np.where(depol_err != 0, depol / depol_err, snr_threshold+1)
            data = np.where(snr>=snr_threshold,data,0)

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
            tc = np.where(qc==0,tc,np.nan)

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
            data = np.where(qc==0,data,0)
            #data = np.where(data<1e-3,data,0)
            data = np.where(data>=0,data,0)
            data = np.where(tc>0,data,0)  #only keep aerosol particles

            #mask the columns when there are liquid clouds or fully attenuated
            mask = (tc2>=-1).any(axis=1)
            data[mask,:] = np.nan

            #Trim the rows when height is negative
            height = AEBD['height'].values

            reversed_data = data[:,::-1]
            reversed_height = height[:,::-1]
            aod = np.trapezoid(reversed_data,x=reversed_height,axis=1)

            '''
            #The following routine identifies the surface altitude for 
            #ATLID-AERONET co-located profiles. It extracts the extinction 
            #coefficient near the surface and assumes a constant extinction 
            #profile below the detected topography (e.g., mountainous terrain).
            #This allows for the calculation of an 'estimated AOD' by extrapolating 
            #the near-surface extinction to the hidden atmospheric column.

            #calculate h
            geoid = AEBD['geoid_offset'].values
            geoid = geoid[:,np.newaxis]
            h = AEBD['height'].values-geoid
            print('start plotting!')
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(1)
            im=ax.pcolormesh(reversed_data.transpose(),cmap='jet')
            xdata = np.arange(data.shape[0])
            #selected co-locating profiles
            mask = (lat<=19.0) & (lat>=18.2)
            xdata = xdata[mask]
            print(xdata[0],xdata[-1])
            ax.axvspan(xdata[0],xdata[-1],color='red')
            ax.set_xlim(0,1000)
            fig.colorbar(im, ax=ax)
            fig.savefig('ext.jpg')
            print(h)


            arr = reversed_data[379:466,:]
            print(np.nanmin(h[379:466],axis=1))
            print(len(np.nanmin(h[379:466],axis=1)))
            fig,(ax,ax2)=plt.subplots(2,1)
            im=ax.pcolormesh(arr.transpose(),cmap='jet')
            fig.colorbar(im, ax=ax)

            mask = ~np.isnan(arr) & (arr > 0)
            has_positive = mask.any(axis=1)
            first_indices = mask.argmax(axis=1)
            first_positive_values = arr[np.where(has_positive)[0], first_indices[has_positive]]

            ax2.plot(np.arange(len(first_positive_values)),first_positive_values)
            fig.tight_layout()
            fig.savefig('extv2.jpg')

            #extraploate extinction coefficient
            loss_aod = np.nanmean(first_positive_values)*np.nanmean(np.nanmin(h[379:466],axis=1))
            print('loss_aod=',loss_aod)
            '''

        return aod, lat, lon,time
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def get_aod_ald(file_path):
    try:
        with ecio.load_AALD(file_path) as ALD:
            data = ALD['aerosol_optical_thickness_355nm'].values
            err  = ALD['aerosol_optical_thickness_355nm_error'].values           
            snr_threshold = 2
            snr = np.where(err != 0, data / err, snr_threshold+1)
            data = np.where(snr>=snr_threshold,data,0)

            lat = ALD['latitude'].values
            lon = ALD['longitude'].values
            time = ALD['time'].values
            data = np.where(data<10,data,0)
        return data, lat, lon,time
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def get_profiles(ANOM,ANOM_h,AEBD,ATC,stlat,edlat):
    try:
        #######ANOM#######
        mie = ANOM['mie_attenuated_backscatter'].values
        cro = ANOM['crosspolar_attenuated_backscatter'].values
        ray = ANOM['rayleigh_attenuated_backscatter'].values

        lat0 = ANOM['latitude'].values
        lon0 = ANOM['longitude'].values
        time0 = ANOM['time'].values
        height0 = ANOM['height'].fillna(-1000).values

        mask0 = (lat0>=stlat) & (lat0<=edlat)
        lat_filtered0 = lat0[mask0]    
        lon_filtered0 = lon0[mask0]    
        time_filtered0 = time0[mask0]
        mie = mie[mask0, :]
        cro = cro[mask0, :]
        ray = ray[mask0, :]
        height_filtered0 = ANOM_h[mask0,:]

        #######AEBD#######       
        qc = ATC['quality_status'].values
        tc = ATC['classification_low_resolution'].values

        tct = tc.copy()
        for ni in range(tc.shape[0]):
            for nj in range(1,tc.shape[1]):
                if tc[ni,nj-1]==-2 and tc[ni,nj]==-1:
                    tct[ni,nj]=-2
        tc = tct
        tc = np.where(qc==0,tc,np.nan)

        #total aerosol
        aer_index = np.array([10,11,12,13,14,15,25,26,27])
        tc = np.where(np.isin(tc,aer_index),tc,np.nan)

        time = AEBD['time'].values
        data = AEBD['particle_extinction_coefficient_355nm_low_resolution'].values
        data = np.where(tc>0,data,np.nan)  #only keep aerosol particles
        err  = AEBD['particle_extinction_coefficient_355nm_low_resolution_error'].values     
        snr_threshold = 2
        snr = np.where(err != 0, data / err, snr_threshold+1)
        ext = np.where(snr>=snr_threshold,data,0)
        snr_out = snr.copy()

        backscatter = AEBD['particle_backscatter_coefficient_355nm_low_resolution'].values
        backscatter = np.where(tc>0,backscatter,np.nan)  #only keep aerosol particles
        backscatter_err = AEBD['particle_backscatter_coefficient_355nm_low_resolution_error'].values
        snr = np.where(backscatter_err != 0, backscatter / backscatter_err, snr_threshold+1)
        backscatter = np.where(snr>=snr_threshold,backscatter,0)

        depol = AEBD['particle_linear_depol_ratio_355nm_low_resolution'].values
        depol = np.where(tc>0,depol,np.nan)  #only keep aerosol particles
        depol_err = AEBD['particle_linear_depol_ratio_355nm_low_resolution_error'].values
        snr = np.where(depol_err != 0, depol / depol_err, snr_threshold+1)
        depol = np.where(snr>=snr_threshold,depol,0)

        lidarratio = AEBD['lidar_ratio_355nm_low_resolution'].values
        lidarratio = np.where(tc>0,lidarratio,np.nan)  #only keep aerosol particles
        lidarratio_err = AEBD['lidar_ratio_355nm_low_resolution_error'].values
        snr = np.where(lidarratio_err != 0, lidarratio / lidarratio_err, snr_threshold+1)
        lidarratio = np.where(snr>=snr_threshold,lidarratio,0)
        
        height = AEBD['height'].values

        lat = AEBD['latitude'].values
        lon = AEBD['longitude'].values
        mask = (lat>=stlat) & (lat<=edlat)
        lat_filtered = lat[mask]    
        lon_filtered = lon[mask]    
        time_filtered = time[mask]
        ext_filtered = ext[mask, :]
        backscatter_filtered = backscatter[mask, :]
        depol_filtered = depol[mask, :]
        lidarratio_filtered = lidarratio[mask, :]
        height_filtered = height[mask, :]
        snr_out = snr_out[mask,:]

        return height_filtered[:,::-1],lidarratio_filtered[:,::-1],depol_filtered[:,::-1],backscatter_filtered[:,::-1],ext_filtered[:,::-1],lat_filtered,lon_filtered,time_filtered,mie[:,::-1],cro[:,::-1],ray[:,::-1],height_filtered0[:,::-1],lat_filtered0,lon_filtered0,time_filtered0,snr_out
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def get_profiles_test(ANOM,ANOM_h,AEBD,ATC,stlat,edlat):
    try:
        #######ANOM#######
        mie = ANOM['mie_attenuated_backscatter'].values
        cro = ANOM['crosspolar_attenuated_backscatter'].values
        ray = ANOM['rayleigh_attenuated_backscatter'].values

        lat0 = ANOM['latitude'].values
        lon0 = ANOM['longitude'].values
        time0 = ANOM['time'].values
        height0 = ANOM['height'].fillna(-1000).values

        mask0 = (lat0>=stlat) & (lat0<=edlat)
        lat_filtered0 = lat0[mask0]
        lon_filtered0 = lon0[mask0]
        time_filtered0 = time0[mask0]
        mie = mie[mask0, :]
        cro = cro[mask0, :]
        ray = ray[mask0, :]
        height_filtered0 = ANOM_h[mask0,:]

        #######AEBD#######       
        qc = ATC['quality_status'].values
        tc = ATC['classification_low_resolution'].values

        tct = tc.copy()
        for ni in range(tc.shape[0]):
            for nj in range(1,tc.shape[1]):
                if tc[ni,nj-1]==-2 and tc[ni,nj]==-1:
                    tct[ni,nj]=-2
        tc = tct
        tc = np.where(qc==0,tc,np.nan)

        #total aerosol
        aer_index = np.array([10,11,12,13,14,15,25,26,27])
        tc = np.where(np.isin(tc,aer_index),tc,np.nan)

        time = AEBD['time'].values
        data = AEBD['particle_extinction_coefficient_355nm_low_resolution'].values
        data = np.where(tc>0,data,np.nan)  #only keep aerosol particles
        ext = data
        err  = AEBD['particle_extinction_coefficient_355nm_low_resolution_error'].values
        snr_threshold = 2
        snr = np.where(err != 0, data / err, snr_threshold+1)
        snr_out = snr.copy()

        backscatter = AEBD['particle_backscatter_coefficient_355nm_low_resolution'].values
        backscatter = np.where(tc>0,backscatter,np.nan)  #only keep aerosol particles
        backscatter_err = AEBD['particle_backscatter_coefficient_355nm_low_resolution_error'].values

        depol = AEBD['particle_linear_depol_ratio_355nm_low_resolution'].values
        depol = np.where(tc>0,depol,np.nan)  #only keep aerosol particles
        depol_err = AEBD['particle_linear_depol_ratio_355nm_low_resolution_error'].values

        lidarratio = AEBD['lidar_ratio_355nm_low_resolution'].values
        lidarratio = np.where(tc>0,lidarratio,np.nan)  #only keep aerosol particles
        lidarratio_err = AEBD['lidar_ratio_355nm_low_resolution_error'].values

        height = AEBD['height'].values

        lat = AEBD['latitude'].values
        lon = AEBD['longitude'].values
        mask = (lat>=stlat) & (lat<=edlat)
        lat_filtered = lat[mask]
        lon_filtered = lon[mask]
        time_filtered = time[mask]
        ext_filtered = ext[mask, :]
        backscatter_filtered = backscatter[mask, :]
        depol_filtered = depol[mask, :]
        lidarratio_filtered = lidarratio[mask, :]
        height_filtered = height[mask, :]
        snr_out = snr_out[mask,:]

        return height_filtered[:,::-1],lidarratio_filtered[:,::-1],depol_filtered[:,::-1],backscatter_filtered[:,::-1],ext_filtered[:,::-1],lat_filtered,lon_filtered,time_filtered,mie[:,::-1],cro[:,::-1],ray[:,::-1],height_filtered0[:,::-1],lat_filtered0,lon_filtered0,time_filtered0,snr_out
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

