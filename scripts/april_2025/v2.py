def cmp_ebd(fn1, data, data_b, dist_min, site, dir_figure):

    lat = data['latitude']
    lon = data['longitude']
    time_bounds = data['time_bounds']

    #t0 = datetime.fromtimestamp(time_bounds[0])  # [1]  local time
    #t1 = datetime.fromtimestamp(time_bounds[1])  # [1]
    #t0 = datetime.utcfromtimestamp(time_bounds[0])
    t0 = datetime.datetime.fromtimestamp(time_bounds[0], datetime.UTC)

    ebd_data = read_atlid_l2(fn1)

    # look for collocated ec product
    lat_a =  ebd_data['latitude']
    lon_a =  ebd_data['longitude']

    dd = np.sqrt((lat_a - lat)**2 + (lon_a - lon)**2)
    sel = np.argmin(dd)

    p1 = (lat_a[sel], lon_a[sel])
    p2 = (lat, lon)
    dist =   geodesic(p1,p2).km


    dd1 = np.zeros(len(lat_a))+1.e8
    for i in range(len(lat_a)):
        p1 = (lat_a[i], lon_a[i])
        p2 = (lat, lon)
        dd1[i] =   geodesic(p1,p2).km

    sel1 = np.argmin(dd1)
    dist1 = dd1[sel1]    
        
    print('closest distance in lat/lon: ', dist1, dd1[sel1], 'pixel: ', sel1)
#    print('closest distance in lat/lon old: ', sel, dd[sel], dist)

    sel = np.copy(sel1)
    dist = np.copy(dist1)


    if dist < dist_min:  #200:      


        #sel_aer3 = np.where(ebd_data['simple_classification'][sel,:] >=2.9)   # 2 ice clouds,  3 aerosol

        ec_st1, ec_st2 = ec_time_to_datetime_str(ebd_data['time'][sel])

        title0 = 'EBD' + ' ' + os.path.basename(fn1)[-9:-3] + ' ' + ec_st2 + ' '+str(sel)+'; E ' + site + ' T ' + str(t0.hour) +  ' D ' + str(dist)[0:4]
        #ec_product1[4:7]
        ext_var_list= ['particle_extinction_coefficient_355nm', 
                       'particle_extinction_coefficient_355nm_medium_resolution',
                       'particle_extinction_coefficient_355nm_low_resolution']

        for ext_var in ext_var_list: 
            ext_ec = ebd_data[ext_var][sel,:]
            ext_ec_err = ebd_data[ext_var+'_error'][sel,:]#[10,:]
            height_ec =  ebd_data['height'][sel,:]#[10,:] 

            if    ext_var.find('low') >=0:
                title = title0 + ' L'
            elif    ext_var.find('medium') >=0:
                title = title0 + ' M'
            else:
                title = title0

            outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_' + ext_var +'_'+ site+ '.png'
        #txt = os.path.basename(outfile)[13:36] + ' ' + os.path.basename(outfile)[54:60] +  ' ' +   os.path.basename(outfile)[-7:-4]      
            
            plot_ecvt_atl_ext_profile_fill(data['extinction'], data['error_extinction'], data['altitude'], z1, z2, outfile=outfile_ec, 
                              ext_ec=ext_ec, ext_ec_err=ext_ec_err, height_ec=height_ec,title=title)


        bks_var_list= ['particle_backscatter_coefficient_355nm', 
                       'particle_backscatter_coefficient_355nm_medium_resolution',
                       'particle_backscatter_coefficient_355nm_low_resolution']
        for bks_var in bks_var_list: 
            backscatter_ec = ebd_data[bks_var][sel,:]
            backscatter_ec_err = ebd_data[bks_var+'_error'][sel,:]#[10,:]
            height_ec =  ebd_data['height'][sel,:]#[10,:] 
            outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_'+bks_var + '_'+ site+ '.png'
            if bks_var.find('low') >=0:
                title = title0 + ' L'
            elif bks_var.find('medium') >=0:
                title = title0 + ' M'
            else:
                title = title0

            plot_ecvt_atl_backscatter_profile_fill(data['backscatter'], data['error_backscatter'], data['altitude'], z1, z2, outfile=outfile_ec, 
                                  backscatter_ec=backscatter_ec, backscatter_ec_err=backscatter_ec_err, height_ec=height_ec,title=title)


        lidarratio_var_list = ['lidar_ratio_355nm',
                               'lidar_ratio_355nm_medium_resolution',
                               'lidar_ratio_355nm_low_resolution']

 237         for depol_var in depol_var_list:
 238 
 239             depol_ec = ebd_data[depol_var][sel,:]
 240             depol_ec_err = ebd_data[depol_var + '_error'][sel,:]#[10,:]
 241             height_ec =  ebd_data['height'][sel,:]#[10,:] 
 242 
 243             outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_'+ depol_var + '_'+ site+ '.png'
 244 
 245             if depol_var.find('low') >=0:
 246                 title = title0 + ' L'
 247             elif depol_var.find('medium') >=0:
 248                 title = title0 + ' M'
 249             else:
 250                 title = title0
 251 
 252             plot_ecvt_atl_depol_profile_fill(data_b['particledepolarization'], data_b['error_particledepolarization'], data_b['altitude'], z1, z2, outfile=outfile_ec,
 253                               depol_ec=depol_ec, depol_ec_err=depol_ec_err, height_ec=height_ec,title=title)
 254 
 255 
 256 
 257         ebd_data_collocate_info = {'filename': fn1, 'pixel': sel, 'distant_km': dist,  'product': 'EBD',
 258                                    'time_ec':  ec_st2, 'site':  site,  'ground_based_time': t0.hour}
 259 
 260         return  ebd_data, ebd_data_collocate_info
 261 
 262     else:
 263         print('!! No data within dt (1.5h) and 200 km site, date', site, date)
 264         return 0, 0

