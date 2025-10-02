#!/usr/bin/env python3

import glob
import os, sys,time
import shutil
#import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
from TLE_class_daily import TLE_class 

#from datetime import datetime

import datetime

from datetime import timedelta
import subprocess

import h5py

from geopy.distance import geodesic

plt.rcParams.update({'font.size': 15})
plt.rcParams['savefig.dpi'] = 300

# read check collocated data
# find_earthcare_ground_site_collocation_v1.py
#
# read_collocated_ground_site.py
# compare collocated data
# 

from comp_atlid_earlinet_v2 import read_ecvt_earlinet_file,  \
              plot_ecvt_ext_profile, plot_ecvt_backscatter_profile, \
              plot_ecvt_lidarratio_profile, plot_ecvt_depol_profile, \
              find_frame, ec_time_to_datetime_str, \
              read_atlid_l2 

from  comp_atlid_earlinet_v2 import  find_atl_l2_collocated_files, \
              plot_ecvt_atl_ext_profile_fill, \
              plot_ecvt_atl_backscatter_profile_fill, \
              plot_ecvt_atl_lidarratio_profile_fill, \
              plot_ecvt_atl_depol_profile_fill
 


def plot_earlinet_figures(data,data_b,fn_e,fn_b,z1,z2):

    alt = data['altitude']
    ext = data['extinction']
    ext_err = data['error_extinction']

    outfile = os.path.dirname(fn_e) + '/figures/'+ os.path.basename(fn_e)[0:-3]+'_ext.png'

    #outfile = fn_e[0:-3]+'_ext.png'

    plot_ecvt_ext_profile(ext, ext_err, alt, z1, z2, outfile=outfile)

    bks = data['backscatter']
    bks_err = data['error_backscatter']
    #outfile = fn_e[0:-3]+'_backscatter.png'
    outfile = os.path.dirname(fn_e) + '/figures/'+ os.path.basename(fn_e)[0:-3]+'_backscatter.png'

    plot_ecvt_backscatter_profile(bks, bks_err, alt, z1, z2, outfile=outfile)

    
    outfile1 = os.path.dirname(fn_e) + '/figures/'+ os.path.basename(fn_e)[0:-3]+'_S.png'
    #outfile1 = fn_e[0:-3]+'_S.png'
    lidarratio = data['lidarratio']

    plot_ecvt_lidarratio_profile(lidarratio, alt, z1, z2, outfile=outfile1)

    outfile1 = os.path.dirname(fn_b) + '/figures/'+ os.path.basename(fn_b)[0:-3]+'_depol.png'
    #outfile1 = fn_b[0:-3]+'_depol.png'

    if len(data_b) == 0:
        print('no depol data, skip depol plot')
        return
    else:

        alt_b = data_b['altitude']
        
        if 'volumedepolarization' in  data_b.keys():
            volume_depol = data_b['volumedepolarization']
        else:
            print('no   volumedepolarization')
            volume_depol = np.zeros(len(alt_b))

        if 'particledepolarization' in data_b.keys():
            depol = data_b['particledepolarization']
        else:
            depol = np.zeros(len(alt_b))
            print('no  particledepolarization')

        plot_ecvt_depol_profile(depol, volume_depol, alt_b, z1, z2, outfile=outfile1)

        return   


def plot_aer_classification(aer_data, data, sel, z1, z2, outfile=None, title=None):

    for i in range(-5,11,1): 
        plt.plot(aer_data['classification'][sel+i,:], aer_data['height'][sel+i,:]*1.e-3, 'k.')

    if 'cloud_mask'  in data.keys():
        plt.plot(data['cloud_mask'][:], data['altitude'][:]*1.e-3, 'b*')
    else:
        print('no cloud_mask in earlinet data')
    plt.ylim(z1, z2)  #z2-0.1)
    plt.xlim([-3, 27])
    plt.xlabel('Classification')  
    plt.ylabel('Height [km]')  #Height
    plt.grid()
    plt.title(title, fontsize=14)
    plt.tight_layout()

    plt.savefig(outfile)


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

        for lidarratio_var in lidarratio_var_list:
            lidarratio_ec = ebd_data[lidarratio_var][sel,:]
            lidarratio_ec_err = ebd_data[lidarratio_var + '_error'][sel,:]#[10,:]
            height_ec =  ebd_data['height'][sel,:]#[10,:] 

            outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_' + lidarratio_var +'_'+ site+ '.png'

            if lidarratio_var.find('low') >=0:
                title = title0 + ' L'
            elif lidarratio_var.find('medium') >=0:
                title = title0 + ' M'
            else:
                title = title0

            plot_ecvt_atl_lidarratio_profile_fill(data['lidarratio'], data['error_lidarratio'], data['altitude'], z1, z2, outfile=outfile_ec, 
                              lidarratio_ec=lidarratio_ec, lidarratio_ec_err=lidarratio_ec_err, height_ec=height_ec,title=title)

        depol_var_list = ['particle_linear_depol_ratio_355nm',
                          'particle_linear_depol_ratio_355nm_medium_resolution',
                          'particle_linear_depol_ratio_355nm_low_resolution']
        for depol_var in depol_var_list:

            depol_ec = ebd_data[depol_var][sel,:]
            depol_ec_err = ebd_data[depol_var + '_error'][sel,:]#[10,:]
            height_ec =  ebd_data['height'][sel,:]#[10,:] 

            outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_'+ depol_var + '_'+ site+ '.png'

            if depol_var.find('low') >=0:
                title = title0 + ' L'
            elif depol_var.find('medium') >=0:
                title = title0 + ' M'
            else:
                title = title0

            plot_ecvt_atl_depol_profile_fill(data_b['particledepolarization'], data_b['error_particledepolarization'], data_b['altitude'], z1, z2, outfile=outfile_ec, 
                              depol_ec=depol_ec, depol_ec_err=depol_ec_err, height_ec=height_ec,title=title)



        ebd_data_collocate_info = {'filename': fn1, 'pixel': sel, 'distant_km': dist,  'product': 'EBD',
                                   'time_ec':  ec_st2, 'site':  site,  'ground_based_time': t0.hour}  

        return  ebd_data, ebd_data_collocate_info

    else:
        print('!! No data within dt (1.5h) and 200 km site, date', site, date)
        return 0, 0


def cmp_aer(fn_a, data, data_b, dist_min, site, dir_figure):

    lat = data['latitude']
    lon = data['longitude']
    time_bounds = data['time_bounds']

    #t0 = datetime.fromtimestamp(time_bounds[0])  # [1]  local time
    #t1 = datetime.fromtimestamp(time_bounds[1])  # [1]
    #t0 = datetime.utcfromtimestamp(time_bounds[0])
    t0 = datetime.datetime.fromtimestamp(time_bounds[0], datetime.UTC)

    aer_data = read_atlid_l2(fn_a)


    # look for collocated ec product
    lat_a =  aer_data['latitude']
    lon_a =  aer_data['longitude']

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
    #print('closest distance in lat/lon old: ', dist, dd[sel],  'pixel: ', sel)

    sel = np.copy(sel1)
    dist = np.copy(dist1)

    #if dd[sel] < 2:      
    if dist < dist_min:  #200: 
        ec_st1, ec_st2 = ec_time_to_datetime_str(aer_data['time'][sel])

        #print('closest distance in lat/lon: ', dist, dd[sel],  'pixel: ', sel)

        title = ec_product2[4:7] + ' ' + os.path.basename(fn_a)[-9:-3] + ' ' + ec_st2 + '; E ' + site + ' T ' + str(t0.hour) +  ' D ' + str(dist)[0:4]          


        ext_ec_a = aer_data['particle_extinction_coefficient_355nm'][sel,:]
        ext_ec_err_a = aer_data['particle_extinction_coefficient_355nm_error'][sel,:]#[10,:]
        height_ec_a =  aer_data['height'][sel,:]#[10,:] 

        outfile_ec_a = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_ext_'+ site+ '.png'

        plot_ecvt_atl_ext_profile_fill(data['extinction'], data['error_extinction'], data['altitude'], z1, z2, outfile=outfile_ec_a, 
                              ext_ec=ext_ec_a, ext_ec_err=ext_ec_err_a, height_ec=height_ec_a, title=title)


        backscatter_ec_a = aer_data['particle_backscatter_coefficient_355nm'][sel,:]
        backscatter_ec_err_a = aer_data['particle_backscatter_coefficient_355nm_error'][sel,:]#[10,:]
        height_ec_a =  aer_data['height'][sel,:]#[10,:] 
        outfile_ec_a = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_backscatter_'+ site+ '.png'

        plot_ecvt_atl_backscatter_profile_fill(data['backscatter'], data['error_backscatter'], data['altitude'], z1, z2, outfile=outfile_ec_a, 
                              backscatter_ec=backscatter_ec_a, backscatter_ec_err=backscatter_ec_err_a, height_ec=height_ec_a,title=title)


        lidarratio_ec_a = aer_data['lidar_ratio_355nm'][sel,:]
        lidarratio_ec_err_a = aer_data['lidar_ratio_355nm_error'][sel,:]#[10,:]
        height_ec_a =  aer_data['height'][sel,:]#[10,:] 

        outfile_ec_a = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_lidarratio_'+ site+ '.png'


        plot_ecvt_atl_lidarratio_profile_fill(data['lidarratio'], data['error_lidarratio'], data['altitude'], z1, z2, outfile=outfile_ec_a, 
                              lidarratio_ec=lidarratio_ec_a, lidarratio_ec_err=lidarratio_ec_err_a, height_ec=height_ec_a,title=title)


        depol_ec_a = aer_data['particle_linear_depol_ratio_355nm'][sel,:]
        depol_ec_err_a = aer_data['particle_linear_depol_ratio_355nm_error'][sel,:]#[10,:]
        height_ec_a =  aer_data['height'][sel,:]#[10,:] 

        outfile_ec_a = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_depol_'+ site+ '.png'

        plot_ecvt_atl_depol_profile_fill(data_b['particledepolarization'], data_b['error_particledepolarization'], data_b['altitude'], z1, z2, outfile=outfile_ec_a, 
                              depol_ec=depol_ec_a, depol_ec_err=depol_ec_err_a, height_ec=height_ec_a,title=title)


        outfile_ec_a = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_classification_'+ site+ '.png'

        plot_aer_classification(aer_data, data, sel, z1, z2, outfile=outfile_ec_a, title=title)

        aer_data_collocate_info = {'filename': fn_a, 'pixel': sel, 'distant_km': dist,  'product': ec_product2[4:7],
                                   'time_ec':  ec_st2, 'site':  site,  'ground_based_time': t0.hour}  

        return aer_data, aer_data_collocate_info
                
    else:
        print('!! No data within 1 h and 200 km site, date', site, date)
        return 0, 0
 
def cmp_ebd_aer_filter(fn1, fn_a, data, data_b, dist_min, site, dir_figure):
    # read ebd, aer data, select ebd using aer classification.  plot ebd, aer

    lat = data['latitude']
    lon = data['longitude']
    time_bounds = data['time_bounds']

    #t0 = datetime.utcfromtimestamp(time_bounds[0])
    t0 = datetime.datetime.fromtimestamp(time_bounds[0], datetime.UTC)

    aer_data = read_atlid_l2(fn_a)

    # look for collocated ec product
    lat_a =  aer_data['latitude']
    lon_a =  aer_data['longitude']

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

    sel1 = np.argmin(dd1)   # pixel with smallest distance to a stie
    dist1 = dd1[sel1]       # the smallest distance 
        
        
    print('closest distance in lat/lon: ', dist1, 'pixel: ', sel1)

    #print('closest distance in lat/lon old: ', dist, dd[sel],  'pixel: ', sel)

    sel = np.copy(sel1)
    dist = np.copy(dist1)
 
    if dist < dist_min:  #200: 

        # read EBD
        ebd_data = read_atlid_l2(fn1)

 
        ec_st1, ec_st2 = ec_time_to_datetime_str(aer_data['time'][sel])

        # print('closest distance in lat/lon: ', dist, dd[sel],  'pixel: ', sel)

        #title = 'AER ' + os.path.basename(fn_a)[-9:-3] + ' ' + ec_st2 + ' '+str(sel)+ '; ' + site + ' ' + t0.strftime('%H:%M:%S') +  ' D' + str(dist)[0:4]      
        title = os.path.basename(fn_a)[4:8] + ' ' + os.path.basename(fn_a)[-9:-3] + ' ' + ec_st2 + ' '+str(sel)+ '; ' + site + ' ' + t0.strftime('%H:%M:%S') +  ' D' + str(dist)[0:4] 
    


        sel_cls10 = np.where(aer_data['classification'][sel,:] > 9.9)   # >=10  for aerosols  no used here


        ext_ec_a = aer_data['particle_extinction_coefficient_355nm'][sel,:]  #[sel_cls10]
        ext_ec_err_a = aer_data['particle_extinction_coefficient_355nm_error'][sel,:]#[sel_cls10]
        height_ec_a =  aer_data['height'][sel,:]#[sel_cls3]#[10,:] 
        classification = aer_data['classification'][sel,:]


        outfile_ec_a = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_ext_'+ site+ '_P'+str(sel)+'_e'+t0.strftime('%H%M%S')+'.png'
        if os.path.isfile(outfile_ec_a):
            print('ext figure exsit, skip ', outfile_ec_a )
            #return 0, 0, 0

        plot_ecvt_atl_ext_profile_fill(data['extinction'], data['error_extinction'], data['altitude'], z1, z2, outfile=outfile_ec_a, 
                              ext_ec=ext_ec_a, ext_ec_err=ext_ec_err_a, height_ec=height_ec_a, title=title, classification = classification)

        backscatter_ec_a = aer_data['particle_backscatter_coefficient_355nm'][sel,:]
        backscatter_ec_err_a = aer_data['particle_backscatter_coefficient_355nm_error'][sel,:]
        height_ec_a =  aer_data['height'][sel,:] 
        outfile_ec_a = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_backscatter_'+ site+ '_'+str(sel)+'_e'+t0.strftime('%H%M%S')+'.png'

        plot_ecvt_atl_backscatter_profile_fill(data['backscatter'], data['error_backscatter'], data['altitude'], z1, z2, outfile=outfile_ec_a, 
                              backscatter_ec=backscatter_ec_a, backscatter_ec_err=backscatter_ec_err_a, height_ec=height_ec_a,title=title, classification = classification)


        lidarratio_ec_a = aer_data['lidar_ratio_355nm'][sel,:]
        lidarratio_ec_err_a = aer_data['lidar_ratio_355nm_error'][sel,:]#[10,:]
        height_ec_a =  aer_data['height'][sel,:]#[10,:] 

        outfile_ec_a = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_lidarratio_'+ site+ '_'+str(sel)+'_e'+t0.strftime('%H%M%S')+'.png'


        plot_ecvt_atl_lidarratio_profile_fill(data['lidarratio'], data['error_lidarratio'], data['altitude'], z1, z2, outfile=outfile_ec_a, 
                              lidarratio_ec=lidarratio_ec_a, lidarratio_ec_err=lidarratio_ec_err_a, height_ec=height_ec_a,title=title, classification = classification)


        depol_ec_a = aer_data['particle_linear_depol_ratio_355nm'][sel,:]
        depol_ec_err_a = aer_data['particle_linear_depol_ratio_355nm_error'][sel,:]#[10,:]
        height_ec_a =  aer_data['height'][sel,:]#[10,:] 

        if len(data_b) > 10:  # normally more than 10 variables in data_b
            try:
                depol_earlinet = data_b['particledepolarization']
                depol_earlinet_err = data_b['error_particledepolarization']
            except:
                print('!! data_b has no particledepolarization data ', site)
                depol_earlinet = data_b['altitude'] * 0.0
                depol_earlinet_err = data_b['altitude'] * 0.0
        else:
            depol_earlinet = np.copy(depol_ec_a) * 0.0
            depol_earlinet_err = np.copy(depol_ec_a) * 0.0
            data_b['altitude'] = np.copy(height_ec_a)

        outfile_ec_a = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_depol_'+ site+'_'+str(sel)+ '_e'+t0.strftime('%H%M%S')+'.png'

        plot_ecvt_atl_depol_profile_fill(depol_earlinet, depol_earlinet_err, data_b['altitude'], z1, z2, outfile=outfile_ec_a, 
                                      depol_ec=depol_ec_a, depol_ec_err=depol_ec_err_a, height_ec=height_ec_a,title=title, classification = classification)

        outfile_ec_a = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_classification_'+ site+'_e'+t0.strftime('%H%M%S')+ '.png'

        plot_aer_classification(aer_data, data, sel, z1, z2, outfile=outfile_ec_a, title=title)


        stat_data_grd = {}   # ground-based
        stat_data_ec = {}    # earthcare 

        stat_data_ec['ext_ec_a'] =  ext_ec_a
        stat_data_ec['ext_ec_err_a'] =  ext_ec_err_a
        stat_data_ec['height_ec_a'] =  height_ec_a
        stat_data_ec['classification'] =  classification

        stat_data_ec['backscatter_ec_a'] = backscatter_ec_a
        stat_data_ec['backscatter_ec_err_a'] =  backscatter_ec_err_a

        stat_data_ec['lidarratio_ec_a'] = lidarratio_ec_a
        stat_data_ec['lidarratio_ec_err_a'] =  lidarratio_ec_err_a

        stat_data_ec['depol_ec_a'] = depol_ec_a
        stat_data_ec['depol_ec_err_a'] =  depol_ec_err_a

        stat_data_grd['ext_grd'] = data['extinction']
        stat_data_grd['ext_grd_err'] = data['error_extinction']

        stat_data_grd['backscatter_grd'] = data['backscatter']
        stat_data_grd['backscatter_grd_err'] = data['error_backscatter']

        stat_data_grd['lidarratio_grd'] = data['lidarratio']
        stat_data_grd['lidarratio_grd_err'] = data['error_lidarratio']

        # interpolate the ground-based data to earthcare grid.
        for var in stat_data_grd.keys():
            y = np.interp( height_ec_a[::-1], data['altitude'][:],  stat_data_grd[var][:])
            stat_data_ec[var] = np.copy(y[::-1]) 

        stat_data_grd['depol_grd'] = depol_earlinet  #data_b['particledepolarization']
        stat_data_grd['depol_grd_err'] = depol_earlinet_err # data_b['error_particledepolarization']

        stat_data_grd['height_b_grd'] =  data_b['altitude']
        stat_data_grd['height_grd'] =  data['altitude']

        if len(data_b) > 10:    # choose ground-based profile that has more data points
            y = np.interp( height_ec_a[::-1], data_b['altitude'][:],  stat_data_grd['depol_grd'][:])
            stat_data_ec['depol_grd'] = np.copy(y[::-1]) 

            y = np.interp( height_ec_a[::-1], data_b['altitude'][:],  stat_data_grd['depol_grd_err'][:])
            stat_data_ec['depol_grd_err'] = np.copy(y[::-1]) 
        else:
            stat_data_ec['depol_grd'] =  stat_data_grd['depol_grd'][:]     # filled with 0, no need to interpolate
            stat_data_ec['depol_grd_err'] = stat_data_grd['depol_grd_err'][:]

        # EBD


        ec_st1, ec_st2 = ec_time_to_datetime_str(ebd_data['time'][sel])

        #title0 = 'EBD ' + os.path.basename(fn1)[-9:-3] + ' ' + ec_st2 + ' '+str(sel)+'; ' + site + ' ' +t0.strftime('%H:%M:%S')+ ' D ' + str(dist)[0:4]        
        title0 = os.path.basename(fn_a)[4:8] + ' ' + os.path.basename(fn1)[-9:-3] + ' ' + ec_st2 + ' '+str(sel)+'; ' + site + ' ' +t0.strftime('%H:%M:%S')+ ' D ' + str(dist)[0:4]          
  
        #ec_product1[4:7]

        ext_var_list= ['particle_extinction_coefficient_355nm', 
                       'particle_extinction_coefficient_355nm_medium_resolution',
                       'particle_extinction_coefficient_355nm_low_resolution']

        
        height_ec =  ebd_data['height'][sel,:]#[10,:] 
        stat_data_ec['height_ec'] = height_ec
        for ext_var in ext_var_list: 
            ext_ec = ebd_data[ext_var][sel,:]
            ext_ec_err = ebd_data[ext_var+'_error'][sel,:]#[10,:]
     

            if    ext_var.find('low') >=0:
                title = title0 + ' L'
            elif    ext_var.find('medium') >=0:
                title = title0 + ' M'
            else:
                title = title0
 
            stat_data_ec[ext_var] = ext_ec
            stat_data_ec[ext_var+'_error'] = ext_ec_err
           
            outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_' + ext_var +'_'+ site+ '_'+str(sel)+'_e'+t0.strftime('%H%M%S')+'.png'
        #txt = os.path.basename(outfile)[13:36] + ' ' + os.path.basename(outfile)[54:60] +  ' ' +   os.path.basename(outfile)[-7:-4]      
            
            plot_ecvt_atl_ext_profile_fill(data['extinction'], data['error_extinction'], data['altitude'], z1, z2, outfile=outfile_ec, 
                        ext_ec=ext_ec, ext_ec_err=ext_ec_err, height_ec=height_ec,title=title, classification = classification)


        bks_var_list= ['particle_backscatter_coefficient_355nm', 
                       'particle_backscatter_coefficient_355nm_medium_resolution',
                       'particle_backscatter_coefficient_355nm_low_resolution']
   
        height_ec =  ebd_data['height'][sel,:]#[10,:] 
        for bks_var in bks_var_list: 
            backscatter_ec = ebd_data[bks_var][sel,:]
            backscatter_ec_err = ebd_data[bks_var+'_error'][sel,:]#[10,:]
   
            outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_'+bks_var + '_'+ site+ '_'+str(sel)+'_e'+t0.strftime('%H%M%S')+'.png'
            if bks_var.find('low') >=0:
                title = title0 + ' L'
            elif bks_var.find('medium') >=0:
                title = title0 + ' M'
            else:
                title = title0

            stat_data_ec[bks_var] = backscatter_ec
            stat_data_ec[bks_var + '_error'] = backscatter_ec_err

            plot_ecvt_atl_backscatter_profile_fill(data['backscatter'], data['error_backscatter'], data['altitude'], z1, z2, outfile=outfile_ec, 
                                  backscatter_ec=backscatter_ec, backscatter_ec_err=backscatter_ec_err, height_ec=height_ec,title=title, classification = classification)


        lidarratio_var_list = ['lidar_ratio_355nm',
                               'lidar_ratio_355nm_medium_resolution',
                               'lidar_ratio_355nm_low_resolution']

        for lidarratio_var in lidarratio_var_list:
            lidarratio_ec = ebd_data[lidarratio_var][sel,:]
            lidarratio_ec_err = ebd_data[lidarratio_var + '_error'][sel,:]#[10,:]
            height_ec =  ebd_data['height'][sel,:]#[10,:] 

            outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_' + lidarratio_var +'_'+ site+ '_'+str(sel)+'_e'+t0.strftime('%H%M%S')+'.png'

            if lidarratio_var.find('low') >=0:
                title = title0 + ' L'
            elif lidarratio_var.find('medium') >=0:
                title = title0 + ' M'
            else:
                title = title0

            stat_data_ec[lidarratio_var] = lidarratio_ec 
            stat_data_ec[lidarratio_var + '_error'] = lidarratio_ec_err

            plot_ecvt_atl_lidarratio_profile_fill(data['lidarratio'], data['error_lidarratio'], data['altitude'], z1, z2, outfile=outfile_ec, 
                              lidarratio_ec=lidarratio_ec, lidarratio_ec_err=lidarratio_ec_err, height_ec=height_ec,title=title, classification = classification)

        depol_var_list = ['particle_linear_depol_ratio_355nm',
                          'particle_linear_depol_ratio_355nm_medium_resolution',
                          'particle_linear_depol_ratio_355nm_low_resolution']
        for depol_var in depol_var_list:


            depol_ec = ebd_data[depol_var][sel,:]
            depol_ec_err = ebd_data[depol_var + '_error'][sel,:]#[10,:]
            height_ec =  ebd_data['height'][sel,:]#[10,:] 

            outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_'+ depol_var + '_'+ site+ '_'+str(sel)+ '_e'+t0.strftime('%H%M%S')+'.png'

            if depol_var.find('low') >=0:
                title = title0 + ' L'
            elif depol_var.find('medium') >=0:
                title = title0 + ' M'
            else:
                title = title0

            stat_data_ec[depol_var] = depol_ec 
            stat_data_ec[depol_var + '_error'] = depol_ec_err

            plot_ecvt_atl_depol_profile_fill(depol_earlinet, depol_earlinet_err, data_b['altitude'], z1, z2, outfile=outfile_ec, 
                              depol_ec=depol_ec, depol_ec_err=depol_ec_err, height_ec=height_ec,title=title, classification = classification)


        sel10 = np.where(classification > 9.9)   # select aerosols, also for ground-based data
        
        stat_data1 = {}
        if len(sel10[0]) > 1:
            for var in stat_data_ec.keys():
                sel36 = np.logical_and( stat_data_ec[var][sel10] > -1.e-6,  stat_data_ec[var][sel10] < 200)   # remove 1e36 filled values
                stat_data1[var+'_mean'] = np.mean(stat_data_ec[var][sel10][sel36])
                stat_data1[var+'_std'] = np.std(stat_data_ec[var][sel10][sel36])

        return stat_data1, stat_data_ec, stat_data_grd, sel  # mean, std; earthcare data; ground-based data

    else:
        #print('!! No data within dt 1.5 h and 200 km site, date', site, date)
        return 0, 0, 0, 0




def plot_ext_backscatter_means(data, fn_figure):
    ext_mean = np.zeros((25,5))
    backscatter_mean = np.zeros((25,5)) 
    ext_std = np.zeros((25,5))
    backscatter_std = np.zeros((25,5)) 


    title = ''

    it = 0
    #print( data['stat_data'].keys())
    for xsite in data['stat_data'].keys():
        if len(data['stat_data'][xsite]) == 0:
            print('plot_ext_backscatter_means: No stat data at ', xsite)
            continue
        
        else:
 
            ext_mean[it,:] = [data['stat_data'][xsite]['ext_grd_mean'],
                          data['stat_data'][xsite]['ext_ec_a_mean'],
                          data['stat_data'][xsite]['particle_extinction_coefficient_355nm_mean'],
                          data['stat_data'][xsite]['particle_extinction_coefficient_355nm_medium_resolution_mean'],
                          data['stat_data'][xsite]['particle_extinction_coefficient_355nm_low_resolution_mean'] ]


            backscatter_mean[it,:] = [data['stat_data'][xsite]['backscatter_grd_mean'],
                          data['stat_data'][xsite]['backscatter_ec_a_mean'],
                          data['stat_data'][xsite]['particle_backscatter_coefficient_355nm_mean'],
                          data['stat_data'][xsite]['particle_backscatter_coefficient_355nm_medium_resolution_mean'],
                          data['stat_data'][xsite]['particle_backscatter_coefficient_355nm_low_resolution_mean'] ]



            ext_std[it,:] = [data['stat_data'][xsite]['ext_grd_std'],
                          data['stat_data'][xsite]['ext_ec_a_std'],
                          data['stat_data'][xsite]['particle_extinction_coefficient_355nm_std'],
                          data['stat_data'][xsite]['particle_extinction_coefficient_355nm_medium_resolution_std'],
                          data['stat_data'][xsite]['particle_extinction_coefficient_355nm_low_resolution_std'] ]

            backscatter_std[it,:] = [data['stat_data'][xsite]['backscatter_grd_std'],
                          data['stat_data'][xsite]['backscatter_ec_a_std'],
                          data['stat_data'][xsite]['particle_backscatter_coefficient_355nm_std'],
                          data['stat_data'][xsite]['particle_backscatter_coefficient_355nm_medium_resolution_std'],
                          data['stat_data'][xsite]['particle_backscatter_coefficient_355nm_low_resolution_std'] ]


            it = it + 1

            title = title + ' ' + xsite 

    ext_mean =  ext_mean[0:it,:]
    backscatter_mean = backscatter_mean[0:it,:]

    ext_std =  ext_std[0:it,:]
    backscatter_std = backscatter_std[0:it,:]
    x = np.arange(len(ext_mean.flatten()))+1

    plt.figure('ext')  
    plt.plot(x,ext_mean.flatten(), '.-')         
    plt.plot(x,(ext_mean - ext_std).flatten())         
    plt.plot(x,(ext_mean + ext_std).flatten())         
     
    plt.title('ext' + title)   
    plt.yscale('log')   
    plt.ylim([1e-7, 1e-2])
    plt.xlabel('Data count')
    plt.ylabel('Part. ext. coef. [m$^{-1}$]')      
    plt.tight_layout()  
    plt.grid()
    plt.savefig(fn_figure +'_ext.png')
    plt.close()  

    plt.figure('bsk')
    plt.plot(x,backscatter_mean.flatten(), '.-')      
    plt.plot(x,(backscatter_mean - backscatter_std).flatten())         
    plt.plot(x,(backscatter_mean + backscatter_std).flatten())         

    plt.title('bsk' + title)
    plt.yscale('log')   
    plt.ylim([1e-8, 1e-3])
    plt.xlabel('Data count')
    plt.ylabel('Part. backscatter. coef. [m$^{-1}$sr$^{-1}$]')      
    plt.tight_layout()            
    plt.grid()
    plt.savefig(fn_figure + '_backscatter.png')
    plt.close()


def generate_date_list(start_date, end_date):
    import datetime
    start = datetime.datetime.strptime(start_date, "%Y%m%d")
    end = datetime.datetime.strptime(end_date, "%Y%m%d")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days+1)]
   
    date_list = []
    for date in date_generated:
        #print(date.strftime("%Y%m%d"))
        date_list.append(date.strftime("%Y%m%d"))

    return  date_generated, date_list


if __name__ == '__main__':
    import pickle

#    from download_ecvt_data_v1 import generate_date_list

#   This script read collocated earlinet data and EarthCARE data, make plot, write collocated data in a pickle file.
#   Ping Wang, KNMI 202409
# 
    #site = 'dus'  # 'dus' #'puy'  #'puy'  #'dus'  #'gra'  #'dus'  #'hpb'  #dus
    #date = '202408100000_202408100100'
    #date =  '20240824'  #'20240824'  # '20240818'  #'20240822'  #'20240815' #'20240814'  #'20240815'
    dir_data = '/nobackup/users/wangp/Earthcare/data/ECVT/earlinet/'
    #fn_e = 'EARLINET_ECVT_AerRemSen_dus_Lev01_e0355_202408061900_202408062000_v01_qc03.nc'
    #fn_b = 'EARLINET_ECVT_AerRemSen_dus_Lev01_b0355_202408061900_202408062000_v01_qc03.nc'
    dir_figure = '/nobackup/users/wangp/Earthcare/figures/comparison/v2b/'



    site_list = ['arr', 'aky', 'atz', 'brc', 'bgd', 'cog', 'ino', 'cbw', 'cvo', 'puy', 'clj', 'dus', 'evo', 'gar', 'gra', 'ipr', 'kuo', 
                 'sal', 'lei', 'lle', 'cyc', 'lim', 'mdr', 'mas', 'nap', 'hpb', 'sir', 'pay', 'pot', 'rme', 'sof', 'spl', 'the', 'waw']

    #site_list = ['dus', 'gra', 'pot', 'puy', 'hpb']
    #date_list = ['20240814', '20240815', '20240818', '20200820', '20240821', '20240822', '20240824', '20240827','20240828','20240829', '20240830']


    t = datetime.datetime.utcnow()
    #date_list = '20240812,...,' + t.strftime('%Y%m%d')
    #x, date_list = generate_date_list('20240901', '20240930')
    #x, date_list = generate_date_list('20241005', t.strftime('%Y%m%d'))
    x, date_list = generate_date_list('20250101', t.strftime('%Y%m%d'))
    #x, date_list = generate_date_list('20240812', '20240901')

    #site_list = ['cbw'] #['cbw']  #['kuo']  #['pot']  ['
    #date_list = ['20240902' ]  #, '20240927'] #814', '20240905','20240906']
        
    plot_earlinet = 1
    cmp_single = 0

    for date in date_list:  
        nsite = 0
        #ec_grd_data = {}
        #stat_data = {}
        #grd_data = {}


        for site in site_list:

            ec_grd_data = {}
            stat_data = {}
            grd_data = {}


            # search fn_e, fn_b
            s_e0 = dir_data + '*_{0}_*_e0355_{1}*.nc'.format(site, date)
            fn_e_all = glob.glob(s_e0, recursive=True)
            fn_b_all = []    #fill in later

            if len(fn_e_all) == 0:
                print('# No earlinet e0355 data: ', site, ' ' , date)
                continue
                #1/0

            #npos = fn_e[0].find('e0355')
            #fn_b = fn_e[0][0:npos]+'b0355'+fn_e[0][npos+5:]
            
            #fn_e = 'EARLINET_ECVT_AerRemSen_hpb_Lev01_e0355_202408100000_202408100100_v01_qc03.nc'
            #fn_b = 'EARLINET_ECVT_AerRemSen_hpb_Lev01_b0355_202408100000_202408100100_v01_qc03.nc'

            z1 = 0
            z2 = 20 

            print('number of 0355 file = ', len(fn_e_all)) 
            for fn_e in fn_e_all:

                print('e0355 file =  ', os.path.basename(fn_e))
                npos = fn_e.find('e0355')
                fn_b = fn_e[0:npos]+'b0355'+fn_e[npos+5:]
                fn_b_all.append(fn_b)

                data = read_ecvt_earlinet_file(fn_e)
                if 'extinction' and 'backscatter' in data.keys():

                    if os.path.isfile(fn_b):  
                        data_b = read_ecvt_earlinet_file(fn_b)
                    else:
                        data_b = {}
                        print('## earlinet e0355 exist but no b0355 data, no depol: ', site, ' ' , date)
                        # continue
                        #1/0
                else:
                    print( '!!! extinction or backscatter are not in earlinet file, skip the file', fn_e)
                    continue


                if plot_earlinet == 1:
         
                    plot_earlinet_figures(data, data_b, fn_e, fn_b, z1, z2)  


                # read collocated file to get lat/lon

                # get lat/lon from earlinet file
                lat = data['latitude']
                lon = data['longitude']
                time_bounds = data['time_bounds']

                #t0 = datetime.fromtimestamp(time_bounds[0])  # [1]  local time
                #t1 = datetime.fromtimestamp(time_bounds[1])  # [1]
                
                t0 = datetime.datetime.fromtimestamp(time_bounds[0], datetime.UTC)
                t1 = datetime.datetime.fromtimestamp(time_bounds[1], datetime.UTC)
                
                # file frame using the lat of the station.
                frame_id = find_frame(lat)

                # look for date and frame in ATL ZIP file 

                # extracted h5 files

                #date_test = '20240815'  #should be the same as date for earlinet but for test here
                if date >= '20250101':  #'20240801':    
                    dir_ec = '/net/pc230016/nobackup_1/users/zadelhof/EarthCARE_DATA/L2/' 
                else:
                    dir_ec = None

                #dir_ec = '/net/pc230016/nobackup_1/users/zadelhof/EarthCARE_DATA/L2/' 
                
                dt=1.5    # 1   4
                dist_min = 100  # km
                ec_product1 = 'ATL_EBD_2A'
                dir_ec_local1 = '/nobackup/users/wangp/Earthcare/data/L2/{0}/{1}/'.format(ec_product1, date[0:4]) 

                collocated_fn_ebd = find_atl_l2_collocated_files(ec_product1, date, frame_id, t0, dt, dir_ec_local1, dir_ec=dir_ec) # dir_ec = None)

                ec_product2 = 'ATL_AER_2A'
                dir_ec_local2 = '/nobackup/users/wangp/Earthcare/data/L2/{0}/{1}/'.format(ec_product2, date[0:4]) 
                collocated_fn_aer = find_atl_l2_collocated_files(ec_product2, date, frame_id, t0, dt, dir_ec_local2, dir_ec=dir_ec) # dir_ec = None)

                

                # read AEL_EBD_2A file  #ATL_AER_2A 
                print()
                print('number of collocated file EBD files in dt = ', len(collocated_fn_ebd) )
                if  len(collocated_fn_ebd)  == 0:
                    print('No EBD file, skip site ', site)                
                    continue


                if cmp_single == 1: 
                    # only compare EBD file 
                    for fn1 in  collocated_fn_ebd:

                        e1, e2 = cmp_ebd(fn1, data, data_b, dist_min, site, dir_figure)
                        
                        if e1 != 0: 
                            ebd_data = np.copy(e1)
                            collocate_info_ebd = np.copy(e2)
                            continue 

                    # only compare AER file 
                    for fn_a in  collocated_fn_aer:

                        a1, a2  = cmp_aer(fn_a, data, data_b, dist_min, site, dir_figure)
                        if a1 != 0: 
                            aer_data = np.copy(a1)
                            collocate_info_aer = np.copy(a2)
                            continue

                
                # use AER classification to select EBD
                
                for fn_a in  collocated_fn_aer:

                    a1=0
                    a2=0
                    a3=0
                    a4=0 
                    data_pickle = 0
                    
                    # look for ebd data
                    #fn1 = glob.glob(dir_ec_local1+'**/*EBD*'+ fn_a[-10:], recursive=True)[0]
                    fn1 = glob.glob(dir_ec_local1+'**/'+ os.path.basename(fn_a)[0:13]+'EBD*'+ fn_a[-10:], recursive=True)  # has to be the same version Difference version may have difference number of measurements. eg. EXAA and EXAC 01985D
                    if len(fn1) == 0:
                        print('AER, EBD are different version, skip this file')
                        continue
                    else:
                        fn1 = fn1[0]
                    
                    #print('### compare data for site, date, fn1, fn_a :', site, date, fn1, fn_a)  
                    #print()
                    a1, a2, a3, a4  = cmp_ebd_aer_filter(fn1, fn_a, data, data_b, dist_min, site, dir_figure)
                    if a1 != 0 : 
                        print('### compare data for site, date, fn1, fn_a :', site, date, os.path.basename(fn1), os.path.basename(fn_a))  
                        print()
                        ec_grd_data[date+'_'+site] = a2
                        stat_data[date+'_'+site] = a1
                        grd_data[date+'_'+site] = a3
                        nsite = nsite + 1
                        data_pickle = {'ec_ground_collocated_data':ec_grd_data, 'stat_data': stat_data, 'ground_site_data':grd_data, 
                                'ebd_file':fn1, 'aer_file':fn_a, 'ealinet_e0355':fn_e, 'ealinet_b0355':fn_b} 
                       
                        #fn_pickle = dir_figure + 'ec_ground_collocated_{0}_site_{1}_P{2}_e{3}.pickle'.format(os.path.basename(fn_a)[20:35], site, a4, t0.strftime('%H%M%S'))  # some sites may have no data
                        fn_pickle = dir_figure + '{0}_ground_collocated_{1}_site_{2}_P{3}_e{4}.pickle'.format(os.path.basename(fn_a)[0:8],os.path.basename(fn_a)[20:35], site, a4, t0.strftime('%H%M%S'))  # some sites may have no data

                        #if os.path.isfile(fn_pickle):
                        #    fn_pickle =   dir_figure + 'ec_ground_collocated_{0}_site_{1}_P{2}_e{3}a.pickle'.format(date,site, a4, t0.strftime('%H%M%S'))     
             
                        pickle_out = open(fn_pickle, 'wb')
                        pickle.dump(data_pickle,pickle_out, protocol=2)

                        fn_figure = fn_pickle[0:-7]
                        if len(a1) != 0:
                            print('mean' , a1['ext_grd_mean'], a1['ext_ec_a_mean'] ) 
                            plot_ext_backscatter_means(data_pickle, fn_figure)
                        
                        
             
                        

        # # pickle data per day 
        # nx = 0  
        # for x in stat_data.keys():
            # if len( stat_data[x]) ==  0:
                # print('no data in stat_data.keys() ', x)
            # else:
                # nx = nx + 1

        # print('number of data in  stat_data.keys()', nx)
  
        # if nx > 0: 
            # data = {'ec_ground_collocated_data':ec_grd_data, 'stat_data': stat_data, 'ground_site_data':grd_data}
            # fn_pickle = dir_figure + 'ec_ground_collocated_{0}_sites_{1}.pickle'.format(date,nx)  #nsite)  # some sites may have no data
            # if os.path.isfile(fn_pickle):
                # fn_pickle =   dir_figure + 'ec_ground_collocated_{0}_sites_{1}_a.pickle'.format(date,nx)     
 
            # pickle_out = open(fn_pickle, 'wb')
            # pickle.dump(data,pickle_out, protocol=2)

            # fn_figure = fn_pickle[0:-7]
            # plot_ext_backscatter_means(data, fn_figure)

    
   #         1/0
           # plt.plot(

        # else:
            # print('returned data empty from cmp_ebd_aer_filter on ', date)


        #1/0 

    1/0


   
 # # Check for the last possible file using the creation time
           # file_in=dirs+"ECA_EXAA_ATL_FM*/*.h5"
           # files=glob.glob(file_in)
           # files.sort(key=os.path.getctime)


# 
