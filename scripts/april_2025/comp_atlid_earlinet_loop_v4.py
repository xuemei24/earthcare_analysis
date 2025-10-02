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

import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature



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
              plot_ecvt_atl_depol_profile_fill, \
              find_atl_l2_collocated_files_downthemall
 


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

        title = 'AER ' + os.path.basename(fn_a)[-9:-3] + ' ' + ec_st2 + ' '+str(sel)+ '; ' + site + ' ' + t0.strftime('%H:%M:%S') +  ' D' + str(dist)[0:4]          


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

        title0 = 'EBD ' + os.path.basename(fn1)[-9:-3] + ' ' + ec_st2 + ' '+str(sel)+'; ' + site + ' ' +t0.strftime('%H:%M:%S')+ ' D ' + str(dist)[0:4]          
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


def cmp_ebd_aer_filter_mean(fn1, fn_a, data, data_b, dist_min, site, dir_figure,z1=0, z2=20):
    # read ebd, aer data, select ebd using aer classification.  plot ebd, aer

    #z1 = 0
    #z2 = 10 
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
        dd1[i] = geodesic(p1,p2).km

    sel1 = np.argmin(dd1)   # pixel with smallest distance to a stie
    dist1 = dd1[sel1]       # the smallest distance 

    sel2x = np.where(dd1 < dist_min)    # all pixels within 100 km to a station
    #sel2 = np.where(dd1 < 100.)    # all pixels within 100 km to a station
        
    print('closest distance in lat/lon: ', dist1, 'pixel: ', sel1)

    #print('closest distance in lat/lon old: ', dist, dd[sel],  'pixel: ', sel)

    sel = np.copy(sel1)
    dist = np.copy(dist1)

    #sel = np.copy(sel2)
    #dist = np.copy(dist1)

    
    if dist < dist_min:  #200: 

        # test select maximum 100 pixels for average
        print('sel2x', sel2x, len(sel2x[0]), sel1)
  
        idx = np.where(sel2x[0] == sel1)
        print('idx', idx)
        
        sel2 = np.zeros(100)   # 100 pixel ~ 100 km
 
        if len(sel2x[0]) > 100:
            sel2 = np.copy(sel2x[0][idx[0][0]-50:idx[0][0]+50])
        else:
            sel2 = np.copy(sel2x[0][:])
        print('sel2=', sel2)
       

        # read EBD
    
        #sel = np.copy(sel2)
        #dist = np.copy(dist1)
        #print('sel2', sel2, len(sel2[0]))

                
        p1a = (lat_a[sel2[0]], lon_a[sel2[0]])
        p2a = (lat_a[sel2[-1]], lon_a[sel2[-1]])

        dist_track_a = geodesic(p1a,p2a).km
        print('along track dist, closest dist, npixel', dist_track_a, dist1, len(sel2))
        print(dd1, sel1)
       
  
        fig = plt.figure('test', figsize=(11,8.5))      
        ax=plt.axes(projection=ccrs.PlateCarree())   #ccrs.Mercator()
        #ax=plt.axes(projection=ccrs.Mercator())   #ccrs.Mercator()
        
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.LAND, edgecolor='black')

        #ax.coastlines()
        ax.plot(lon_a[sel2], lat_a[sel2], '.')
        ax.plot([lon_a[sel2][0],lon, lon_a[sel2][-1]], [lat_a[sel2][0], lat, lat_a[sel2][-1] ])
        ax.plot([lon, lon_a[sel1]], [lat, lat_a[sel1] ], 'k*')
    
        ax.set_xticks(np.arange(lon-3,lon+3,1), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(lat-3,lat+3,1), crs=ccrs.PlateCarree())

        plt.ylabel('Latitude [N$^{\circ}$]')
        plt.xlabel('Longitude [E$^{\circ}$]')
        title = 'dist {0:.2f} track {1:.2f}'.format( dist1, dist_track_a)
        plt.title(title)
        plt.text(lon, lat, site, horizontalalignment='right')
        filename_fig = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_lat_lon_dist_'+ site+ '_P'+str(sel)+'_e'+t0.strftime('%H%M%S')+'.png'
        plt.savefig(filename_fig)
        plt.close(fig)

        #plt.text(lon, lat, 'cbw', horizontalalignment='right'),
         # transform=ccrs.Geodetic())

        # plt.plot([lon, lon_a[sel1]], [lat, lat_a[sel1] ],
         # color='red', linestyle='--',
         # transform=ccrs.PlateCarree(),
         # )

        # plt.plot([lon, lon_a[sel1]], [lat, lat_a[sel1] ],
         # color='blue', linestyle=':',
         # transform=ccrs.Geodetic(),
         # )

        #plt.show()


#        1/0
        ebd_data = read_atlid_l2(fn1)

        plot_aer = 1
  
        if plot_aer == 1:
            ec_st1, ec_st2 = ec_time_to_datetime_str(aer_data['time'][sel])   #closest

            # print('closest distance in lat/lon: ', dist, dd[sel],  'pixel: ', sel)

            # title = 'AER ' + os.path.basename(fn_a)[-9:-3] + ' ' + ec_st2 + ' '+str(sel)+ '; ' + site + ' ' + t0.strftime('%H:%M:%S') +  ' D' + str(dist)[0:4] 
            title = os.path.basename(fn_a)[4:8] + ' ' + os.path.basename(fn_a)[-9:-3] + ' ' + ec_st2 + ' '+str(sel)+ '; ' + site + ' ' + t0.strftime('%H:%M:%S') +  ' D' + str(dist)[0:4] 


            sel_qs =  np.where(aer_data['quality_status'][:,:] > 1)
            aer_tmp = np.copy( aer_data['particle_extinction_coefficient_355nm'])
            aer_tmp[sel_qs] = np.nan     

            aer_err_tmp = np.copy( aer_data['particle_extinction_coefficient_355nm_error'])
            aer_err_tmp[sel_qs] = np.nan     

            
            ext_ec_a = np.copy(aer_tmp[sel,:])
            ext_ec_err_a = np.copy(aer_err_tmp[sel,:])

            height_ec_a =  aer_data['height'][sel,:]#[sel_cls3]#[10,:] 
            classification = aer_data['classification'][sel,:]

            
            #outfile_ec_a = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_ext_'+ site+ '_P'+str(sel)+'_e'+t0.strftime('%H%M%S')+'.png'
            outfile_ec_a = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_ext_'+ site+ '_P'+str(sel)+'_e'+t0.strftime('%H%M%S')+'F.png'
                    
            #if os.path.isfile(outfile_ec_a):
            #    print('ext figure exsit, skip ', outfile_ec_a )
            #    #return 0, 0, 0
            
            #print('ext_ec_a ', ext_ec_a.shape, height_ec_a.shape)

            plot_ecvt_atl_ext_profile_fill(data['extinction'], data['error_extinction'], data['altitude'], z1, z2, outfile=outfile_ec_a, 
                                  ext_ec=ext_ec_a, ext_ec_err=ext_ec_err_a, height_ec=height_ec_a, title=title, classification = None)

            aer_tmp = np.copy( aer_data['particle_backscatter_coefficient_355nm'])
            aer_tmp[sel_qs] = np.nan     

            aer_err_tmp = np.copy( aer_data['particle_backscatter_coefficient_355nm_error'])
            aer_err_tmp[sel_qs] = np.nan     


            backscatter_ec_a = np.copy(aer_tmp[sel,:])
            backscatter_ec_err_a = np.copy(aer_err_tmp[sel,:])
            height_ec_a =  aer_data['height'][sel,:] 
            outfile_ec_a = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_backscatter_'+ site+ '_'+str(sel)+'_e'+t0.strftime('%H%M%S')+'F.png'

            plot_ecvt_atl_backscatter_profile_fill(data['backscatter'], data['error_backscatter'], data['altitude'], z1, z2, outfile=outfile_ec_a, 
                                  backscatter_ec=backscatter_ec_a, backscatter_ec_err=backscatter_ec_err_a, height_ec=height_ec_a,title=title, classification = classification)


            aer_tmp = np.copy( aer_data['lidar_ratio_355nm'])
            aer_tmp[sel_qs] = np.nan     

            aer_err_tmp = np.copy( aer_data['lidar_ratio_355nm_error'])
            aer_err_tmp[sel_qs] = np.nan     

            lidarratio_ec_a = np.copy(aer_tmp[sel,:])
            lidarratio_ec_err_a = np.copy(aer_err_tmp[sel,:])
            height_ec_a =  aer_data['height'][sel,:] 

            outfile_ec_a = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_lidarratio_'+ site+ '_'+str(sel)+'_e'+t0.strftime('%H%M%S')+'F.png'


            plot_ecvt_atl_lidarratio_profile_fill(data['lidarratio'], data['error_lidarratio'], data['altitude'], z1, z2, outfile=outfile_ec_a, 
                                  lidarratio_ec=lidarratio_ec_a, lidarratio_ec_err=lidarratio_ec_err_a, height_ec=height_ec_a,title=title, classification = classification)

            aer_tmp = np.copy( aer_data['particle_linear_depol_ratio_355nm'])
            aer_tmp[sel_qs] = np.nan     

            aer_err_tmp = np.copy( aer_data['particle_linear_depol_ratio_355nm_error'])
            aer_err_tmp[sel_qs] = np.nan     


            #depol_ec_a = aer_data['particle_linear_depol_ratio_355nm'][sel,:]
            #depol_ec_err_a = aer_data['particle_linear_depol_ratio_355nm_error'][sel,:]#[10,:]
            depol_ec_a = np.copy(aer_tmp[sel,:])
            depol_ec_err_a = np.copy(aer_err_tmp[sel,:])

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

            outfile_ec_a = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_depol_'+ site+'_'+str(sel)+ '_e'+t0.strftime('%H%M%S')+'F.png'

            plot_ecvt_atl_depol_profile_fill(depol_earlinet, depol_earlinet_err, data_b['altitude'], z1, z2, outfile=outfile_ec_a, 
                                          depol_ec=depol_ec_a, depol_ec_err=depol_ec_err_a, height_ec=height_ec_a,title=title, classification = classification)

            outfile_ec_a = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_classification_'+ site+'_e'+t0.strftime('%H%M%S')+ '.png'

            plot_aer_classification(aer_data, data, sel, z1, z2, outfile=outfile_ec_a, title=title)


            stat_data_grd = {}   # ground-based
            stat_data_ec = {}    # earthcare 

            stat_data_ec['distance'] =  dist1
            stat_data_ec['track_length'] = dist_track_a

            stat_data_ec['ext_ec_a'] =  ext_ec_a
            stat_data_ec['ext_ec_err_a'] =  ext_ec_err_a
            
            stat_data_ec['height_ec_a_org'] =  np.copy(height_ec_a)
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
            h_ec = np.copy(height_ec_a)   #[::-1])
            #sel_h = np.where(((h_ec >= min(data['altitude'])) & (h_ec <= max(data['altitude']))) )
            sel_h = np.where(((h_ec < min(data['altitude'])) | (h_ec > max(data['altitude']))) )
           # print('sel_h =', sel_h)
           # print('h_ec[sel_h] =', h_ec[sel_h])

            stat_data_ec['height_ec_a'] =  np.copy(h_ec)
            for var in stat_data_grd.keys():
                y = np.interp( h_ec[::-1], data['altitude'][:],  stat_data_grd[var][:])    # original earthcare grid
                yvar = np.copy(y)
                sel36 = np.where(yvar > 1.e6)
                if len(sel36[0]) > 0:
                     yvar[sel36[0]] = np.nan
                yvar[::-1][sel_h[0]] = np.nan   # remove interpolated data at earthcare grid for the grids above earlinet top and below earlinet ground

                stat_data_ec[var] = np.copy(yvar[::-1]) 

            stat_data_grd['depol_grd'] = depol_earlinet  #data_b['particledepolarization']
            stat_data_grd['depol_grd_err'] = depol_earlinet_err # data_b['error_particledepolarization']

            stat_data_grd['height_b_grd'] =  data_b['altitude']
            stat_data_grd['height_grd'] =  data['altitude']

            if len(data_b) > 10:    # choose ground-based profile that has more data points
                #y = np.interp( height_ec_a[::-1], data_b['altitude'][:],  stat_data_grd['depol_grd'][:])
                y = np.interp( h_ec[::-1], data_b['altitude'][:],  stat_data_grd['depol_grd'][:])
                sel36 = np.where(y > 1.e6)   # remove filled value 1e36
                if len(sel36[0]) > 0:
                    y[sel36[0]] = np.nan
                y[::-1][sel_h[0]] = np.nan 
                stat_data_ec['depol_grd'] = np.copy(y[::-1]) 

                y = np.interp( h_ec[::-1], data_b['altitude'][:],  stat_data_grd['depol_grd_err'][:])
                if len(sel36[0]) > 0:   # same filter as depol
                    y[sel36[0]] = np.nan 
                y[::-1][sel_h[0]] = np.nan 

                stat_data_ec['depol_grd_err'] = np.copy(y[::-1]) 
            else:
                stat_data_ec['depol_grd'] =  stat_data_grd['depol_grd'][:]     # filled with 0, no need to interpolate
                stat_data_ec['depol_grd_err'] = stat_data_grd['depol_grd_err'][:]

        # EBD  ########### mean 
 
       

        ec_st1, ec_st2 = ec_time_to_datetime_str(ebd_data['time'][sel])

        # title0 = 'EBD ' + os.path.basename(fn1)[-9:-3] + ' ' + ec_st2 + ' '+str(sel)+'; ' + site + ' ' +t0.strftime('%H:%M:%S')+ ' D ' + str(dist)[0:4]          
        title0 = os.path.basename(fn_a)[4:8] + ' ' + os.path.basename(fn1)[-9:-3] + ' ' + ec_st2 + ' '+str(sel)+'; ' + site + ' ' +t0.strftime('%H:%M:%S')+ ' D ' + str(dist)[0:4]          

        ext_var_list= ['particle_extinction_coefficient_355nm', 
                       'particle_extinction_coefficient_355nm_medium_resolution',
                       'particle_extinction_coefficient_355nm_low_resolution']

        #sel2_cls10 = np.where(aer_data['classification'][:,:] < 10)
        #sel2_qs = np.where(ebd_data['quality_status'][:,:] == 0)
        sel2_qs_aer = np.logical_or( aer_data['classification'][:,:] < 10, ebd_data['quality_status'][:,:] > 0)


        height_ec =  ebd_data['height'][sel,:]#[10,:] 
        stat_data_ec['height_ec'] = height_ec
        for ext_var in ext_var_list:

            # use mean mext_ec for mean ext_ec
            ebd_tmp = np.copy(ebd_data[ext_var])
            ebd_tmp[sel2_qs_aer] = np.nan        # good, use sel2_qs_aer for logical_*, not sel2_qs_aer[0]

            ebd_err_tmp = np.copy(ebd_data[ext_var+'_error'])
            ebd_err_tmp[sel2_qs_aer] = np.nan

            seln = np.isnan(ebd_tmp[sel2,:])        # true is 1
            ntot = len(sel2) - np.sum(seln, axis=0)
            print('ntot=', ntot)

            mext_ec =  np.nanmean(ebd_tmp[sel2,:],axis=0).flatten()
            mext_ec_err =  np.sqrt(np.nansum(ebd_err_tmp[sel2,:]*ebd_err_tmp[sel2,:], axis = 0)).flatten()/ntot  #len(sel2[0])
            #mext_ec_err1 =  np.nanmean(ebd_err_tmp[sel2[0],:], axis = 0).flatten()

            mext_ec1 = np.copy(mext_ec)
            sel_ntot = np.where(ntot < max(ntot)*0.1)   # 10% of max(ntot)  ntot = 100, then 10 is used., remove layers with few aerosol bins, so only use relatively homogeneous layers 
            mext_ec1[sel_ntot] = np.nan
            mext_ec_err1 = np.copy(mext_ec_err)
            mext_ec_err1[sel_ntot] = np.nan

            #print('mext_ec, _err1', mext_ec1, mext_ec_err1)

            #print(mext_ec.shape)

            #ext_ec = ebd_data[ext_var][sel,:]
            #ext_ec_err = ebd_data[ext_var+'_error'][sel,:]  
            # filtered clouds and bad data
            ext_ec = np.copy(ebd_tmp[sel,:])
            ext_ec_err = np.copy(ebd_err_tmp[sel,:]) 
            

            if  ext_var.find('low') >=0:
                title = title0 + ' L'
            elif    ext_var.find('medium') >=0:
                title = title0 + ' M'
            else:
                title = title0
 
            stat_data_ec[ext_var] = ext_ec
            stat_data_ec[ext_var+'_error'] = ext_ec_err

            stat_data_ec[ext_var+'_M'] = mext_ec
            stat_data_ec[ext_var+'_error_M'] = mext_ec_err

            stat_data_ec[ext_var + '_ntot_M'] = ntot

           
            classification = aer_data['classification'][sel,:]

            outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_' + ext_var +'_'+ site+ '_'+str(sel)+'_e'+t0.strftime('%H%M%S')+'.png'
        #txt = os.path.basename(outfile)[13:36] + ' ' + os.path.basename(outfile)[54:60] +  ' ' +   os.path.basename(outfile)[-7:-4]      
            
            plot_ecvt_atl_ext_profile_fill(data['extinction'], data['error_extinction'], data['altitude'], z1, z2, outfile=outfile_ec, 
                        ext_ec=ext_ec, ext_ec_err=ext_ec_err, height_ec=height_ec,title=title, classification = classification)

            outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_' + ext_var +'_'+ site+ '_'+str(sel)+'_e'+t0.strftime('%H%M%S')+'_M.png'
            plot_ecvt_atl_ext_profile_fill(data['extinction'], data['error_extinction'], data['altitude'], z1, z2, outfile=outfile_ec, 
                        ext_ec=mext_ec1, ext_ec_err=mext_ec_err1, height_ec=height_ec,title=title, classification = classification)


            
        bks_var_list= ['particle_backscatter_coefficient_355nm', 
                       'particle_backscatter_coefficient_355nm_medium_resolution',
                       'particle_backscatter_coefficient_355nm_low_resolution']
   
        height_ec =  ebd_data['height'][sel,:]#[10,:] 
        for bks_var in bks_var_list: 

     # use mean mext_ec for mean ext_ec
            ebd_tmp = np.copy(ebd_data[bks_var])
            ebd_tmp[sel2_qs_aer] = np.nan

            ebd_err_tmp = np.copy(ebd_data[bks_var+'_error'])
            ebd_err_tmp[sel2_qs_aer] = np.nan

            seln = np.isnan(ebd_tmp[sel2,:])        # true is 1
            ntot = len(sel2) - np.sum(seln, axis=0)
            #print('ntot=', ntot)

            mbackscatter_ec =  np.nanmean(ebd_tmp[sel2,:],axis=0).flatten()
            mbackscatter_ec_err =  np.sqrt(np.nansum(ebd_err_tmp[sel2,:]*ebd_err_tmp[sel2,:], axis = 0)).flatten()/ntot  #len(sel2[0])
            

            #backscatter_ec = ebd_data[bks_var][sel,:]
            #backscatter_ec_err = ebd_data[bks_var+'_error'][sel,:] 
            
            backscatter_ec = np.copy(ebd_tmp[sel,:])
            backscatter_ec_err = np.copy(ebd_err_tmp[sel,:])

   
            outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_'+bks_var + '_'+ site+ '_'+str(sel)+'_e'+t0.strftime('%H%M%S')+'.png'
            if bks_var.find('low') >=0:
                title = title0 + ' L'
            elif bks_var.find('medium') >=0:
                title = title0 + ' M'
            else:
                title = title0

            stat_data_ec[bks_var] = backscatter_ec
            stat_data_ec[bks_var + '_error'] = backscatter_ec_err

            stat_data_ec[bks_var+'_M'] = mbackscatter_ec
            stat_data_ec[bks_var + '_error_M'] = mbackscatter_ec_err

            stat_data_ec[bks_var + '_ntot_M'] = ntot


            plot_ecvt_atl_backscatter_profile_fill(data['backscatter'], data['error_backscatter'], data['altitude'], z1, z2, outfile=outfile_ec, 
                                  backscatter_ec=backscatter_ec, backscatter_ec_err=backscatter_ec_err, height_ec=height_ec,title=title, classification = classification)

            outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_'+bks_var + '_'+ site+ '_'+str(sel)+'_e'+t0.strftime('%H%M%S')+'_M.png'
            plot_ecvt_atl_backscatter_profile_fill(data['backscatter'], data['error_backscatter'], data['altitude'], z1, z2, outfile=outfile_ec, 
                                  backscatter_ec=mbackscatter_ec, backscatter_ec_err=mbackscatter_ec_err, height_ec=height_ec,title=title, classification = classification)

        #stat_data_ec['ntot'] = ntot
        lidarratio_var_list = ['lidar_ratio_355nm',
                               'lidar_ratio_355nm_medium_resolution',
                               'lidar_ratio_355nm_low_resolution']

        for lidarratio_var in lidarratio_var_list:

            ebd_tmp = np.copy(ebd_data[lidarratio_var])
            ebd_tmp[sel2_qs_aer] = np.nan

            ebd_err_tmp = np.copy(ebd_data[lidarratio_var+'_error'])
            ebd_err_tmp[sel2_qs_aer] = np.nan
 
            sel_s = np.where(abs(ebd_err_tmp/ebd_tmp) > 5)       # remove large errors
            ebd_err_tmp[sel_s] = np.nan
            ebd_tmp[sel_s] = np.nan

            seln_lr = np.isnan(ebd_tmp[sel2,:])        # true is 1
            ntot_lr = len(sel2) - np.sum(seln_lr, axis=0)
            #print('ntot_lr', ntot_lr)

            mlidarratio_ec =  np.nanmean(ebd_tmp[sel2,:],axis=0).flatten()
            mlidarratio_ec_err =  np.sqrt(np.nansum(ebd_err_tmp[sel2,:]*ebd_err_tmp[sel2,:], axis = 0)).flatten()/ntot_lr  #len(sel2[0])

            #lidarratio_ec = ebd_data[lidarratio_var][sel,:]       
            #lidarratio_ec_err = ebd_data[lidarratio_var + '_error'][sel,:]#[10,:]

            lidarratio_ec = np.copy(ebd_tmp[sel,:])
            lidarratio_ec_err = np.copy(ebd_err_tmp[sel,:])
            height_ec =  ebd_data['height'][sel,:] 

            outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_' + lidarratio_var +'_'+ site+ '_'+str(sel)+'_e'+t0.strftime('%H%M%S')+'.png'

            if lidarratio_var.find('low') >=0:
                title = title0 + ' L'
            elif lidarratio_var.find('medium') >=0:
                title = title0 + ' M'
            else:
                title = title0

            stat_data_ec[lidarratio_var] = lidarratio_ec 
            stat_data_ec[lidarratio_var + '_error'] = lidarratio_ec_err

            stat_data_ec[lidarratio_var +'_M'] = mlidarratio_ec 
            stat_data_ec[lidarratio_var + '_error_M'] = mlidarratio_ec_err

            stat_data_ec[lidarratio_var +'_ntot_M'] = ntot_lr

            plot_ecvt_atl_lidarratio_profile_fill(data['lidarratio'], data['error_lidarratio'], data['altitude'], z1, z2, outfile=outfile_ec, 
                              lidarratio_ec=lidarratio_ec, lidarratio_ec_err=lidarratio_ec_err, height_ec=height_ec,title=title, classification = classification)

            outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_' + lidarratio_var +'_'+ site+ '_'+str(sel)+'_e'+t0.strftime('%H%M%S')+'_M.png'
            plot_ecvt_atl_lidarratio_profile_fill(data['lidarratio'], data['error_lidarratio'], data['altitude'], z1, z2, outfile=outfile_ec, 
                              lidarratio_ec=mlidarratio_ec, lidarratio_ec_err=mlidarratio_ec_err, height_ec=height_ec,title=title, classification = classification)


        #stat_data_ec['ntot_lr'] = ntot_lr


        depol_var_list = ['particle_linear_depol_ratio_355nm',
                          'particle_linear_depol_ratio_355nm_medium_resolution',
                          'particle_linear_depol_ratio_355nm_low_resolution']
        for depol_var in depol_var_list:
            

            ebd_tmp = np.copy(ebd_data[depol_var])
            ebd_tmp[sel2_qs_aer] = np.nan

            ebd_err_tmp = np.copy(ebd_data[depol_var+'_error'])
            ebd_err_tmp[sel2_qs_aer] = np.nan
 
            sel_s = np.where(abs(ebd_err_tmp/ebd_tmp) > 10)       # remove large errors
            ebd_err_tmp[sel_s] = np.nan
            ebd_tmp[sel_s] = np.nan

            sel_s = np.where(abs(ebd_tmp) > 1.0)       # remove large errors
            ebd_err_tmp[sel_s] = np.nan
            ebd_tmp[sel_s] = np.nan


            seln_dp = np.isnan(ebd_tmp[sel2,:])        # true is 1
            ntot_dp = len(sel2) - np.sum(seln_dp, axis=0)
            #print('ntot_dp', ntot_dp)

            mdepol_ec =  np.nanmean(ebd_tmp[sel2,:],axis=0).flatten()
            mdepol_ec_err =  np.sqrt(np.nansum(ebd_err_tmp[sel2,:]*ebd_err_tmp[sel2,:], axis = 0)).flatten()/ntot_dp  #len(sel2[0])

            #depol_ec = ebd_data[depol_var][sel,:]
            #depol_ec_err = ebd_data[depol_var + '_error'][sel,:]

            depol_ec = np.copy(ebd_tmp[sel,:])
            depol_ec_err = np.copy(ebd_err_tmp[sel,:])
            height_ec =  ebd_data['height'][sel,:]



            outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_'+ depol_var + '_'+ site+ '_'+str(sel)+ '_e'+t0.strftime('%H%M%S')+'.png'

            if depol_var.find('low') >=0:
                title = title0 + ' L'
            elif depol_var.find('medium') >=0:
                title = title0 + ' M'
            else:
                title = title0

            stat_data_ec[depol_var] = depol_ec 
            stat_data_ec[depol_var + '_error'] = depol_ec_err

            stat_data_ec[depol_var+'_M'] = mdepol_ec 
            stat_data_ec[depol_var + '_error_M'] = mdepol_ec_err

            stat_data_ec[depol_var+'_ntot_M'] = ntot_dp

            plot_ecvt_atl_depol_profile_fill(depol_earlinet, depol_earlinet_err, data_b['altitude'], z1, z2, outfile=outfile_ec, 
                              depol_ec=depol_ec, depol_ec_err=depol_ec_err, height_ec=height_ec,title=title, classification = classification)

            outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_'+ depol_var + '_'+ site+ '_'+str(sel)+ '_e'+t0.strftime('%H%M%S')+'_M.png'
            plot_ecvt_atl_depol_profile_fill(depol_earlinet, depol_earlinet_err, data_b['altitude'], z1, z2, outfile=outfile_ec, 
                              depol_ec=mdepol_ec, depol_ec_err=mdepol_ec_err, height_ec=height_ec,title=title, classification = classification)

        #stat_data_ec['ntot_depol'] = ntot_dp

        #1/0 
        #  ############## change from here #################
        classification_short = np.copy(classification)
        print(classification_short)
        
        #sel10 = np.where(classification > 9.9)   # select aerosols, also for ground-based data        #############select part of the profile with overlap height of ground-based.
                     
        classification_short[sel_h[0]] = -4                                                      # otherwise the ground-based profile is not correcy
        
        sel10 = np.where(classification_short > 9.9)   # sele
        
        stat_data1 = {}

        if len(sel10[0]) > 1:

            sel_nan = np.where(np.isnan(stat_data_ec['particle_extinction_coefficient_355nm']) == True) 
            print(stat_data_ec.keys())
            for var in stat_data_ec.keys():
                # x = np.copy(stat_data_ec[var])                  
                # if var.find('grd') > 0:
                    # print('var', x[var])
                        
                    # x[var][sel_nan[0]] = np.nan
                    # print('var', x[var])
                # 1/0 
                try : 
                    x = np.copy(stat_data_ec[var]) 
                    # if var.find('grd') > 0:
                    #print('var', var,  x)
                            
                        # #x[sel_nan[0]] = np.nan
                        # #print('var nan', x)
          
 
                    #x[0:150] = np.nan    # remove data above ~9 km 
                    sel36 = np.logical_and( x[sel10[0]] > -1.e-6,  x[sel10[0]] < 200)   # remove 1e36 filled values in earlinet
                    print('var', var)
                    stat_data1[var+'_mean'] = np.nanmean(x[sel10[0]][sel36])    # sel36[0] does not work for logical_add
                    stat_data1[var+'_std'] = np.nanstd(x[sel10[0]][sel36])
                    
                except Exception as error:
                    print('ext var=', var)
                    print('exception error ', error)
                    stat_data1[var] = np.copy(x)

        return stat_data1, stat_data_ec, stat_data_grd, sel  # mean, std; earthcare data; ground-based data

    else:
        #print('!! No data within dt 1.5 h and 200 km site, date', site, date)
        return 0, 0, 0, 0


#def cmp_ebd_aer_atc_filter_mean(fn1, fn_a, fn_atc, data, data_b, dist_min, site, dir_figure, z1=0, z2=20):
def cmp_ebd_aer_atc_filter_mean(data, data_b, dist_min, site, dir_figure, fn1=None, fn_a=None, fn_atc=None, z1=0, z2=20):

    # read ebd, aer data, select ebd using aer classification.  plot ebd, aer

    #z1 = 0
    #z2 = 10 
    lat = data['latitude']
    lon = data['longitude']
    time_bounds = data['time_bounds']

    #t0 = datetime.utcfromtimestamp(time_bounds[0])
    t0 = datetime.datetime.fromtimestamp(time_bounds[0], datetime.UTC)

    if fn_a is not None:
        aer_data = read_atlid_l2(fn_a)

        # look for collocated ec product
        lat_a =  aer_data['latitude']
        lon_a =  aer_data['longitude']
    
    if fn1 is not None:    
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
        dd1[i] = geodesic(p1,p2).km

    sel1 = np.argmin(dd1)   # pixel with smallest distance to a stie
    dist1 = dd1[sel1]       # the smallest distance 

    sel2x = np.where(dd1 < dist_min)    # all pixels within 100 km to a station
    #sel2 = np.where(dd1 < 100.)    # all pixels within 100 km to a station
        
    print('closest distance in lat/lon: ', dist1, 'pixel: ', sel1)

    #print('closest distance in lat/lon old: ', dist, dd[sel],  'pixel: ', sel)

    sel = np.copy(sel1)
    dist = np.copy(dist1)

    #sel = np.copy(sel2)
    #dist = np.copy(dist1)

    
    if dist < dist_min:  #200: 

        # test select maximum 100 pixels for average
        print('sel2x', sel2x, len(sel2x[0]), sel1)
  
        idx = np.where(sel2x[0] == sel1)
        print('idx', idx)
        
        sel2 = np.zeros(100)   # 100 pixel ~ 100 km
 
        if len(sel2x[0]) > 100:    #60 km or 100 km
            sel2 = np.copy(sel2x[0][idx[0][0]-50:idx[0][0]+50])    # 50 or 30
        else:
            sel2 = np.copy(sel2x[0][:])
        print('sel2=', sel2)
       
 
        #sel = np.copy(sel2)
        #dist = np.copy(dist1)
        #print('sel2', sel2, len(sel2[0]))

                
        p1a = (lat_a[sel2[0]], lon_a[sel2[0]])
        p2a = (lat_a[sel2[-1]], lon_a[sel2[-1]])

        dist_track_a = geodesic(p1a,p2a).km
        print('along track dist, closest dist, npixel', dist_track_a, dist1, len(sel2))
        print(dd1, sel1)
       
  
        fig = plt.figure('test', figsize=(11,8.5))      
        ax=plt.axes(projection=ccrs.PlateCarree())   #ccrs.Mercator()
        #ax=plt.axes(projection=ccrs.Mercator())   #ccrs.Mercator()
        
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.LAND, edgecolor='black')

        #ax.coastlines()
        ax.plot(lon_a[sel2], lat_a[sel2], '.')
        ax.plot([lon_a[sel2][0],lon, lon_a[sel2][-1]], [lat_a[sel2][0], lat, lat_a[sel2][-1] ])
        ax.plot([lon, lon_a[sel1]], [lat, lat_a[sel1] ], 'k*')
    
        ax.set_xticks(np.arange(lon-3,lon+3,1), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(lat-3,lat+3,1), crs=ccrs.PlateCarree())

        plt.ylabel('Latitude [N$^{\circ}$]')
        plt.xlabel('Longitude [E$^{\circ}$]')
        title = 'dist {0:.2f} track {1:.2f}'.format( dist1, dist_track_a)
        plt.title(title)
        plt.text(lon, lat, site, horizontalalignment='right')
       
        if fn_a is not None: 
            filename_fig = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_lat_lon_dist_'+ site+ '_P'+str(sel)+'_e'+t0.strftime('%H%M%S')+'.png'
        else:
            filename_fig = dir_figure +  os.path.basename(fn1)[0:-3]+ '_lat_lon_dist_'+ site+ '_P'+str(sel)+'_e'+t0.strftime('%H%M%S')+'.png'
        plt.savefig(filename_fig)
        plt.close(fig)

        #plt.text(lon, lat, 'cbw', horizontalalignment='right'),
         # transform=ccrs.Geodetic())

        # plt.plot([lon, lon_a[sel1]], [lat, lat_a[sel1] ],
         # color='red', linestyle='--',
         # transform=ccrs.PlateCarree(),
         # )

        # plt.plot([lon, lon_a[sel1]], [lat, lat_a[sel1] ],
         # color='blue', linestyle=':',
         # transform=ccrs.Geodetic(),
         # )

        #plt.show()


#        1/0
        
    
        stat_data_grd = {}   # ground-based
        stat_data_ec = {}    # earthcare 

        stat_data_ec['distance'] =  dist1
        stat_data_ec['track_length'] = dist_track_a

        stat_data_grd['ext_grd'] = data['extinction']
        stat_data_grd['ext_grd_err'] = data['error_extinction']

        stat_data_grd['backscatter_grd'] = data['backscatter']
        stat_data_grd['backscatter_grd_err'] = data['error_backscatter']

        stat_data_grd['lidarratio_grd'] = data['lidarratio']
        stat_data_grd['lidarratio_grd_err'] = data['error_lidarratio']

        #stat_data_grd['height_b_grd'] =  data_b['altitude']
        #stat_data_grd['height_grd'] =  data['altitude']

        # interpolate the earlinet data to atlid grid

        if fn1 is not None:
            #ebd_data = read_atlid_l2(fn1)
            height_ec_a = ebd_data['height'][sel,:]
            h_ec = np.copy(height_ec_a) 

        elif fn_a is not None:
            #aer_data = read_atlid_l2(fn_a)
            height_ec_a =  aer_data['height'][sel,:]
            h_ec = np.copy(height_ec_a) 

        else:
            print('No AER and EBD file')


        if len(data_b) > 10:  # normally more than 10 variables in data_b
            try:
                depol_earlinet = data_b['particledepolarization']
                depol_earlinet_err = data_b['error_particledepolarization']
            except:
                print('!! data_b has no particledepolarization data ', site)
                depol_earlinet = data_b['altitude'] * 0.0
                depol_earlinet_err = data_b['altitude'] * 0.0
        else:
            depol_earlinet = np.copy(height_ec_a) * 0.0
            depol_earlinet_err = np.copy(height_ec_a) * 0.0
            data_b['altitude'] = np.copy(height_ec_a)

 
         # interpolate the ground-based data to earthcare grid.
        h_ec = np.copy(height_ec_a)   #[::-1])
        #sel_h = np.where(((h_ec >= min(data['altitude'])) & (h_ec <= max(data['altitude']))) )
        sel_h = np.where(((h_ec < min(data['altitude'])) | (h_ec > max(data['altitude']))) )
        #print('sel_h =', sel_h)
        #print('h_ec[sel_h] =', h_ec[sel_h])

        stat_data_ec['height_ec_a'] =  np.copy(h_ec)
        for var in stat_data_grd.keys():
            print('var in stat_data_grd', var)
            y = np.interp( h_ec[::-1], data['altitude'][:],  stat_data_grd[var][:])    # original earthcare grid
            yvar = np.copy(y)
            sel36 = np.where(yvar > 1.e6)
            if len(sel36[0]) > 0:
                 yvar[sel36[0]] = np.nan
            yvar[::-1][sel_h[0]] = np.nan   # remove interpolated data at earthcare grid for the grids above earlinet top and below earlinet ground

            stat_data_ec[var] = np.copy(yvar[::-1]) 

        stat_data_grd['depol_grd'] = depol_earlinet  #data_b['particledepolarization']
        stat_data_grd['depol_grd_err'] = depol_earlinet_err # data_b['error_particledepolarization']

        # stat_data_grd['height_b_grd'] =  data_b['altitude']
        # stat_data_grd['height_grd'] =  data['altitude']

        if len(data_b) > 10:    # choose ground-based profile that has more data points
            #y = np.interp( height_ec_a[::-1], data_b['altitude'][:],  stat_data_grd['depol_grd'][:])
            y = np.interp( h_ec[::-1], data_b['altitude'][:],  stat_data_grd['depol_grd'][:])
            sel36 = np.where(y > 1.e6)   # remove filled value 1e36
            if len(sel36[0]) > 0:
                y[sel36[0]] = np.nan
            y[::-1][sel_h[0]] = np.nan 
            stat_data_ec['depol_grd'] = np.copy(y[::-1]) 

            y = np.interp( h_ec[::-1], data_b['altitude'][:],  stat_data_grd['depol_grd_err'][:])
            if len(sel36[0]) > 0:   # same filter as depol
                y[sel36[0]] = np.nan 
            y[::-1][sel_h[0]] = np.nan 

            stat_data_ec['depol_grd_err'] = np.copy(y[::-1]) 
        else:
            stat_data_ec['depol_grd'] =  stat_data_grd['depol_grd'][:]     # filled with 0, no need to interpolate
            stat_data_ec['depol_grd_err'] = stat_data_grd['depol_grd_err'][:]


  
        if fn_a is not None:

            ec_st1, ec_st2 = ec_time_to_datetime_str(aer_data['time'][sel])   #closest

            # print('closest distance in lat/lon: ', dist, dd[sel],  'pixel: ', sel)

            # title = 'AER ' + os.path.basename(fn_a)[-9:-3] + ' ' + ec_st2 + ' '+str(sel)+ '; ' + site + ' ' + t0.strftime('%H:%M:%S') +  ' D' + str(dist)[0:4] 
            title = os.path.basename(fn_a)[4:8] + ' ' + os.path.basename(fn_a)[-9:-3] + ' ' + ec_st2 + ' '+str(sel)+ '; ' + site + ' ' + t0.strftime('%H:%M:%S') +  ' D' + str(dist)[0:4] 


            sel_qs =  np.where(aer_data['quality_status'][:,:] > 0)   # 0 good, 1 probably good 
            aer_tmp = np.copy( aer_data['particle_extinction_coefficient_355nm'])
            aer_tmp[sel_qs] = np.nan     

            aer_err_tmp = np.copy( aer_data['particle_extinction_coefficient_355nm_error'])
            aer_err_tmp[sel_qs] = np.nan     

            
            ext_ec_a = np.copy(aer_tmp[sel,:])
            ext_ec_err_a = np.copy(aer_err_tmp[sel,:])

            height_ec_a =  aer_data['height'][sel,:]#[sel_cls3]#[10,:] 
            classification = aer_data['classification'][sel,:]

            
            #outfile_ec_a = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_ext_'+ site+ '_P'+str(sel)+'_e'+t0.strftime('%H%M%S')+'.png'
            outfile_ec_a = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_ext_'+ site+ '_P'+str(sel)+'_e'+t0.strftime('%H%M%S')+'F.png'
                    
            #if os.path.isfile(outfile_ec_a):
            #    print('ext figure exsit, skip ', outfile_ec_a )
            #    #return 0, 0, 0
            
            #print('ext_ec_a ', ext_ec_a.shape, height_ec_a.shape)

            plot_ecvt_atl_ext_profile_fill(data['extinction'], data['error_extinction'], data['altitude'], z1, z2, outfile=outfile_ec_a, 
                                  ext_ec=ext_ec_a, ext_ec_err=ext_ec_err_a, height_ec=height_ec_a, title=title, classification = None)

            aer_tmp = np.copy( aer_data['particle_backscatter_coefficient_355nm'])
            aer_tmp[sel_qs] = np.nan     

            aer_err_tmp = np.copy( aer_data['particle_backscatter_coefficient_355nm_error'])
            aer_err_tmp[sel_qs] = np.nan     


            backscatter_ec_a = np.copy(aer_tmp[sel,:])
            backscatter_ec_err_a = np.copy(aer_err_tmp[sel,:])
            height_ec_a =  aer_data['height'][sel,:] 
            outfile_ec_a = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_backscatter_'+ site+ '_'+str(sel)+'_e'+t0.strftime('%H%M%S')+'F.png'

            plot_ecvt_atl_backscatter_profile_fill(data['backscatter'], data['error_backscatter'], data['altitude'], z1, z2, outfile=outfile_ec_a, 
                                  backscatter_ec=backscatter_ec_a, backscatter_ec_err=backscatter_ec_err_a, height_ec=height_ec_a,title=title, classification = classification)


            aer_tmp = np.copy( aer_data['lidar_ratio_355nm'])
            aer_tmp[sel_qs] = np.nan     

            aer_err_tmp = np.copy( aer_data['lidar_ratio_355nm_error'])
            aer_err_tmp[sel_qs] = np.nan     

            lidarratio_ec_a = np.copy(aer_tmp[sel,:])
            lidarratio_ec_err_a = np.copy(aer_err_tmp[sel,:])
            height_ec_a =  aer_data['height'][sel,:] 

            outfile_ec_a = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_lidarratio_'+ site+ '_'+str(sel)+'_e'+t0.strftime('%H%M%S')+'F.png'


            plot_ecvt_atl_lidarratio_profile_fill(data['lidarratio'], data['error_lidarratio'], data['altitude'], z1, z2, outfile=outfile_ec_a, 
                                  lidarratio_ec=lidarratio_ec_a, lidarratio_ec_err=lidarratio_ec_err_a, height_ec=height_ec_a,title=title, classification = classification)

            aer_tmp = np.copy( aer_data['particle_linear_depol_ratio_355nm'])
            aer_tmp[sel_qs] = np.nan     

            aer_err_tmp = np.copy( aer_data['particle_linear_depol_ratio_355nm_error'])
            aer_err_tmp[sel_qs] = np.nan     


            #depol_ec_a = aer_data['particle_linear_depol_ratio_355nm'][sel,:]
            #depol_ec_err_a = aer_data['particle_linear_depol_ratio_355nm_error'][sel,:]#[10,:]
            depol_ec_a = np.copy(aer_tmp[sel,:])
            depol_ec_err_a = np.copy(aer_err_tmp[sel,:])

            height_ec_a =  aer_data['height'][sel,:]#[10,:] 

            # move it out for AER and EBD both
            # if len(data_b) > 10:  # normally more than 10 variables in data_b
                # try:
                    # depol_earlinet = data_b['particledepolarization']
                    # depol_earlinet_err = data_b['error_particledepolarization']
                # except:
                    # print('!! data_b has no particledepolarization data ', site)
                    # depol_earlinet = data_b['altitude'] * 0.0
                    # depol_earlinet_err = data_b['altitude'] * 0.0
            # else:
                # depol_earlinet = np.copy(depol_ec_a) * 0.0
                # depol_earlinet_err = np.copy(depol_ec_a) * 0.0
                # data_b['altitude'] = np.copy(height_ec_a)

            outfile_ec_a = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_depol_'+ site+'_'+str(sel)+ '_e'+t0.strftime('%H%M%S')+'F.png'

            plot_ecvt_atl_depol_profile_fill(depol_earlinet, depol_earlinet_err, data_b['altitude'], z1, z2, outfile=outfile_ec_a, 
                                          depol_ec=depol_ec_a, depol_ec_err=depol_ec_err_a, height_ec=height_ec_a,title=title, classification = classification)

            outfile_ec_a = dir_figure +  os.path.basename(fn_a)[0:-3]+ '_classification_'+ site+'_e'+t0.strftime('%H%M%S')+ '.png'

            plot_aer_classification(aer_data, data, sel, z1, z2, outfile=outfile_ec_a, title=title)



           # stat_data_ec['distance'] =  dist1
           # stat_data_ec['track_length'] = dist_track_a

            stat_data_ec['ext_ec_a'] =  ext_ec_a
            stat_data_ec['ext_ec_err_a'] =  ext_ec_err_a
            
            stat_data_ec['height_ec_a_org'] =  np.copy(height_ec_a)
            stat_data_ec['classification_a'] =  classification

            stat_data_ec['backscatter_ec_a'] = backscatter_ec_a
            stat_data_ec['backscatter_ec_err_a'] =  backscatter_ec_err_a

            stat_data_ec['lidarratio_ec_a'] = lidarratio_ec_a
            stat_data_ec['lidarratio_ec_err_a'] =  lidarratio_ec_err_a

            stat_data_ec['depol_ec_a'] = depol_ec_a
            stat_data_ec['depol_ec_err_a'] =  depol_ec_err_a

            # stat_data_grd['ext_grd'] = data['extinction']
            # stat_data_grd['ext_grd_err'] = data['error_extinction']

            # stat_data_grd['backscatter_grd'] = data['backscatter']
            # stat_data_grd['backscatter_grd_err'] = data['error_backscatter']

            # stat_data_grd['lidarratio_grd'] = data['lidarratio']
            # stat_data_grd['lidarratio_grd_err'] = data['error_lidarratio']

            # # interpolate the ground-based data to earthcare grid.
            # h_ec = np.copy(height_ec_a)   #[::-1])
            # #sel_h = np.where(((h_ec >= min(data['altitude'])) & (h_ec <= max(data['altitude']))) )
            # sel_h = np.where(((h_ec < min(data['altitude'])) | (h_ec > max(data['altitude']))) )
            # #print('sel_h =', sel_h)
            # #print('h_ec[sel_h] =', h_ec[sel_h])
                       

            # stat_data_ec['height_ec_a'] =  np.copy(h_ec)
            # for var in stat_data_grd.keys():
                # print('var in stat_data_grd', var)
                # y = np.interp( h_ec[::-1], data['altitude'][:],  stat_data_grd[var][:])    # original earthcare grid
                # yvar = np.copy(y)
                # sel36 = np.where(yvar > 1.e6)
                # if len(sel36[0]) > 0:
                     # yvar[sel36[0]] = np.nan
                # yvar[::-1][sel_h[0]] = np.nan   # remove interpolated data at earthcare grid for the grids above earlinet top and below earlinet ground

                # stat_data_ec[var] = np.copy(yvar[::-1]) 

            # stat_data_grd['depol_grd'] = depol_earlinet  #data_b['particledepolarization']
            # stat_data_grd['depol_grd_err'] = depol_earlinet_err # data_b['error_particledepolarization']

            # # stat_data_grd['height_b_grd'] =  data_b['altitude']
            # # stat_data_grd['height_grd'] =  data['altitude']

            # if len(data_b) > 10:    # choose ground-based profile that has more data points
                # #y = np.interp( height_ec_a[::-1], data_b['altitude'][:],  stat_data_grd['depol_grd'][:])
                # y = np.interp( h_ec[::-1], data_b['altitude'][:],  stat_data_grd['depol_grd'][:])
                # sel36 = np.where(y > 1.e6)   # remove filled value 1e36
                # if len(sel36[0]) > 0:
                    # y[sel36[0]] = np.nan
                # y[::-1][sel_h[0]] = np.nan 
                # stat_data_ec['depol_grd'] = np.copy(y[::-1]) 

                # y = np.interp( h_ec[::-1], data_b['altitude'][:],  stat_data_grd['depol_grd_err'][:])
                # if len(sel36[0]) > 0:   # same filter as depol
                    # y[sel36[0]] = np.nan 
                # y[::-1][sel_h[0]] = np.nan 

                # stat_data_ec['depol_grd_err'] = np.copy(y[::-1]) 
            # else:
                # stat_data_ec['depol_grd'] =  stat_data_grd['depol_grd'][:]     # filled with 0, no need to interpolate
                # stat_data_ec['depol_grd_err'] = stat_data_grd['depol_grd_err'][:]

        

        # EBD  ########### mean 
 
        if fn1 is not None:

#            ebd_data = read_atlid_l2(fn1)
            atc_data = read_atlid_l2(fn_atc)   # classification_low_resolution


            ec_st1, ec_st2 = ec_time_to_datetime_str(ebd_data['time'][sel])

            # title0 = 'EBD ' + os.path.basename(fn1)[-9:-3] + ' ' + ec_st2 + ' '+str(sel)+'; ' + site + ' ' +t0.strftime('%H:%M:%S')+ ' D ' + str(dist)[0:4]          
            #title0 = os.path.basename(fn_a)[4:8] + ' ' + os.path.basename(fn1)[-9:-3] + ' ' + ec_st2 + ' '+str(sel)+'; ' + site + ' ' +t0.strftime('%H:%M:%S')+ ' D ' + str(dist)[0:4]          
            title0 = os.path.basename(fn1)[4:8] + ' ' + os.path.basename(fn1)[-9:-3] + ' ' + ec_st2 + ' '+str(sel)+'; ' + site + ' ' +t0.strftime('%H:%M:%S')+ ' D ' + str(dist)[0:4]          

            ext_var_list= ['particle_extinction_coefficient_355nm', 
                           'particle_extinction_coefficient_355nm_medium_resolution',
                           'particle_extinction_coefficient_355nm_low_resolution']

            #sel2_cls10 = np.where(aer_data['classification'][:,:] < 10)
            #sel2_qs = np.where(ebd_data['quality_status'][:,:] == 0)
            # sel2_qs_aer = np.logical_or( aer_data['classification'][:,:] < 10, ebd_data['quality_status'][:,:] > 0)
            # change to atc classificiation

            sel2_qs_aer = np.logical_or( atc_data['classification_low_resolution'][:,:] < 10, ebd_data['quality_status'][:,:] > 0)

            #mask_atc = (atc_data['classification_low_resolution'][sel,:] >= 10) | (atc_data['classification_low_resolution'][sel,:] == 3)
            mask_atc = (atc_data['classification_low_resolution'][sel,:] < 3)


            height_ec = ebd_data['height'][sel,:]#[10,:] 
            stat_data_ec['height_ec'] = height_ec
            for ext_var in ext_var_list:

                # use mean mext_ec for mean ext_ec
                ebd_tmp = np.copy(ebd_data[ext_var])
                ebd_tmp[sel2_qs_aer] = np.nan        # good, use sel2_qs_aer for logical_*, not sel2_qs_aer[0]

                ebd_err_tmp = np.copy(ebd_data[ext_var+'_error'])
                ebd_err_tmp[sel2_qs_aer] = np.nan

                seln = np.isnan(ebd_tmp[sel2,:])        # true is 1
                ntot = len(sel2) - np.sum(seln, axis=0)
                print('ntot=', ntot)

                mext_ec =  np.nanmean(ebd_tmp[sel2,:],axis=0).flatten()
                mext_ec_err =  np.sqrt(np.nansum(ebd_err_tmp[sel2,:]*ebd_err_tmp[sel2,:], axis = 0)).flatten()/ntot  #len(sel2[0])
                #mext_ec_err1 =  np.nanmean(ebd_err_tmp[sel2[0],:], axis = 0).flatten()


                mext_ec1 = np.copy(mext_ec)
                sel_ntot = np.where(ntot <= max(ntot)*0.1)   # 10% of max(ntot)  ntot = 100, then 10 is used., remove layers with few aerosol bins, so only use relatively homogeneous layers 
                mext_ec1[sel_ntot] = np.nan
                mext_ec_err1 = np.copy(mext_ec_err)
                mext_ec_err1[sel_ntot] = np.nan

                #print('mext_ec, _err1', mext_ec1, mext_ec_err1)

                #print(mext_ec.shape)

                #ext_ec = ebd_data[ext_var][sel,:]
                #ext_ec_err = ebd_data[ext_var+'_error'][sel,:]  
                # filtered clouds and bad data
                ext_ec = np.copy(ebd_tmp[sel,:])
                ext_ec_err = np.copy(ebd_err_tmp[sel,:]) 
                

                if  ext_var.find('low') >=0:
                    title = title0 + ' L'
                elif    ext_var.find('medium') >=0:
                    title = title0 + ' M'
                else:
                    title = title0
     
                stat_data_ec[ext_var] = ext_ec
                stat_data_ec[ext_var+'_error'] = ext_ec_err

                stat_data_ec[ext_var+'_M'] = mext_ec
                stat_data_ec[ext_var+'_error_M'] = mext_ec_err

                stat_data_ec[ext_var + '_ntot_M'] = ntot
              
                #classification = aer_data['classification'][sel,:]
                classification = np.copy(atc_data['classification_low_resolution'][sel,:])
                stat_data_ec['classification'] =  classification

                ext_plt = np.copy(ebd_data[ext_var][sel,:])
                ext_plt[mask_atc] = np.nan
                ext_plte = np.copy(ebd_data[ext_var+'_error'][sel,:])
                ext_plte[mask_atc] = np.nan

                outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_' + ext_var +'_'+ site+ '_'+str(sel)+'_e'+t0.strftime('%H%M%S')+'.png'
            #txt = os.path.basename(outfile)[13:36] + ' ' + os.path.basename(outfile)[54:60] +  ' ' +   os.path.basename(outfile)[-7:-4]      
                
                # only aerosols
                #plot_ecvt_atl_ext_profile_fill(data['extinction'], data['error_extinction'], data['altitude'], z1, z2, outfile=outfile_ec, 
                #            ext_ec=ext_ec, ext_ec_err=ext_ec_err, height_ec=height_ec,title=title, classification = classification)
 
                # with clouds
                plot_ecvt_atl_ext_profile_fill(data['extinction'], data['error_extinction'], data['altitude'], z1, z2, outfile=outfile_ec, 
                            ext_ec=ext_plt, ext_ec_err=ext_plte, height_ec=height_ec,title=title, classification = classification)

                outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_' + ext_var +'_'+ site+ '_'+str(sel)+'_e'+t0.strftime('%H%M%S')+'_M.png'

                plot_ecvt_atl_ext_profile_fill(data['extinction'], data['error_extinction'], data['altitude'], z1, z2, outfile=outfile_ec, 
                            ext_ec=mext_ec, ext_ec_err=mext_ec_err, height_ec=height_ec,title=title+' m', classification = classification)

                # removed bins with less than 10% data in the average, looks better but not a complete overview
                #plot_ecvt_atl_ext_profile_fill(data['extinction'], data['error_extinction'], data['altitude'], z1, z2, outfile=outfile_ec, 
                #            ext_ec=mext_ec1, ext_ec_err=mext_ec_err1, height_ec=height_ec,title=title, classification = classification)


                
            bks_var_list= ['particle_backscatter_coefficient_355nm', 
                           'particle_backscatter_coefficient_355nm_medium_resolution',
                           'particle_backscatter_coefficient_355nm_low_resolution']
       
            height_ec =  ebd_data['height'][sel,:]#[10,:] 
            for bks_var in bks_var_list: 

         # use mean mext_ec for mean ext_ec
                ebd_tmp = np.copy(ebd_data[bks_var])
                ebd_tmp[sel2_qs_aer] = np.nan

                ebd_err_tmp = np.copy(ebd_data[bks_var+'_error'])
                ebd_err_tmp[sel2_qs_aer] = np.nan

                seln = np.isnan(ebd_tmp[sel2,:])        # true is 1
                ntot = len(sel2) - np.sum(seln, axis=0)
                #print('ntot=', ntot)

                mbackscatter_ec =  np.nanmean(ebd_tmp[sel2,:],axis=0).flatten()
                mbackscatter_ec_err =  np.sqrt(np.nansum(ebd_err_tmp[sel2,:]*ebd_err_tmp[sel2,:], axis = 0)).flatten()/ntot  #len(sel2[0])
                

                #backscatter_ec = ebd_data[bks_var][sel,:]
                #backscatter_ec_err = ebd_data[bks_var+'_error'][sel,:] 
                
                backscatter_ec = np.copy(ebd_tmp[sel,:])
                backscatter_ec_err = np.copy(ebd_err_tmp[sel,:])

       
                outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_'+bks_var + '_'+ site+ '_'+str(sel)+'_e'+t0.strftime('%H%M%S')+'.png'
                if bks_var.find('low') >=0:
                    title = title0 + ' L'
                elif bks_var.find('medium') >=0:
                    title = title0 + ' M'
                else:
                    title = title0

                stat_data_ec[bks_var] = backscatter_ec
                stat_data_ec[bks_var + '_error'] = backscatter_ec_err

                stat_data_ec[bks_var+'_M'] = mbackscatter_ec
                stat_data_ec[bks_var + '_error_M'] = mbackscatter_ec_err

                stat_data_ec[bks_var + '_ntot_M'] = ntot

                bsk_plt = np.copy(ebd_data[bks_var][sel,:])
                bsk_plt[mask_atc] = np.nan
                bsk_plte = np.copy(ebd_data[bks_var+'_error'][sel,:])
                bsk_plte[mask_atc] = np.nan


                plot_ecvt_atl_backscatter_profile_fill(data['backscatter'], data['error_backscatter'], data['altitude'], z1, z2, outfile=outfile_ec, 
                                      backscatter_ec=bsk_plt, backscatter_ec_err=bsk_plte, height_ec=height_ec,title=title, classification = classification)

                outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_'+bks_var + '_'+ site+ '_'+str(sel)+'_e'+t0.strftime('%H%M%S')+'_M.png'
                plot_ecvt_atl_backscatter_profile_fill(data['backscatter'], data['error_backscatter'], data['altitude'], z1, z2, outfile=outfile_ec, 
                                      backscatter_ec=mbackscatter_ec, backscatter_ec_err=mbackscatter_ec_err, height_ec=height_ec,title=title+ ' m', classification = classification)

            #stat_data_ec['ntot'] = ntot
            lidarratio_var_list = ['lidar_ratio_355nm',
                                   'lidar_ratio_355nm_medium_resolution',
                                   'lidar_ratio_355nm_low_resolution']

            for lidarratio_var in lidarratio_var_list:

                ebd_tmp = np.copy(ebd_data[lidarratio_var])
                ebd_tmp[sel2_qs_aer] = np.nan

                ebd_err_tmp = np.copy(ebd_data[lidarratio_var+'_error'])
                ebd_err_tmp[sel2_qs_aer] = np.nan
     
                sel_s = np.where(abs(ebd_err_tmp/ebd_tmp) > 5)       # remove large errors
                ebd_err_tmp[sel_s] = np.nan
                ebd_tmp[sel_s] = np.nan

                seln_lr = np.isnan(ebd_tmp[sel2,:])        # true is 1
                ntot_lr = len(sel2) - np.sum(seln_lr, axis=0)
                #print('ntot_lr', ntot_lr)

                mlidarratio_ec =  np.nanmean(ebd_tmp[sel2,:],axis=0).flatten()
                mlidarratio_ec_err =  np.sqrt(np.nansum(ebd_err_tmp[sel2,:]*ebd_err_tmp[sel2,:], axis = 0)).flatten()/ntot_lr  #len(sel2[0])

                #lidarratio_ec = ebd_data[lidarratio_var][sel,:]       
                #lidarratio_ec_err = ebd_data[lidarratio_var + '_error'][sel,:]#[10,:]

                lidarratio_ec = np.copy(ebd_tmp[sel,:])
                lidarratio_ec_err = np.copy(ebd_err_tmp[sel,:])
                height_ec =  ebd_data['height'][sel,:] 

                outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_' + lidarratio_var +'_'+ site+ '_'+str(sel)+'_e'+t0.strftime('%H%M%S')+'.png'

                if lidarratio_var.find('low') >=0:
                    title = title0 + ' L'
                elif lidarratio_var.find('medium') >=0:
                    title = title0 + ' M'
                else:
                    title = title0

                stat_data_ec[lidarratio_var] = lidarratio_ec 
                stat_data_ec[lidarratio_var + '_error'] = lidarratio_ec_err

                stat_data_ec[lidarratio_var +'_M'] = mlidarratio_ec 
                stat_data_ec[lidarratio_var + '_error_M'] = mlidarratio_ec_err

                stat_data_ec[lidarratio_var +'_ntot_M'] = ntot_lr

                lr_plt = np.copy(ebd_data[lidarratio_var][sel,:])
                lr_plt[mask_atc] = np.nan
                lr_plte = np.copy(ebd_data[lidarratio_var+'_error'][sel,:])
                lr_plte[mask_atc] = np.nan


                plot_ecvt_atl_lidarratio_profile_fill(data['lidarratio'], data['error_lidarratio'], data['altitude'], z1, z2, outfile=outfile_ec, 
                                  lidarratio_ec=lr_plt, lidarratio_ec_err=lr_plte, height_ec=height_ec,title=title, classification = classification)

                outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_' + lidarratio_var +'_'+ site+ '_'+str(sel)+'_e'+t0.strftime('%H%M%S')+'_M.png'
                plot_ecvt_atl_lidarratio_profile_fill(data['lidarratio'], data['error_lidarratio'], data['altitude'], z1, z2, outfile=outfile_ec, 
                                  lidarratio_ec=mlidarratio_ec, lidarratio_ec_err=mlidarratio_ec_err, height_ec=height_ec,title=title +' m', classification = classification)


            #stat_data_ec['ntot_lr'] = ntot_lr


            depol_var_list = ['particle_linear_depol_ratio_355nm',
                              'particle_linear_depol_ratio_355nm_medium_resolution',
                              'particle_linear_depol_ratio_355nm_low_resolution']
            for depol_var in depol_var_list:


                ebd_tmp = np.copy(ebd_data[depol_var])
                ebd_tmp[sel2_qs_aer] = np.nan

                ebd_err_tmp = np.copy(ebd_data[depol_var+'_error'])
                ebd_err_tmp[sel2_qs_aer] = np.nan
     
                sel_s = np.where(abs(ebd_err_tmp/ebd_tmp) > 10)       # remove large errors
                ebd_err_tmp[sel_s] = np.nan
                ebd_tmp[sel_s] = np.nan

                sel_s = np.where(abs(ebd_tmp) > 1.0)       # remove large errors
                ebd_err_tmp[sel_s] = np.nan
                ebd_tmp[sel_s] = np.nan


                seln_dp = np.isnan(ebd_tmp[sel2,:])        # true is 1
                ntot_dp = len(sel2) - np.sum(seln_dp, axis=0)
                #print('ntot_dp', ntot_dp)

                mdepol_ec =  np.nanmean(ebd_tmp[sel2,:],axis=0).flatten()
                mdepol_ec_err =  np.sqrt(np.nansum(ebd_err_tmp[sel2,:]*ebd_err_tmp[sel2,:], axis = 0)).flatten()/ntot_dp  #len(sel2[0])

                #depol_ec = ebd_data[depol_var][sel,:]
                #depol_ec_err = ebd_data[depol_var + '_error'][sel,:]

                depol_ec = np.copy(ebd_tmp[sel,:])
                depol_ec_err = np.copy(ebd_err_tmp[sel,:])
                height_ec =  ebd_data['height'][sel,:]



                outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_'+ depol_var + '_'+ site+ '_'+str(sel)+ '_e'+t0.strftime('%H%M%S')+'.png'

                if depol_var.find('low') >=0:
                    title = title0 + ' L'
                elif depol_var.find('medium') >=0:
                    title = title0 + ' M'
                else:
                    title = title0

                stat_data_ec[depol_var] = depol_ec 
                stat_data_ec[depol_var + '_error'] = depol_ec_err

                stat_data_ec[depol_var+'_M'] = mdepol_ec 
                stat_data_ec[depol_var + '_error_M'] = mdepol_ec_err

                stat_data_ec[depol_var+'_ntot_M'] = ntot_dp

                depol_plt = np.copy(ebd_data[depol_var][sel,:])
                depol_plt[mask_atc] = np.nan
                depol_plte = np.copy(ebd_data[depol_var+'_error'][sel,:])
                depol_plte[mask_atc] = np.nan

                plot_ecvt_atl_depol_profile_fill(depol_earlinet, depol_earlinet_err, data_b['altitude'], z1, z2, outfile=outfile_ec, 
                                  depol_ec=depol_plt, depol_ec_err=depol_plte, height_ec=height_ec,title=title, classification = classification)

                outfile_ec = dir_figure +  os.path.basename(fn1)[0:-3]+ '_'+ depol_var + '_'+ site+ '_'+str(sel)+ '_e'+t0.strftime('%H%M%S')+'_M.png'
                plot_ecvt_atl_depol_profile_fill(depol_earlinet, depol_earlinet_err, data_b['altitude'], z1, z2, outfile=outfile_ec, 
                                  depol_ec=mdepol_ec, depol_ec_err=mdepol_ec_err, height_ec=height_ec,title=title+' m', classification = classification)

            #stat_data_ec['ntot_depol'] = ntot_dp



        #1/0 
        #  ############## change from here #################
        classification_short = np.copy(classification)
        print(classification_short)
        
        #sel10 = np.where(classification > 9.9)   # select aerosols, also for ground-based data        #############select part of the profile with overlap height of ground-based.
                     
        classification_short[sel_h[0]] = -4                                                      # otherwise the ground-based profile is not correcy
        
        sel10 = np.where(classification_short > 9.9)   # sele
        
        stat_data1 = {}

        if len(sel10[0]) > 1:

            sel_nan = np.where(np.isnan(stat_data_ec['particle_extinction_coefficient_355nm']) == True) 
            print(stat_data_ec.keys())
            for var in stat_data_ec.keys():
                # x = np.copy(stat_data_ec[var])                  
                # if var.find('grd') > 0:
                    # print('var', x[var])
                        
                    # x[var][sel_nan[0]] = np.nan
                    # print('var', x[var])
                # 1/0 
                try : 
                    x = np.copy(stat_data_ec[var]) 
                    # if var.find('grd') > 0:
                    #print('var', var,  x)
                            
                        # #x[sel_nan[0]] = np.nan
                        # #print('var nan', x)
          
 
                    #x[0:150] = np.nan    # remove data above ~9 km 
                    sel36 = np.logical_and( x[sel10[0]] > -1.e-6,  x[sel10[0]] < 200)   # remove 1e36 filled values in earlinet
                    print('var', var)
                    stat_data1[var+'_mean'] = np.nanmean(x[sel10[0]][sel36])    # sel36[0] does not work for logical_add
                    stat_data1[var+'_std'] = np.nanstd(x[sel10[0]][sel36])
                    
                except Exception as error:
                    print('ext var=', var)
                    print('exception error ', error)
                    stat_data1[var] = np.copy(x)

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
    dir_data = '/nobackup/users/wangp/Earthcare/data/ECVT/earlinet/qc04/'
    #fn_e = 'EARLINET_ECVT_AerRemSen_dus_Lev01_e0355_202408061900_202408062000_v01_qc03.nc'
    #fn_b = 'EARLINET_ECVT_AerRemSen_dus_Lev01_b0355_202408061900_202408062000_v01_qc03.nc'
    dir_figure = '/nobackup/users/wangp/Earthcare/figures/comparison/v4b/'   #v3a10/'



    site_list = ['arr', 'aky', 'atz', 'brc', 'bgd', 'cog', 'ino', 'cbw', 'cvo', 'puy', 'clj', 'dus', 'evo', 'gar', 'gra', 'ipr', 'kuo', 
                 'sal', 'lei', 'lle', 'cyc', 'lim', 'mdr', 'mas', 'nap', 'hpb', 'sir', 'pay', 'pot', 'rme', 'sof', 'spl', 'the', 'waw']

    #site_list = ['dus', 'waw', 'pot', 'puy']  #, 'hpb']
    #date_list = ['20240814', '20240815', '20240818', '20200820', '20240821', '20240822', '20240824', '20240827','20240828','20240829', '20240830']


    t = datetime.datetime.utcnow()
    #date_list = '20240812,...,' + t.strftime('%Y%m%d')
    x, date_list = generate_date_list('20250518', '20250519')
    #x, date_list = generate_date_list('20250101', t.strftime('%Y%m%d'))
    #x, date_list = generate_date_list('20241005', t.strftime('%Y%m%d'))
    #x, date_list = generate_date_list('20241005', '20241005')
    #x, date_list = generate_date_list('20250415', '20250415')


    site_list = ['cbw'] #['cbw']  #['kuo']  #['pot']  ['
    #date_list = ['20250519' ]  #, '20240902', '20240927'] #814', '20240905','20240906']
    #date_list = ['20241006', '20241013', '20240918']   #     '20240815', 
    date_list = ['20250613', '20250620']
    plot_earlinet = 0
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
            fn_e_all.sort() 
            fn_b_all = []    #fill in later

            if len(fn_e_all) == 0:
                print('# No earlinet e0355 data: ', site, ' ' , date)
                continue
                #1/0

            # check if fn_e_all have different versions
            # use the latest version if there are more versions
            x = []
            for fx in fn_e_all :
                x.append(os.path.basename(fx[0:-12]))
            x1 = np.unique(x)
            x = np.array(x)
            x1 = np.array(x1)

            fn_e_all_new = []
            for i in range(len(x1)):
                sel = np.where(x1[i] == x)
                print('sel = ', sel[0][-1])
                fn_e_all_new.append(fn_e_all[sel[0][-1]])


            #npos = fn_e[0].find('e0355')
            #fn_b = fn_e[0][0:npos]+'b0355'+fn_e[0][npos+5:]
            
            #fn_e = 'EARLINET_ECVT_AerRemSen_hpb_Lev01_e0355_202408100000_202408100100_v01_qc03.nc'
            #fn_b = 'EARLINET_ECVT_AerRemSen_hpb_Lev01_b0355_202408100000_202408100100_v01_qc03.nc'

            z1 = 0
            z2 = 20 

            print('number of 0355 file, lastest version = ', len(fn_e_all), len(fn_e_all_new)) 
            for fn_e in fn_e_all_new:

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
                    #1/0 

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
                if date >= '20250901':  #'20241001':  #'20240901':    
                    dir_ec_gj = '/net/pc230016/nobackup_1/users/zadelhof/EarthCARE_DATA/L2/'    # copy data from pc230016
                else:
                    dir_ec_gj = None                                                            # do not copy data,  use local data pc230036  
                #for using data in downthemall 
                #dir_ec = None                                                               # do not copy data,  use local data pc230036                                          
                #dir_downthemall = '/nobackup/users/wangp/Earthcare/data/L2/DownThemAll/' 
                dir_downthemall = None
                
                dt=1.5    # 1   4
                dist_min = 100  # km
                dir_year = date[0:4]
                ec_product1 = 'ATL_EBD_2A'
                dir_ec_local1 = '/nobackup/users/wangp/Earthcare/data/L2/{0}/{1}/'.format(ec_product1,dir_year)   # directory added year

                #collocated_fn_ebd = find_atl_l2_collocated_files(ec_product1, date, frame_id, t0, dt, dir_ec_local1, dir_ec=dir_ec) # dir_ec = None)
                collocated_fn_ebd = find_atl_l2_collocated_files_downthemall(ec_product1, date, frame_id, t0, dt, dir_ec_local1, dir_ec=dir_ec_gj, dir_downthemall=dir_downthemall) # dir_ec = None)

                ec_product2 = 'ATL_AER_2A'
                dir_ec_local2 = '/nobackup/users/wangp/Earthcare/data/L2/{0}/{1}/'.format(ec_product2,dir_year) 
                #collocated_fn_aer = find_atl_l2_collocated_files(ec_product2, date, frame_id, t0, dt, dir_ec_local2, dir_ec=dir_ec) # dir_ec = None)
                collocated_fn_aer = find_atl_l2_collocated_files_downthemall(ec_product2, date, frame_id, t0, dt, dir_ec_local2, dir_ec=dir_ec_gj, dir_downthemall=dir_downthemall) # dir_ec = None)

                ec_product3 = 'ATL_TC__2A'
                dir_ec_local3 = '/nobackup/users/wangp/Earthcare/data/L2/{0}/{1}/'.format(ec_product3,dir_year) 
                # #collocated_fn_aer = find_atl_l2_collocated_files(ec_product2, date, frame_id, t0, dt, dir_ec_local2, dir_ec=dir_ec) # dir_ec = None)
                collocated_fn_atc = find_atl_l2_collocated_files_downthemall(ec_product3, date, frame_id, t0, dt, dir_ec_local3, dir_ec=dir_ec_gj, dir_downthemall=dir_downthemall) # dir_ec = None)

                if len(collocated_fn_aer) == 0:
                    print('Not find AER file, continue')
                    continue
                    #1/0 
                # read AEL_EBD_2A file  #ATL_AER_2A 
                print()
                print('number of collocated file EBD files in dt = ', len(collocated_fn_ebd) )
                if  len(collocated_fn_ebd)  == 0:
                    print('No EBD file, skip site ', site)                
                    continue


                if cmp_single == 1:     #make quick look figures, not save data.
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

                
                # use AER pr ATC classification to select EBD
                cmp_aer_ebd_atc = 1
                if cmp_aer_ebd_atc == 1:
                    
                    for fn_a in  collocated_fn_aer:

                        a1=0
                        a2=0
                        a3=0
                        a4=0 
                        data_pickle = 0
                        
                        # look for ebd data
                        #fn1 = glob.glob(dir_ec_local1+'**/*EBD*'+ fn_a[-10:], recursive=True)[0]
                        # select ECA_EXAC_ATL_EBD to be sure the same AER and EBD version. Difference version may have difference number of measurements. eg. EXAA and EXAC 01985D
                        # assume AER, EBD, TC_ have the same version number. They are from A-PRO, should be the same version.
                        #fn1 = glob.glob(dir_ec_local1+'**/'+ os.path.basename(fn_a)[0:13]+'EBD*'+ fn_a[-10:], recursive=True)[0]
                        fn1 = glob.glob(dir_ec_local1+'**/'+ os.path.basename(fn_a)[0:13]+'EBD*'+ fn_a[-10:], recursive=True)
                        if len(fn1) > 0 and isinstance(fn1, list):
                            fn1 = fn1[0] 
                              
 
                        fn_atc = glob.glob(dir_ec_local3+'**/'+ os.path.basename(fn_a)[0:13]+'TC_*'+ fn_a[-10:], recursive=True)
                        if len(fn_atc) == 0:
                            print('no ATC file ',     os.path.basename(fn_a)[0:13]+'TC_*'+ fn_a[-10:])
                           
                            continue
                        else:
                            fn_atc = fn_atc[0]
                        
                        
                        #print('### compare data for site, date, fn1, fn_a :', site, date, fn1, fn_a)  
                        #print()

                        #a1, a2, a3, a4  = cmp_ebd_aer_filter_mean(fn1, fn_a, data, data_b, dist_min, site, dir_figure,z1=0,z2=20)   #should also work
                        a1, a2, a3, a4  =  cmp_ebd_aer_atc_filter_mean(data, data_b, dist_min, site, dir_figure, fn1=fn1, fn_a=fn_a, fn_atc=fn_atc, z1=0, z2=20)
                        if a1 != 0 : 
                            print('### compare data for site, date, fn1, fn_a :', site, date, os.path.basename(fn1), os.path.basename(fn_a))  
                            print()
                            ec_grd_data[date+'_'+site] = a2
                            stat_data[date+'_'+site] = a1
                            grd_data[date+'_'+site] = a3
                            nsite = nsite + 1
                            data_pickle = {'ec_ground_collocated_data':ec_grd_data, 'stat_data': stat_data, 'ground_site_data':grd_data, 
                                    'ebd_file':fn1, 'aer_file':fn_a, 'ealinet_e0355':fn_e, 'ealinet_b0355':fn_b} 
                           
                            fn_pickle = dir_figure + '{0}_ground_collocated_{1}_site_{2}_P{3}_e{4}_ATC.pickle'.format(os.path.basename(fn_a)[0:8],os.path.basename(fn_a)[20:35], site, a4, t0.strftime('%H%M%S'))  # some sites may have no data
                            #if os.path.isfile(fn_pickle):
                            #    fn_pickle =   dir_figure + 'ec_ground_collocated_{0}_site_{1}_P{2}_e{3}a.pickle'.format(date,site, a4, t0.strftime('%H%M%S'))     
                 
                            pickle_out = open(fn_pickle, 'wb')
                            pickle.dump(data_pickle,pickle_out, protocol=2)

                            fn_figure = fn_pickle[0:-7]
                            if len(a1) != 0:
                                print('mean' , a1['ext_grd_mean'], a1['ext_ec_a_mean'] ) 
                                plot_ext_backscatter_means(data_pickle, fn_figure)
                        
                cmp_ebd_atc = 0        
                if cmp_ebd_atc == 1:   # only ebd tc_ file.

                    for fn1 in  collocated_fn_ebd:

                        a1=0
                        a2=0
                        a3=0
                        a4=0 
                        data_pickle = 0
                        
                        # look for ebd data
                        #fn1 = glob.glob(dir_ec_local1+'**/*EBD*'+ fn_a[-10:], recursive=True)[0]
                        # select ECA_EXAC_ATL_EBD to be sure the same AER and EBD version. Difference version may have difference number of measurements. eg. EXAA and EXAC 01985D

                        fn_atc = glob.glob(dir_ec_local3+'**/'+ os.path.basename(fn1)[0:13]+'TC_*'+ fn1[-10:], recursive=True)        # save version of ATC EBD
                        if len(fn_atc) == 0:
                            continue
                        else:
                            fn_atc = fn_atc[0]

                        
                        #print('### compare data for site, date, fn1, fn_a :', site, date, fn1, fn_a)  
                        #print()

                        #a1, a2, a3, a4  = cmp_ebd_aer_filter_mean(fn1, fn_a, data, data_b, dist_min, site, dir_figure,z1=0,z2=20)    # use AER classification
                        a1, a2, a3, a4  =  cmp_ebd_aer_atc_filter_mean(data, data_b, dist_min, site, dir_figure, fn1=fn1, fn_a=None, fn_atc=fn_atc, z1=0, z2=20)
                        if a1 != 0 : 
                            print('### compare data for site, date, fn1 :', site, date, os.path.basename(fn1) )  
                            print()
                            ec_grd_data[date+'_'+site] = a2
                            stat_data[date+'_'+site] = a1
                            grd_data[date+'_'+site] = a3
                            nsite = nsite + 1
                            data_pickle = {'ec_ground_collocated_data':ec_grd_data, 'stat_data': stat_data, 'ground_site_data':grd_data, 
                                    'ebd_file':fn1, 'ealinet_e0355':fn_e, 'ealinet_b0355':fn_b} 
                           
                            fn_pickle = dir_figure + '{0}_ground_collocated_{1}_site_{2}_P{3}_e{4}_EBD.pickle'.format(os.path.basename(fn1)[0:8],os.path.basename(fn1)[20:35], site, a4, t0.strftime('%H%M%S'))  # some sites may have no data
                            #if os.path.isfile(fn_pickle):
                            #    fn_pickle =   dir_figure + 'ec_ground_collocated_{0}_site_{1}_P{2}_e{3}a.pickle'.format(date,site, a4, t0.strftime('%H%M%S'))     
                 
                            pickle_out = open(fn_pickle, 'wb')
                            pickle.dump(data_pickle,pickle_out, protocol=2)

                            # fn_figure = fn_pickle[0:-7]
                            # if (len(a1) != 0) and (fn_a is not None):
                                # print('mean' , a1['ext_grd_mean'] ) 
                                # plot_ext_backscatter_means(data_pickle, fn_figure)



             
                        

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

# check the A_TC classification, and classification low resolution.  
# /nobackup/users/wangp/Earthcare/figures/keep_data/ECA_EXAD_ATL_TC__2A_20250218T230343Z_20250219T075120Z_04139B
