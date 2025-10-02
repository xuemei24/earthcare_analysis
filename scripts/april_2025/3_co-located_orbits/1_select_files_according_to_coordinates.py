import numpy as np
import glob
import sys
import os

script_path = '/usr/people/wangxu/Desktop/earthcare_scripts/scripts/april_2025/'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), script_path)))

from ectools import ecio
def is_within_bounds(latitudes, longitudes, lat_min, lat_max, lon_min, lon_max):
    lat_mask = (latitudes >= lat_min) & (latitudes <= lat_max)
    lon_mask = (longitudes >= lon_min) & (longitudes <= lon_max)
    return np.any(lat_mask & lon_mask)

def sel_files(file_paths,lat_min,lat_max,lon_min,lon_max):
    selected_ebd = []
    for i,file_path in enumerate(file_paths):
        print(i)
        with ecio.load_AEBD(file_path,prodmod_code="ECA_EXAE") as AEBD:
        #with h5py.File(file_path, 'r') as file:
            lat = AEBD['latitude'].values
            lon = AEBD['longitude'].values
 
            if is_within_bounds(lat, lon, lat_min, lat_max, lon_min, lon_max):
                print(lon)
                selected_ebd.append(file_path)

    return selected_ebd

month = 'august'
srcdir = '/net/pc190625/nobackup_1/users/wangxu/earthcare_data/'+month+'_2025/EBD/'
ebd_files = sorted(glob.glob(srcdir+'*h5'))

lat_min = 51
lat_max = 83
lon_min = -72
lon_max = 20
fname = 'Greenland'

lat_min = -51
lat_max = -9
lon_min = 109
lon_max = 178
fname = 'Australia'

lat_min = -12
lat_max = 25
lon_min = -31
lon_max = 31
fname = 'Africa'


'''
lat_min = -70
lat_max = -40
lon_min = -175
lon_max = -79
fname = 'Southern_Ocean'

lat_min = 8.5
lat_max = 33
lon_min = 70.8
lon_max = 89.3
fname = 'India'
'''

lat_min = 40
lat_max = 90
lon_min = -114
lon_max = 64
fname = 'smoke_region'
f_regional = sel_files(ebd_files,lat_min,lat_max,lon_min,lon_max)
#np.savetxt('selected_files_Greenland.txt',np.array(f_regional),delimiter=',')

f = open('selected_files_'+fname+'_'+month+'_2025.txt','w')
f.write("\n".join(f_regional) + "\n")
f.close()
