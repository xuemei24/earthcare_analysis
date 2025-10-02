import wget
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

data_dict = {
    'endpoint': 'https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3',
    #'station': '',#'Palma_de_Mallorca',
    'year': 2025,
    'month': 3,
    'day': 11,
    'year2': 2025,
    'month2': 3,
    'day2': 31,
    #'lat1': -8,
    #'lon1': -33,
    #'lat2': 38,
    #'lon2': 31,
    #'if_no_html': 1,
    'AOD15': 1,
    'AVG': 10
}

url = '{endpoint}?year={year}&month={month}&day={day}&year2={year2}&month2={month2}&day2={day2}&AOD15={AOD15}&AVG={AVG}'.format(**data_dict)
#url = '{endpoint}?site={station}&year={year}&month={month}&day={day}&year2={year2}&month2={month2}&day2={day2}&AOD20={AOD20}&AVG={AVG}'.format(**data_dict)
#url = '{endpoint}?site={station}&year={year}&month={month}&day={day}&year2={year2}&month2={month2}&day2={day2}&lat1={lat1}&lon1={lon1}&lat2={lat2}&lon2={lon2}&AOD20={AOD20}&AVG={AVG}&if_no_html={if_no_html}'.format(**data_dict)


print(url)
wget.download(url, '/net/pc190625/nobackup_1/users/wangxu/aeronet/{year}0{month}_all_sites_aod15_allpoints.txt'.format(**data_dict))

#wget.download(url, '{year}0{month}_africa_aod20_10.txt'.format(**data_dict))

