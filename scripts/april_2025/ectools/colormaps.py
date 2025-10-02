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


import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
import pandas as pd
import glob
import os
import xarray as xr
import numpy as np

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib as mpl


from matplotlib.colors import LogNorm, Normalize, ListedColormap, LinearSegmentedColormap, ColorConverter

def register_rgb_colormap(r, g, b, colormap_name):

    # Ensure R, G, B arrays are of the same length
    assert len(r) == len(g) == len(b), "R, G, B arrays must have the same length."

    # Stack R, G, B arrays horizontally and normalize to the range [0, 1]
    rgb_values = np.vstack((r, g, b)).T / 255.0

    # Create the colormap
    colormap = ListedColormap(rgb_values, name=colormap_name)

    # Background color
    colormap.set_bad("white")

    return register_colormap(colormap)


def register_colormap(colormap, overwrite=True):
   
    # Check if the colormap is already registered
    if colormap.name in plt.colormaps():
        if not overwrite:
            print(f"Colormap '{colormap.name}' is already registered. Skipping registration.")
            return plt.get_cmap(colormap.name)
        else:
            print(f"Colormap '{colormap.name}' is already registered. Overwriting by default.")

    # Register the colormap
    mpl.colormaps.register(cmap=colormap, force=True)

    return colormap


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return LinearSegmentedColormap('CustomMap', cdict)


def make_coloralphamap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
            cdict['alpha'].append([item, b1, b2])
    return LinearSegmentedColormap('CustomMap', cdict)


def define_calipso_cm():
    #Predefined list of RGB values
    red=(0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,\
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,\
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,\
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,\
         0,  0,  0,  0,  0,  0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,\
         255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,\
         255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,\
         255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255, 70, 70,\
         70, 90, 90, 90,110,110,110,130,130,130,150,150,150,150,150,150,170,170,170,170,\
         170,170,180,180,180,180,180,180,190,190,190,190,190,190,200,200,200,200,200,200,\
         210,210,210,210,210,210,215,215,215,215,215,215,220,220,220,220,220,220,225,225,\
         225,225,225,225,230,230,230,230,230,230,235,235,235,235,235,235,240,240,240,240,\
         240,240,240,245,245,245,245,245,245,245,255,255,255,255,255,255,255)


    green=(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,\
           42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,127,127,127,127,127,127,\
           127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,\
           127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,255,255,255,127,\
           127,127,170,170,170,170,255,255,255,255,255,255,255,255,255,255,255,255,255,255,\
           255,230,230,230,230,230,230,230,230,230,212,212,212,212,212,212,212,212,212,170,\
           170,170,170,170,170,170,170,170,127,127,127,127,127,127, 85, 85, 85, 85, 85, 85,\
           0,  0,  0,  0,  0,  0, 42, 42, 42, 42, 42, 42, 85, 85, 85,127,127,127, 70, 70,\
           70, 90, 90, 90,110,110,110,130,130,130,150,150,150,150,150,150,170,170,170,170,\
           170,170,180,180,180,180,180,180,190,190,190,190,190,190,200,200,200,200,200,200,\
           210,210,210,210,210,210,215,215,215,215,215,215,220,220,220,220,220,220,225,225,\
           225,225,225,225,230,230,230,230,230,230,235,235,235,235,235,235,240,240,240,240,\
           240,240,240,245,245,245,245,245,245,245,255,255,255,255,255,255,255)

    blue=(170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,\
          170,170,170,170,170,170,170,170,170,170,170,170,170,170,255,255,255,255,255,255,\
          255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,\
          255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,170,170,170,127,\
          127,127, 85, 85, 85, 85,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,\
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,\
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,\
          0,  0,  0,  0,  0,  0, 85, 85, 85, 85, 85, 85,127,127,127,170,170,170, 70, 70,\
          70, 90, 90, 90,110,110,110,130,130,130,150,150,150,150,150,150,170,170,170,170,\
          170,170,180,180,180,180,180,180,190,190,190,190,190,190,200,200,200,200,200,200,\
          210,210,210,210,210,210,215,215,215,215,215,215,220,220,220,220,220,220,225,225,\
          225,225,225,225,230,230,230,230,230,230,235,235,235,235,235,235,240,240,240,240,\
          240,240,240,245,245,245,245,245,245,245,255,255,255,255,255,255,255)
    
    # Register the colormap
    cmap= register_rgb_colormap(red, green, blue, 'calipso')
    cmap.set_under(np.array([0,42,170])/255.)
    return cmap

def define_calipso_smooth():

    darkest_blue = tuple(np.array([0,20,100])/255.)
    dark_blue = tuple(np.array([0,40,132])/255.)
    light_blue = tuple(np.array([63,200,238])/255.)
    green = tuple(np.array([27,169,89])/255.)
    yellow = tuple(np.array((241,231,32))/255.)
    orange = tuple(np.array((249,165,25),)/255.)
    red = tuple(np.array([240,85,43])/255.)
    magenta = tuple(np.array([241,126,171])/255.)
    dark_grey = tuple(np.array((70,70,70))/255.)
    mid_grey = tuple(np.array((200,200,200))/255.)
    white = tuple(np.array((255,255,255))/255.)
    black = 'k'
    
    c = ColorConverter().to_rgb
    cmap = make_colormap([darkest_blue, 0.05, darkest_blue,
                            dark_blue, 0.15, dark_blue, 
                            light_blue, 0.27, light_blue, 
                            green, 0.32, green,
                            green, 0.33, yellow,
                            yellow, 0.35, yellow,
                            orange, 0.53, orange,
                            red, 0.6, red, 
                            magenta, 0.66, magenta, 
                            magenta, 0.67, dark_grey, 
                            mid_grey, 0.80, mid_grey, 
                            white, 0.95, white])
    cmap.name = 'calipso_smooth'
    
    cmap.set_under(darkest_blue)
    cmap.set_over(white)
    cmap.set_bad(darkest_blue)
    
    # Register the colormap
    return register_colormap(cmap)
    

def define_calipso2_cm():
    cmap_points = np.array([(1,40,132),
                        (18,54,137),
                       (34,60,147),
                       (60,119,182),
                       (70,161,220),
                       (63,200,238),
                       (110,200,218),
                       (114,193,146),
                       (10,125,127),
                       (27,169,89),
                       (241,231,32),
                       (245,232,23),
                       (249,211,3),
                       (249,165,25),
                       (248,124,38),
                       (240,85,43),
                       (234,35,30),
                       (239,43,93),
                       (238,85,125),
                       (241,126,171),
                       (70,70,70),
                       (99,99,99),
                       (130,130,130),
                       (155,155,155),
                       (180,180,180),
                       (200,200,200),
                       (210,215,215),
                       (230,230,230),
                       (240,240,240),
                       (245,245,245),
                       (250,250,250),
                       (254,254,254)])
    
    cmap = register_rgb_colormap(cmap_points[:-1,0], cmap_points[:-1,1], cmap_points[:-1,2], 'calipso2')
    
    cmap.set_under(tuple(cmap_points[0]/255))
    cmap.set_over(tuple(cmap_points[-1]/255))
    cmap.set_bad('k')
    return cmap


def define_chiljet2_cm():
    c = ColorConverter().to_rgb
    cmap = make_colormap([c('0.9'), 
                          c('blue'), 0.2, c('blue'), 
                          c('green'), 0.35, c('green'), 
                          c('yellow'), 0.67, c('yellow'), 
                          c('red'), 0.9, c('red'), 
                          c('black')])
    cmap.name = 'chiljet2'
    
    # Register the colormap
    return register_colormap(cmap)
    

def define_chiljet3_cm():
    darkest_blue = tuple(np.array([0,20,100])/255.)
    dark_blue = tuple(np.array([0,40,132])/255.)
    light_blue = tuple(np.array([63,200,238])/255.)
    green = tuple(np.array([27,169,89])/255.)
    yellow = tuple(np.array((241,231,32))/255.)
    orange = tuple(np.array((249,165,25),)/255.)
    red = tuple(np.array([240,85,43])/255.)
    magenta = tuple(np.array([241,126,171])/255.)
    
    c = ColorConverter().to_rgb
    cmap = make_colormap([darkest_blue, 
                          dark_blue,  0.1,  dark_blue,
                          light_blue, 0.33, light_blue, 
                          green,      0.5,  green, 
                          yellow,     0.67, yellow, 
                          red,        0.9,  red, 
                          magenta])
    cmap.name = 'chiljet3'
    
    # Register the colormap
    return register_colormap(cmap)


def define_litmus_cm(blue_high=False):
    if blue_high:
        cmap = 'RdBu'
    else:
        cmap = 'RdBu_r'
        
    cmap = plt.get_cmap(cmap)
    cmap.name = 'litmus'
    
    # Register the colormap
    register_colormap(cmap)
    return cmap


def define_truncated_cm(cmap, v=[0,0.5,1.0], suffix='truncated'):
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)

    def centred_v(v):
        # Get the larger interval
        dv = np.max([v[1]-v[0], v[2]-v[1]])
        
        # Dummy interval with centred values
        return [v[1]-dv, v[1], v[1]+dv]
        
    def normalize_v(v=[0, 0.5, 1.0]):
        cv = centred_v(v)
    
        # Calculating interval with normalized values
        return [0.5-(cv[1]-v[0])/(cv[2]-cv[0]), 0.5, 0.5+(v[2]-cv[1])/(cv[2]-cv[0])]
            
    def truncated_colormap(cmap, v, n=100):
        nv = normalize_v(v)
        
        from matplotlib.colors import LinearSegmentedColormap
        truncated_cmap = LinearSegmentedColormap.from_list(
                                        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=nv[0], b=nv[2]),
                                        cmap(np.linspace(nv[0], nv[2], n)))
        return truncated_cmap
    
    tcmap = truncated_colormap(cmap, v)
    tcmap.name = f"{cmap.name}_{suffix}"
    mpl.colormaps.register(cmap=tcmap, name=tcmap.name, force=True)
    
    return tcmap


def define_TIR_blue_cm():
    c = ColorConverter().to_rgb
    cmap = make_colormap([c('1.0'),
                                    c('0.9'), 0.1, c('0.9'),
                                    c('0.8'), 0.2, c('0.8'), 
                                    c('0.7'), 0.3, c('0.7'), 
                                    c(sns.xkcd_rgb['grey blue']), 1-0.52, c(sns.xkcd_rgb['grey blue']), 
                                    c(sns.xkcd_rgb['medium blue']), 0.6, c(sns.xkcd_rgb['medium blue']), 
                                    c(sns.xkcd_rgb['twilight blue']), 1-0.23, c(sns.xkcd_rgb['twilight blue']), 
                                    c(sns.xkcd_rgb['deep blue']), 1-0.04, c(sns.xkcd_rgb['deep blue']), 
                                    c('black')])
    cmap.name = 'SW'

    
    # Register the colormap
    register_colormap(cmap)
    register_colormap(cmap.reversed())
    return cmap


def define_TIR_red_cm():
    c = ColorConverter().to_rgb
    cmap = make_colormap([c('1.0'),
                            c('0.9'), 0.1, c('0.9'),
                            c('0.8'), 0.2, c('0.8'), 
                            c('0.7'), 0.3, c('0.7'), 
                            c(sns.xkcd_rgb['pastel red']), 1-0.45, c(sns.xkcd_rgb['pastel red']), 
                            c(sns.xkcd_rgb['burnt red']), 1-0.23, c(sns.xkcd_rgb['burnt red']), 
                            c(sns.xkcd_rgb['dried blood']), 1-0.065, c(sns.xkcd_rgb['dried blood']), 
                            c('black')])
    cmap.name = 'LW'
    
    # Register the colormap
    register_colormap(cmap)
    register_colormap(cmap.reversed())
    return cmap


def define_TIR_redblue_cm():
    c = ColorConverter().to_rgb
    cmap = make_colormap([c(sns.xkcd_rgb['ice blue']),
                          c('1.0'), 0.10, c('1.0'),
                          c('1.0'), 0.15, c('1.0'),
                          c('0.9'), 0.225, c('0.9'),
                          c('0.8'), 0.35, c('0.8'), 
                          c('0.7'), 0.4, c('0.7'), 
                          c(sns.xkcd_rgb['pastel red']), 1-0.45, c(sns.xkcd_rgb['pastel red']), 
                          c(sns.xkcd_rgb['burnt red']), 1-0.23, c(sns.xkcd_rgb['burnt red']), 
                          c(sns.xkcd_rgb['dried blood']), 1-0.065, c(sns.xkcd_rgb['dried blood']), 
                          c('black')])
    cmap.name = 'LW_coldtops'
    
    # Register the colormap
    register_colormap(cmap)
    register_colormap(cmap.reversed())
    return cmap

sw = define_TIR_blue_cm()
lw = define_TIR_red_cm()
lw_coldtops = define_TIR_redblue_cm()
chiljet2 = define_chiljet2_cm()
chiljet3 = define_chiljet3_cm()
calipso = define_calipso_cm()
calipso_smooth = define_calipso_smooth()
litmus = define_litmus_cm()
litmus_doppler = define_truncated_cm('litmus', v=[-2,0,6], suffix='doppler')


def example_colorbar(cmap, norm, units):
    from matplotlib.cm import ScalarMappable
    
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    
    fig, ax = plt.subplots(figsize=(6, 0.5))
    
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax, orientation='horizontal', label=units)
    ax.set_title(cmap.name)
