"""
Copyright 2023- ECMWF

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,def quickook_ACM(
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
sns.set_style('ticks')
sns.set_context('poster')

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, ListedColormap, LinearSegmentedColormap, ColorConverter

from . import colormaps 

import numpy as np
import xarray as xr
import pandas as pd


def format_time(ax, format_string="%H:%M:%S", label='Time (UTC)', fontsize='medium'):
    import matplotlib.dates as mdates
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    ax.xaxis.set_major_formatter(mdates.DateFormatter(format_string))
    ax.set_xlabel(label, fontsize=fontsize)    

def format_height(ax, scale=1.0e3, label='Height [km]'):
    import matplotlib.ticker as ticker
    ticks_y = ticker.FuncFormatter(lambda x, pos: '${0:g}$'.format(x/scale))
    ax.yaxis.set_major_formatter(ticks_y)
    ax.set_ylabel(label)

def format_temperature(ax, label='Temperature [K]'):
    import matplotlib.ticker as ticker
    ax.set_ylabel(label)
    ax.invert_yaxis()

def format_across_track(ax, label='across-track\npixel [-]'):
    import matplotlib.ticker as ticker
    ticks_y = ticker.FuncFormatter(lambda x, pos: '${0:g}$'.format(x))
    ax.yaxis.set_major_formatter(ticks_y)
    ax.set_ylabel(label) 

def format_latitude(ax, axis='x'):
    import matplotlib.ticker as ticker
    latFormatter = ticker.FuncFormatter(lambda x, pos: "${:g}^\circ$S".format(-1*x) if x < 0 else "${:g}^\circ$N".format(x))
    if axis == 'x':
        ax.xaxis.set_major_formatter(latFormatter)
    else:
        ax.yaxis.set_major_formatter(latFormatter)

def format_longitude(ax, axis='x'):
    import matplotlib.ticker as ticker
    lonFormatter = ticker.FuncFormatter(lambda x, pos: "${:g}^\circ$W".format(-1*x) if x < 0 else "${:g}^\circ$E".format(x))
    if axis == 'x':
        ax.xaxis.set_major_formatter(lonFormatter)
    else:
        ax.yaxis.set_major_formatter(lonFormatter)


import matplotlib
import copy
cmap_grey_r = copy.copy(matplotlib.cm.get_cmap('Greys_r')) 
cmap_grey_r.set_over('magenta', alpha=0.5)
cmap_grey_r.set_under('cyan', alpha=0.5) 

cmap_grey = copy.copy(matplotlib.cm.get_cmap('Greys'))
cmap_grey.set_over('magenta', alpha=0.5)
cmap_grey.set_under('cyan', alpha=0.5)

cmap_rnbw = copy.copy(matplotlib.cm.get_cmap('rainbow'))
cmap_rnbw.set_over('grey', alpha=0.5)
cmap_rnbw.set_under('grey', alpha=0.5) 

cmap_org = copy.copy(matplotlib.cm.get_cmap('Oranges'))
cmap_org.set_over('darkred')
cmap_org.set_under('white') 

cmap_smc = copy.copy(matplotlib.cm.get_cmap('seismic'))
cmap_smc.set_over('magenta', alpha=0.5)
cmap_smc.set_under('cyan', alpha=0.5) 

def add_colorbar(ax, cm, label, on_left=False, horz_buffer=0.025, width_ratio="1.25%"):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    if on_left:
        bbox_left = 0 - horz_buffer
    else:
        bbox_left = 1 + horz_buffer
       
    cax = inset_axes(ax,
                     width=width_ratio,  # percentage of parent_bbox width
                     height="100%",  # height : 50%
                     loc=3,
                     bbox_to_anchor=(bbox_left,0,1,1),
                     bbox_transform=ax.transAxes,
                     borderpad=0.0,
                     )       
    return plt.colorbar(cm, cax=cax, label=label)

     
ACTC_category_colors = [sns.xkcd_rgb['silver'],         #unknown
                        sns.xkcd_rgb['reddish brown'],         #surface and subsurface
                        sns.xkcd_rgb['white'],         #clear
                        sns.xkcd_rgb['dull red'],      #rain in clutter
                        sns.xkcd_rgb['off blue'],     #snow in clutter
                        sns.xkcd_rgb['dull yellow'],   #cloud in clutter
                        sns.xkcd_rgb['dark red'],      #heavy rain',
                        sns.xkcd_rgb["navy blue"],   #heavy mixed-phase precipitation
                        sns.xkcd_rgb['light grey'],    #clear (poss. liquid) 
                        sns.xkcd_rgb['pale yellow'],   #liquid cloud
                        sns.xkcd_rgb['golden'],        #drizzling liquid
                        sns.xkcd_rgb['orange'],        #warm rain
                        sns.xkcd_rgb['bright red'],    #cold rain
                        sns.xkcd_rgb['easter purple'], # melting snow
                        sns.xkcd_rgb['dark sky blue'],        # snow (possible liquid)
                        sns.xkcd_rgb['bright blue'], # snow
                        sns.xkcd_rgb["prussian blue"],   # rimed snow (poss. liquid)
                        sns.xkcd_rgb['dark teal'],   # rimed snow and SLW
                        sns.xkcd_rgb['teal'],              # snow and SLW
                        sns.xkcd_rgb['light green'],   # supercooled liquid
                        sns.xkcd_rgb["sky blue"],      # ice (poss. liquid)
                        sns.xkcd_rgb['bright teal'],   # ice and SLW
                        sns.xkcd_rgb['light blue'],    # ice (no liquid)
                        sns.xkcd_rgb['pale blue'],     # strat. ice, PSC II
                        sns.xkcd_rgb['neon green'],    # PSC Ia
                        sns.xkcd_rgb['greenish cyan'], # PSC Ib
                        sns.xkcd_rgb['ugly green'],    # insects
                        sns.xkcd_rgb['sand'],          # dust
                        sns.xkcd_rgb['pastel pink'],   # sea salt
                        sns.xkcd_rgb['dust'],          # continental pollution
                        sns.xkcd_rgb['purpley grey'],  # smoke
                        sns.xkcd_rgb['dark lavender'], # dusty smoke
                        sns.xkcd_rgb['dusty lavender'],# dusty mix
                        sns.xkcd_rgb['pinkish grey'],  # stratospheric aerosol 1 (ash)
                        sns.xkcd_rgb['light khaki'],       # stratospheric aerosol 2 (sulphate)
                        sns.xkcd_rgb['light grey'],    # stratospheric aerosol 3 (smoke)]
                  ]

ACTC_qstat_colors = [sns.xkcd_rgb['earth'],        #0: surface
                    sns.xkcd_rgb['white'],         #1: clear (high)
                    sns.xkcd_rgb['sky blue'],      #2: hydrometeors (high)
                    sns.xkcd_rgb['pale blue'],     #3: hydrometeors (lidar only)
                    sns.xkcd_rgb['bright red'],    #4: aerosols (lidar only)
                    sns.xkcd_rgb['bright green'],  #5: stratosphere (radar clear)
                    sns.xkcd_rgb['off white'],     #6: clear (no radar)
                    sns.xkcd_rgb["pale green"],    #7: stratosphere (no radar)
                    sns.xkcd_rgb['bright blue'],   #8: hydrometeors (lidar ext)
                    sns.xkcd_rgb['pale grey'],     #9: clear (lidar ext)
                    sns.xkcd_rgb['light grey'],    #10:clear (no lidar)
                    sns.xkcd_rgb['neon blue'],     #11:hydrometeors (no lidar)
                    sns.xkcd_rgb['grey'],          #12:unknown
                    sns.xkcd_rgb['ugly green'],    #13:radar artefact
                    sns.xkcd_rgb['midnight'],      #14:both instruments obscured
                    sns.xkcd_rgb['fawn'],          #15: instruments disagree on surface
                    'm',         #16:missing data
                    ]

ACTC_synergy_colors = [sns.xkcd_rgb['grey'],       #-4: no information
                    sns.xkcd_rgb['brown'],         #-3: subsurface (radar-lidar)
                    sns.xkcd_rgb['light brown'],      #-2: subsurface (lidar only)
                    sns.xkcd_rgb['earth'],     #-1: subsurface (radar only)
                    'm',                           #0: unassigned
                    sns.xkcd_rgb['white'],  #1: clear (radar-lidar)
                    '0.95',     #2: clear (lidar only)
                    '0.85',    #3: clear (radar only)
                    sns.xkcd_rgb['bright green'],   #4: target (radar-lidar)
                    sns.xkcd_rgb['butter'],     #5: target (lidar only)
                    sns.xkcd_rgb['sky blue'],    #6: target (radar only)
                    ]

ATC_category_colors = [sns.xkcd_rgb['silver'],      #missing data
                       sns.xkcd_rgb['reddish brown'],       #surface and subsurface
                       sns.xkcd_rgb['light grey'],      #noise in both Mie and Ray channels
                       sns.xkcd_rgb['white'],       #clear
                       sns.xkcd_rgb['pale yellow'],   #liquid cloud
                       sns.xkcd_rgb['light green'], # supercooled liquid
                       sns.xkcd_rgb['light blue'],    # ice (no liquid)
                       sns.xkcd_rgb['sand'],          # dust
                       sns.xkcd_rgb['pastel pink'],   # sea salt
                       sns.xkcd_rgb['dust'],   # continental pollution
                       sns.xkcd_rgb['purpley grey'],  # smoke
                       sns.xkcd_rgb['dark lavender'], # dusty smoke
                       sns.xkcd_rgb['dusty lavender'],# dusty mix
                       sns.xkcd_rgb['pale blue'],      # strat. ice, PSC II
                       sns.xkcd_rgb['neon green'],     # PSC Ia
                       sns.xkcd_rgb['greenish cyan'],     # PSC Ib
                       sns.xkcd_rgb['pinkish grey'],     # stratospheric aerosol 1 (ash)
                       sns.xkcd_rgb['light khaki'],     # stratospheric aerosol 2 (sulphate)
                       sns.xkcd_rgb['light grey'],     # stratospheric aerosol 3 (smoke)]
                       '0.9', #'101: Unknown: Aerosol Target has a very low probability (no class assigned)',
                       '0.9', #'102: Unknown: Aerosol classification outside of param space',
                       '0.9', #'104: Unknown: Strat. Aerosol Target has a very low probability (no class assigned)',
                       '0.9', #'105: Unknown: Strat. Aerosol classification outside of param space',
                       '0.9', #'106: Unknown: PSC Target has a very low probability (no class assigned)',
                       '0.9'  #'107: Unknown: PSC classification outside of param space'
                      ]

CTC_category_colors = [sns.xkcd_rgb['silver'],      #missing data
                       sns.xkcd_rgb['reddish brown'],       #surface and subsurface
                       sns.xkcd_rgb['white'],       #clear
                       sns.xkcd_rgb['pale yellow'],  #liquid cloud
                       sns.xkcd_rgb['golden'], # drizzling liquid cloud
                       sns.xkcd_rgb['orange'], # warm rain
                       sns.xkcd_rgb['bright red'],    #cold rain
                       sns.xkcd_rgb['easter purple'], # melting snow
                       sns.xkcd_rgb["prussian blue"],   # rimed snow (poss. liquid)
                       sns.xkcd_rgb['bright blue'], # snow
                       sns.xkcd_rgb['light blue'],    # ice (no liquid)
                       sns.xkcd_rgb['ice blue'],      # strat. ice
                       sns.xkcd_rgb['ugly green'],    # insects
                       sns.xkcd_rgb['dark red'],      # heavy rain likely 
                       sns.xkcd_rgb["royal blue"],   # mixed-phase precip. likely
                       sns.xkcd_rgb['dark red'],      # heavy rain
                       sns.xkcd_rgb["navy blue"],   #heavy mixed-phase precipitation
                       sns.xkcd_rgb['dull red'],     # rain in clutter 
                       sns.xkcd_rgb['off blue'],     #snow in clutter
                       sns.xkcd_rgb['dull yellow'],    # cloud in clutter 
                       sns.xkcd_rgb['light grey'],    # clear (poss. liquid) 
                       sns.xkcd_rgb['silver'],        # unknown
                      ]

MAOT_qstat_category_colors = [sns.xkcd_rgb['green'],
                              sns.xkcd_rgb['light green'],
                              sns.xkcd_rgb['pale yellow'],
                              sns.xkcd_rgb['golden'],
                              sns.xkcd_rgb['orange']]

MCOP_qstat_category_colors = [sns.xkcd_rgb['green'],
                              sns.xkcd_rgb['light green'],
                              sns.xkcd_rgb['pale yellow'],
                              sns.xkcd_rgb['golden'],
                              sns.xkcd_rgb['orange']]
    
MCM_maskphase_category_colors = [sns.xkcd_rgb['pale yellow'],
                                 sns.xkcd_rgb['light blue'],
                                 sns.xkcd_rgb['cyan'],
                                 sns.xkcd_rgb['orange'],
                                 sns.xkcd_rgb['dark red']]

MCM_type_category_colors = [sns.xkcd_rgb['light grey'],
                            sns.xkcd_rgb['navy'],
                            sns.xkcd_rgb['blue'],
                            sns.xkcd_rgb['cyan'],
                            sns.xkcd_rgb['lime'],
                            sns.xkcd_rgb['green'],
                            sns.xkcd_rgb['yellow'],
                            sns.xkcd_rgb['orange'],
                            sns.xkcd_rgb['red'],
                            sns.xkcd_rgb['salmon'],
                            sns.xkcd_rgb['dark red']]

MCM_maskphase_category_colors = [sns.xkcd_rgb['light grey'],
                                 sns.xkcd_rgb['navy'],
                                 sns.xkcd_rgb['cyan'],
                                 sns.xkcd_rgb['orange'],
                                 sns.xkcd_rgb['dark red']]

MCM_qstat_category_colors = [sns.xkcd_rgb['green'],
                             sns.xkcd_rgb['light green'],
                             sns.xkcd_rgb['pale yellow'],
                             sns.xkcd_rgb['golden'],
                             sns.xkcd_rgb['orange']]
    
    
def add_nadir_track(ax, idx_across_track=284, dark_mode=False, label_below=True, label_offset=1, zorder=11):
    if dark_mode:
        text_color = 'w'
        text_shading= 'k'
        shading_alpha = 0.5
        shading_lw = 5
    else:
        text_color = 'k'
        text_shading = 'w'
        shading_alpha = 0.5
        shading_lw = 5
    
    _x0, _x1 = ax.get_xlim()
    ax.plot([_x0, _x1], [idx_across_track, idx_across_track], 
            color=text_shading, lw=4, ls='-', alpha=0.33, zorder=zorder)
    ax.plot([_x0, _x1], [idx_across_track, idx_across_track], 
            color=text_color, lw=1.5, ls='--', zorder=zorder+1)
    
    if label_below:
        shade_around_text(ax.text(_x0, idx_across_track+label_offset, " nadir", ha='left', va='top', color=text_color, fontsize='xx-small'), 
                            lw=shading_lw, alpha=shading_alpha, fg=text_shading)
        shade_around_text(ax.text(_x1, idx_across_track+label_offset, "nadir ", ha='right', va='top', color=text_color, fontsize='xx-small'), 
                            lw=shading_lw, alpha=shading_alpha, fg=text_shading)
    else:
        shade_around_text(ax.text(_x0, idx_across_track-label_offset, " nadir", ha='left', va='bottom', color=text_color, fontsize='xx-small'), 
                            lw=shading_lw, alpha=shading_alpha, fg=text_shading)
        shade_around_text(ax.text(_x1, idx_across_track-label_offset, "nadir ", ha='right', va='bottom', color=text_color, fontsize='xx-small'), 
                            lw=shading_lw, alpha=shading_alpha, fg=text_shading)

def linesplit(s, line_break):
    if ((len(s) <= line_break) | (len(s) <= line_break+0.25*line_break)):
        return [s]
    else:
        last_idx  = s[0:line_break].rfind(' ')
        beginning = s[:last_idx]
        remaining = s[last_idx+1:]
        return [beginning] + linesplit(remaining, line_break)

def linebreak(s, line_break):
    return '\n'.join( linesplit(s, line_break))


def plot_EC_target_classification(ax, ds, varname, category_colors, 
                                   hmax=15e3, hmin=-0.5e3, label_fontsize='xx-small', 
                                   processor=None, title=None, title_prefix=None,
                                   savefig=False, dstdir="./", show_latlon=True,
                                   use_latitude=False, dark_mode=False, use_localtime=True,
                                   timevar='time', heightvar='height', latvar='latitude', lonvar='longitude', 
                                   across_track=False, line_break=None, fillna=None, 
                                   show_colorbar=True):
    
    if processor is None:
        processor = "-".join([t.replace("_","") for t in ds.encoding['source'].split("/")[-1][9:16].split('_', maxsplit=1)])
        
    if title is None:
        long_name = ds[varname].attrs['long_name'].split(" ")
        #Removing capitalizations unless it's an acronym
        for i, l in enumerate(long_name):
            if not l.isupper():
                long_name[i] = l.lower()
        if title_prefix:
            long_name = [title_prefix.strip()] + long_name
        else:
            title_prefix=""
        long_name = " ".join(long_name)
        title = f"{processor} {title_prefix}{long_name}"

        if len(title) > 50:
            title_parts = title.split(" ")
            title = "\n".join([" ".join(title_parts[:4]), " ".join(title_parts[4:])])
    
    cleanup_category = lambda s: s.strip().replace('possible', 'poss.').replace('supercooled', "s\'cooled").replace("stratospheric", 'strat.').replace('extinguished', 'ext.').replace('precipitation', 'precip.').replace('and', '&').replace('unknown', 'unk.').replace('precipitation', 'precip.')
    
    if "\n" in ds[varname].attrs['definition']:
        definitions = ds[varname].attrs['definition']
        if definitions.endswith("\n"):
            definitions = definitions[:-1]
        categories = [cleanup_category(s) for s in definitions.split('\n')]
    else:
        #C-TC uses comma-separated "definition" attribute, but also uses commas within definitions.
        #categories = [cleanup_category(s) for s in ds[varname].attrs['definition'].replace("ground clutter,", "ground clutter;").split(',')]
        #Same for M-COP's quality_status.
        categories = [cleanup_category(s) for s in ds[varname].attrs['definition'].replace("ground clutter,", "ground clutter;").replace('valid, quality','valid; quality').replace('valid, degraded','valid; degraded').split(',')]

    categories = [c.replace("_", " ") for c in categories]
    
    import pandas as pd
    if line_break is not None: categories = [linebreak(c, line_break) for c in categories]
    if ':' in categories[0]:
        try:               # try/except added due to bit# instead of integers in M-CM *_quality_status ... only for ':',
                           #     could be adapted for '=' definitions too, if needed or for completeness...
            first_c = int(categories[0].split(":")[0])
            last_c  = int(categories[-1].split(":")[0])
            u = np.array([int(c.split(':')[0]) for c in categories])
            
        except ValueError: 
            first_c = int(categories[0].split(":")[0].split('bit')[-1])-1
            last_c  = int(categories[-1].split(":")[0].split('bit')[-1])-1
            u = 2**np.array([int(c.split(':')[0].split('bit')[-1])-1 for c in categories])
        categories_formatted = [f"${c.split(':')[0]}$:{c.split(':')[1]}" for c in categories]
        
    elif '=' in categories[0]:
        first_c = int(categories[0].split("=")[0])
        last_c  = int(categories[-1].split("=")[0])
        u = np.array([int(c.split('=')[0]) for c in categories])
        categories_formatted = [f"${c.split('=')[0]}$:{c.split('=')[1]}" for c in categories]
        
    else:
        print("category values are not included within categories")

    # to account for categories that are not strictly monotonically increasing in attribute definition: 0,1,2,..., -127
    idx                  = np.argsort(u)
    u                    = u[idx]
    categories_formatted = list(np.array(categories_formatted)[idx]) 
    
    bounds = np.concatenate(([u.min()-1], u[:-1]+np.diff(u)/2., [u.max()+1]))
    
    from matplotlib.colors import ListedColormap, BoundaryNorm
    norm = BoundaryNorm(bounds, len(bounds)-1)
    cmap = ListedColormap(sns.color_palette(category_colors[:len(u)]).as_hex())
    
    if use_latitude:
        if fillna is None:
            _l, _h, _z = xr.broadcast(ds[latvar], ds[heightvar], ds[varname])
        else:
            _l, _h, _z = xr.broadcast(ds[latvar], ds[heightvar], ds[varname].fillna(fillna))
            
        if (np.isnan(_h).sum() > 0):
            _cm = ax.pcolor(_l, _h, _z, norm=norm, cmap=cmap)
        else:
            _cm = ax.pcolormesh(_l, _h, _z, norm=norm, cmap=cmap)
            
    else:
        if fillna is None:
            _t, _h, _z = xr.broadcast(ds[timevar], ds[heightvar], ds[varname])
        else:
            _t, _h, _z = xr.broadcast(ds[timevar], ds[heightvar], ds[varname].fillna(fillna))
        if (np.isnan(_h).sum() > 0):
            _cm = ax.pcolor(_t, _h, _z, norm=norm, cmap=cmap)
        else:
            _cm = ax.pcolormesh(_t, _h, _z, norm=norm, cmap=cmap)

    _cb = add_colorbar(ax, _cm, '', horz_buffer=0.01)
    _cb.set_ticks(bounds[:-1]+np.diff(bounds)/2.)
    _cb.ax.set_yticklabels(categories_formatted, fontsize=label_fontsize)
        
    if not show_colorbar:
        _cb.remove()
    
    format_plot(ax, ds, title, hmax=hmax, hmin=hmin, heightvar=heightvar, 
                dark_mode=dark_mode, use_localtime=use_localtime, 
                latvar=latvar, lonvar=lonvar, across_track=across_track, use_latlon=show_latlon)
    
    if savefig:
        import os
        dstfile = f"{product_code}_{varname}.png"
        fig.savefig(os.path.join(dstdir,dstfile), bbox_inches='tight')
        
    
    
AAER_aerosol_classes = [10,11,12,13,14,15]
CCLD_ice_classes = [7,8,9,13,15,17]
CCLD_rain_classes = [3,4,5,12,14,16]


def format_plot(ax, ds, title, hmax=20e3, hmin=-0.5e3, dark_mode=False, 
                heightvar='height', timevar='time', latvar='latitude', lonvar='longitude', across_track=False,
                dim_name='along_track', use_localtime=True, use_latlon=True,
                short_timestep=False):
    
    #Set title
    ax.set_title(title)
    if across_track:
        ax.set_ylim(hmax,0)
        format_across_track(ax)
        
    else:
        if 'temperature' in heightvar.lower():
            ax.set_ylim(hmin, hmax)
            format_temperature(ax)
            
        #elif 'height' in heightvar:
        else:    
            if hmax > 1e3:
                ax.set_ylim(hmin, hmax)
                format_height(ax, scale=1e3)
            else:
                ax.set_ylim(hmin, hmax)
                format_height(ax, scale=1)

    if short_timestep:
        if False:
            format_time_ticks(ax, ds, timevar, lonvar, dim_name, 
                          major_step='10s', minor_step='2s',
                          use_localtime=use_localtime)
        else:
            format_time_ticks(ax, ds, timevar, lonvar, dim_name, 
                          major_step='30s', minor_step='10s',
                          use_localtime=use_localtime)
    else:
        format_time_ticks(ax, ds, timevar, lonvar, dim_name, use_localtime=use_localtime)
    
    #Complement time axis ticks with lat/lon information
    if use_latlon:
        _ax = ax.twiny()
        _ax.set_xlim(ax.get_xlim())
        format_latlon_ticks(ax, _ax, ds, timevar, lonvar, latvar, dim_name)
    
    if dark_mode:
        text_color = 'w'
        text_shading= 'k'
        shading_alpha = 0.5
        shading_lw = 5
    else:
        text_color = 'k'
        text_shading = 'w'
        shading_alpha = 0.5
        shading_lw = 5

    #Specify the product filename
    product_code = ds.encoding['source'].split('/')[-1].split('.')[0]   
    
    shade_around_text(ax.text(0.9975,0.98, product_code, ha='right', va='top', 
                      fontsize='xx-small', color=text_color, transform=ax.transAxes), 
                      lw=shading_lw, alpha=shading_alpha, fg=text_shading)
    
    
    
def plot_EC_2D(ax, ds, varname, label, 
             plot_where=True, min_value=None, fill_value=None, scale_factor=1, 
             smoother=None,
             hmax=20e3, hmin=-0.5e3, plot_scale=None, plot_range=None, cmap=None,
             units=None, processor=None, title=None, title_prefix="",
             timevar='time', heightvar='height', latvar='latitude', lonvar='longitude',
    	     across_track=False, dark_mode=False, use_localtime=True,
             short_timestep=False):

    sns.set_style('ticks')
    sns.set_context('poster')
    
    import pandas as pd
    
    if plot_scale is None:
        plot_scale = ds[varname].attrs['plot_scale']
    
    if plot_range is None:
        plot_range = ds[varname].attrs['plot_range']
        
    if 'log' in plot_scale:
        norm=LogNorm(plot_range[0], plot_range[-1])
    else:
        norm=Normalize(plot_range[0], plot_range[-1])
    
    if cmap is None:
        cmap=colormaps.chiljet2
    
    if processor is None:
        processor = "-".join([t.replace("_","") for t in ds.encoding['source'].split("/")[-1][9:16].split('_', maxsplit=1)])
    
    if units is None:
        units = ds[varname].attrs['units']
    
    if title is None:
        long_name = ds[varname].attrs['long_name'].split(" ")
        #Removing capitalizations unless it's an acronym
        for i, l in enumerate(long_name):
            if not l.isupper():
                long_name[i] = l.lower()
        if title_prefix:
            long_name = [title_prefix.strip()] + long_name
        long_name = " ".join(long_name)
    
        title = f"{processor} {long_name}"

        if len(title) > 50:
            title_parts = title.split(" ")
            title = "\n".join([" ".join(title_parts[:4]), " ".join(title_parts[4:])])

    # White background 
    if smoother:
        _t, _h, _z = xr.broadcast(ds[timevar], ds[heightvar].fillna(-1000), scale_factor*ds[varname].rolling(smoother, center=True).mean())
    else:
        _t, _h, _z = xr.broadcast(ds[timevar], ds[heightvar].fillna(-1000), scale_factor*ds[varname])
        
    if fill_value:
        _z = _z.where(plot_where).fillna(fill_value)

    if min_value and fill_value:
        _z = _z.where(_z > min_value).fillna(fill_value)

    _cm = ax.pcolormesh(_t, _h, _z.where(plot_where), norm=norm, cmap=cmap)

    if len(units) > 0:
        cb_label = f"{label} [{units}]"
        if len(cb_label) > 25:
            add_colorbar(ax, _cm, f"{label}\n[{units}]", horz_buffer=0.01, width_ratio='1%')
        else:
            add_colorbar(ax, _cm, cb_label, horz_buffer=0.01, width_ratio='1%')
    else:
        add_colorbar(ax, _cm, f"{label}", horz_buffer=0.01, width_ratio='1%')
    
    format_plot(ax, ds, title, hmax=hmax, hmin=hmin, dark_mode=dark_mode, timevar=timevar, heightvar=heightvar,
                latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime,
               short_timestep=short_timestep)
    

def add_subfigure_labels(axes, xloc=0.0, yloc=1.125, zorder=0, fontsize='medium',
                         label_list=[], flatten_order='F'):
    if label_list == []:
        import string
        labels = string.ascii_lowercase
    else:
        labels = label_list
        
    for i, ax in enumerate(axes.flatten(order=flatten_order)):
        if ax:
            #ax.text(xloc, yloc, "%s)" %(labels[i]), va='baseline', fontsize=fontsize,
            #        transform=ax.transAxes, fontweight='bold', zorder=zorder)
            ax.set_title(f"{labels[i]})", fontsize=fontsize, loc='left', fontweight='bold')


def add_surface(ax, ds, 
                elevation_var='surface_elevation', 
                land_var='land_flag', hmin=-1e3):

    if (ds is not None) & (elevation_var in ds.data_vars):
        ax.axhspan(hmin,0,lw=0, color=sns.xkcd_rgb['sky blue'], zorder=20)

        if land_var in ds.data_vars:
            ax.fill_between(ds.time[:], ds[elevation_var][:], y2=hmin,
                        lw=0, color=sns.xkcd_rgb['sky blue'], step='mid', zorder=21)
            ax.fill_between(ds.time[:], ds[elevation_var].where(ds[land_var]==1)[:], y2=hmin,
                            lw=0, color=sns.xkcd_rgb['pale brown'], step='mid', zorder=22)
        else:
            ax.fill_between(ds.time[:], ds[elevation_var][:], y2=hmin,
                        lw=0, color='0.5', step='mid', zorder=21)
            
        ax.plot(ds.time[:], ds[elevation_var][:], color='k', lw=2.5)

    
def add_ruler(ax, ds, timevar='time', dx=500, d0=100, x0=100, y0=0.5, pixel_scale_km=1, dark_mode=False):
    
    if ds is not None:

        if dark_mode:
            text_color = 'w'
            text_shading= 'k'
            shading_alpha = 0.5
            shading_lw = 5
        else:
            text_color = 'k'
            text_shading = 'w'
            shading_alpha = 0.5
            shading_lw = 5

        buffer = 0.015
        ground_speed = 7 #km/s
        #transform from axes to data coordinates
        inv = (ax.transScale + ax.transLimits).inverted()
        y0_data = inv.transform([0, y0])[-1]
        ylabel_data = inv.transform([0, y0+buffer])[-1]
        ytick = inv.transform([0, y0-buffer])[-1]
        
        t0 = ds[timevar][x0].values
        t1 = ds[timevar][x0].values + np.timedelta64(int((1000*dx*pixel_scale_km)//ground_speed), 'ms')
        ax.plot([t0, t1], [y0_data,y0_data], color='w', lw=9, alpha=0.33, solid_capstyle='projecting', zorder=99)
        ax.plot([t0, t1], [y0_data,y0_data], color='k', lw=5, solid_capstyle='butt', zorder=100)
        rticks = np.arange(x0,x0+dx+1,d0)
        nticks = len(rticks)

        tticks = np.arange(t0,t1+1,(t1-t0)//5)
        ax.plot(tticks, nticks*[y0_data], color='k', lw=0, marker='|', markersize=6)

        shade_around_text(ax.text(t0 + (t1-t0)/2, ylabel_data, f"scale [km]", fontsize='xx-small', 
                                  va='bottom', ha='center', color=text_color), 
                          lw=shading_lw, alpha=shading_alpha, fg=text_shading)
        for i,ttick in enumerate(tticks):
            shade_around_text(ax.text(ttick, ytick, f"{int((rticks[i]-x0)*pixel_scale_km)}", 
                                      fontsize=10, va='top', ha='center', color=text_color), 
                              alpha=shading_alpha, lw=shading_lw, fg=text_shading)
            if (i%2 == 1) & (i < nticks-1):
                ax.plot([ttick,ttick+np.timedelta64(int((1000*d0*pixel_scale_km)//ground_speed), 'ms')], [y0_data,y0_data], color='w', lw=5, solid_capstyle='butt', zorder=101)

            
def shade_around_text(t, alpha=0.2, lw=2.5, fg='k'):
    import matplotlib.patheffects as PathEffects
    return t.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground=fg, alpha=alpha)])


def add_temperature(ax, ds, 
                   timevar='time', heightvar='height', tempvar='temperature'):
    
    if 'temperature_level' in ds.data_vars:
        _x, _y, _t = xr.broadcast(ds[timevar], ds.height_level, ds.temperature_level - 273.15)
    elif 'elevation' in ds.data_vars:
        _x, _y, _t = xr.broadcast(ds[timevar], ds[heightvar], ds[tempvar].where(ds[heightvar] >= ds.elevation) - 273.15)
    else:
        _x, _y, _t = xr.broadcast(ds[timevar], ds[heightvar], ds[tempvar] - 273.15)
        
    _cn = ax.contour(_x, _y, _t, levels=np.arange(-90,31,10),
                        colors='k', 
                        linewidths=[0.5, 1.0, 0.1, 0.5, 0.5, 1.0, 0.5, 1.0, 0.5, 2.0, 0.5, 1.0, 0.5], zorder=10)
    _cl = plt.clabel(_cn, [l for l in [-80,-40,0] if l in _cn.levels], 
               inline=1, fmt='$%.0f^{\circ}$C', fontsize='xx-small', zorder=11)
    
    for t in _cl:
        t = shade_around_text(t, fg='w', alpha=0.5, lw=5)
        
    for l in _cn.labelTexts:
        
        l.set_rotation(0)
    return _cl


def add_specific_humidity(ax, ds):
    _x, _y, _q = xr.broadcast(ds.time, ds.height_layer, ds.specific_humidity_layer_mean)
    
    contour_levels = [1e-5, 1e-4, 1e-3, 1e-2]
    _cn = ax.contour(_x, _y, _q,
                     levels=contour_levels, cmap='Blues', norm=LogNorm(1e-6,1e-2), 
                     linewidths=3, alpha=0.67, zorder=10)
    _cl = plt.clabel(_cn, inline=1, fmt='%.1e', inline_spacing=15, colors='k',
                     fontsize='12', zorder=10)
    
    for i,t in enumerate(_cl):
        s = f"{int(np.log10(float(t.get_text()))):g}"
        t.set_text("$10^{" + s + "}$ kg/kg")
        t = shade_around_text(t, fg='w', alpha=0.67, lw=3.)
        
    for l in _cn.labelTexts:
        l.set_rotation(0)
    return _cl


def add_relative_humidity(ax, ds):
    _x, _y, _rh = xr.broadcast(ds.time, ds.height, 100.*ds.relative_humidity)
    contour_levels = [10, 25, 50, 75, 99]
    _cn = ax.contour(_x, _y, _rh, 
                     levels=contour_levels, cmap='Blues', norm=Normalize(0,100), 
                     linewidths=3, alpha=0.67, zorder=10)
    _cl = plt.clabel(_cn, inline=1, fmt='%.1e', inline_spacing=15, colors='k',
                     fontsize='12', zorder=10)
    
    for i,t in enumerate(_cl):
        s = int(float(t.get_text()))
        t.set_text(f"${s:d}$%")
        t = shade_around_text(t, fg='w', alpha=0.67, lw=3.)
        
    for l in _cn.labelTexts:
        l.set_rotation(0)
    return _cl


def add_land_sea_border(ax, ds, col='black', zorder=10):
    
    _x, _y, _lm = xr.broadcast(ds.time, ds.across_track, ds.land_flag)
    
    contour_levels = [0,0.5,1]
    _cn = ax.contour(_x, _y, _lm, 
                     levels=contour_levels, colors=col,   
                     linewidths=[1.,1.,0.5], alpha=0.67, zorder=zorder)

def snap_xlims(axes):
    
    xlim = list(axes[0].get_xlim())
    for ax in axes[1:]:
        _xlim = list(ax.get_xlim())
        if _xlim[0] < xlim[0]:
            xlim[0] = _xlim[0]
        if _xlim[1] > _xlim[1]:
            xlim[1] = _xlim[1]
        
    for ax in axes:
        ax.set_xlim(xlim[0], xlim[1])
    
    
def add_extras(ax, ds, ruler_y0=0.9, 
               show_surface=True, show_ruler=True,
               show_humidity=True, show_temperature=True, 
              dark_mode=False):
    
    if show_ruler:
        add_ruler(ax, ds, dx=500, d0=100, x0=100, y0=ruler_y0, dark_mode=dark_mode)
    
    if show_surface:
        add_surface(ax, ds)
    
    if show_temperature:
        add_temperature(ax, ds)
        
    if show_humidity:
        if 'relative_humidity' in ds.data_vars:
            add_relative_humidity(ax, ds)
        elif 'specific_humidity' in ds.data_vars:
            add_specific_humidity(ax, ds)


def quicklook_measurements_CNOM(CNOM, dstdir=None):
    nrows=5
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, gridspec_kw={'hspace':0.67})
    
    cmap = colormaps.chiljet2
    units = 'dBZ'
    hmax=20e3
    plot_scale='linear'
    plot_range=[-40,20] 
    label=r"Z"
    
    plot_EC_2D(axes[0], CNOM, 'radarReflectivityFactor', label, cmap=cmap, units=units, title="Radar reflectivity", fillvalue=None, hmax=hmax,
                          plot_scale=plot_scale, plot_range=plot_range, timevar='profileTime', heightvar='binHeight')
    
    plot_EC_1D(axes[1], CNOM, {'C-NOM':{'xdata':CNOM['profileTime'], 'ydata':CNOM['sigmaZero']}}, 
                     "surface signal", "$\sigma_0$ [dB]", timevar='profileTime', include_ruler=False)

    plot_EC_1D(axes[2], CNOM, {'C-NOM':{'xdata':CNOM['profileTime'], 'ydata':CNOM['noiseFloorPower']}}, 
                     "noise floor power", "$F$ [dBW]", timevar='profileTime', include_ruler=False)
    
    cmap = colormaps.litmus
    units = 'ms$^{-1}$'
    hmax=20e3
    plot_scale='linear'
    plot_range=[-6,6] 
    label=r"V$_D$"
    plot_EC_2D(axes[3], CNOM, 'dopplerVelocity', label, scale_factor=-1., cmap=cmap, units=units, title="mean Doppler velocity", hmax=hmax,
                          plot_scale=plot_scale,  plot_range=plot_range, timevar='profileTime', heightvar='binHeight')
    
    cmap = colormaps.chiljet2
    units = 'ms$^{-1}$'
    hmax=20e3
    plot_scale='linear'
    plot_range=[3,10] 
    label=r"$\sigma_D$"
    plot_EC_2D(axes[4], CNOM, 'spectrumWidth', label, cmap=cmap, units=units, title="Doppler spectrum width", hmax=hmax,
                          plot_scale=plot_scale,  plot_range=plot_range, timevar='profileTime', heightvar='binHeight')
    
    add_subfigure_labels(axes)
    
    if dstdir:
        srcfile_string = CNOM.encoding['source'].split("/")[-1].split(".")[0]
        dstfile = f"{srcfile_string}_measurements_quicklook.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')



def quicklook_platform_CNOM(CNOM, dstdir=None):
    nrows=5
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, gridspec_kw={'hspace':0.67})
    
    plot_EC_1D(axes[0], CNOM, {'C-NOM':{'xdata':CNOM['profileTime'], 'ydata':CNOM['pitchAngle']}}, 
                     "CPR pitch angle", r"$\theta$ [$^\circ$]", timevar='profileTime', include_ruler=False)

    plot_EC_1D(axes[1], CNOM, {'C-NOM':{'xdata':CNOM['profileTime'], 'ydata':CNOM['rollAngle']}}, 
                     "CPR roll angle", r"$\phi$ [$^\circ$]", timevar='profileTime', include_ruler=False)
    
    plot_EC_1D(axes[2], CNOM, {'C-NOM':{'xdata':CNOM['profileTime'], 'ydata':CNOM['yawAngle']}}, 
                     "CPR yaw angle", r"$\psi$ [$^\circ$]", timevar='profileTime', include_ruler=False)
        
    plot_EC_1D(axes[3], CNOM, {'C-NOM':{'xdata':CNOM['profileTime'], 'ydata':CNOM['satelliteVelocityContaminationInLOS']}}, 
                     "CPR satellite velocity contamination in line-of-sight", "V$_c$ [ms$^{-1}$]", timevar='profileTime', include_ruler=False)

    plot_EC_1D(axes[4], CNOM, {'C-NOM':{'xdata':CNOM['profileTime'], 'ydata':CNOM['navigationLandWaterFlg']}}, 
                     "land/water flag", "", timevar='profileTime', include_ruler=False)
    axes[4].set_yticks([0,1])
    axes[4].set_yticklabels(['sea', 'land'])
    axes[4].set_ylim(-0.5,1.5)
    
    add_subfigure_labels(axes)
    
    if dstdir:
        srcfile_string = CNOM.encoding['source'].split("/")[-1].split(".")[0]
        dstfile = f"{srcfile_string}_platform_quicklook.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')


def quicklook_orbit_CNOM(srcdir, prodmod_code, orbit, frame_datetime="*", production_datetime="*", 
                         var='radar_reflectivity', dstdir=None, flat_directories=False):
    from . import ecio
    plotting_dict = {'radar_reflectivity':dict(varname='radarReflectivityFactor',
                                                cmap = colormaps.chiljet2,
                                                units = 'dBZ',
                                                hmax=20e3,
                                                plot_scale='linear',
                                                plot_range=[-40,20] ,
                                                label=r"Z"),
                     'doppler_velocity':dict(varname='dopplerVelocity',
                                                cmap = colormaps.litmus,
                                                units = 'ms$^{-1}$',
                                                hmax=20e3,
                                                plot_scale='linear',
                                                plot_range=[-6,6] ,
                                                label=r"V$_D$"),
                     'spectrum_width':dict(varname='spectrumWidth',
                                                cmap = colormaps.chiljet2,
                                                units = 'ms$^{-1}$',
                                                hmax=20e3,
                                                plot_scale='linear',
                                                plot_range=[3,10] ,
                                                label=r"$\sigma_D$")
                        }

    if var in plotting_dict.keys():
        d = plotting_dict[var]
    else:
        print(f"Plotting configuration for variable {var} not available")
        return

    srcfile_strings = []

    srcfiles = ecio.get_filelist("/perm/pasm/CARDINAL/commissioning_phase/first_CPR_data/CPR_NOM_1B/", 
                                 prodmod_code=prodmod_code, product_code="CPR_NOM_1B", 
                                 frame_datetime=frame_datetime, production_datetime=production_datetime, frame_code=f"{orbit:05d}*",
                                flat_directories=flat_directories)
    frames = [f.split("/")[-1].split(".")[0].split("_")[-1][-1] for f in srcfiles]
    
    nrows=len(frames)
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, gridspec_kw={'hspace':0.67})
    
    for i, f in enumerate(frames):
        CNOM = ecio.load_CNOM(srcdir, prodmod_code=prodmod_code, 
                              frame_datetime=frame_datetime, production_datetime=production_datetime, 
                              frame_code=f"{orbit:05d}{f}")

        srcfile_strings.append(CNOM.encoding['source'].split("/")[-1].split(".")[0])
        plot_EC_2D(axes[i], CNOM, d['varname'], d['label'], cmap=d['cmap'], units=d['units'], title=CNOM[d['varname']].attrs['long_name'], 
                         hmax=d['hmax'], plot_scale=d['plot_scale'], plot_range=d['plot_range'], timevar='profileTime', heightvar='binHeight')
        CNOM.close()
        
    add_subfigure_labels(axes)

    if dstdir:
        dstfile = f"{srcfile_strings[0]}_{'_'.join(srcfile_strings[-1].split('_')[5:])}_{d['varname']}.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')
        plt.close(fig)


def quicklook_orbit_ANOM(srcdir, orbit, frame_datetime="*", prodmod_code="ECA_EXAA", production_datetime="*", 
                         var='mie_attenuated_backscatter', dstdir=None, flat_directories=False):
    from . import ecio
    plotting_dict = {'mie_attenuated_backscatter':dict(varname='mie_attenuated_backscatter',
                                                cmap = colormaps.calipso,
                                                units = 'sr$^{-1}$m$^{-1}$',
                                                hmax=30e3,
                                                plot_scale='log',
                                                plot_range=[1e-8,2e-5] ,
                                                label=r"$\beta_\mathrm{mie}$"),
                     'rayleigh_attenuated_backscatter':dict(varname='rayleigh_attenuated_backscatter',
                                                cmap = colormaps.calipso,
                                                units = 'sr$^{-1}$m$^{-1}$',
                                                hmax=30e3,
                                                plot_scale='log',
                                                plot_range=[1e-8,2e-5] ,
                                                label=r"$\beta_\mathrm{ray}$"),
                     'crosspolar_attenuated_backscatter':dict(varname='crosspolar_attenuated_backscatter',
                                                cmap = colormaps.calipso,
                                                units = 'sr$^{-1}$m$^{-1}$',
                                                hmax=30e3,
                                                plot_scale='log',
                                                plot_range=[1e-8,2e-5] ,
                                                label=r"$\beta_\mathrm{x}$")
                        }

    if var in plotting_dict.keys():
        d = plotting_dict[var]
    else:
        print(f"Plotting configuration for variable {var} not available")
        return

    srcfile_strings = []

    srcfiles = ecio.get_filelist("/perm/pasm/CARDINAL/commissioning_phase/first_ATL_data/ATL_NOM_1B/", 
                                 prodmod_code=prodmod_code, product_code="ATL_NOM_1B", 
                                 frame_datetime=frame_datetime, production_datetime=production_datetime, frame_code=f"{orbit:05d}*",
                                flat_directories=flat_directories)
    frames = [f.split("/")[-1].split(".")[0].split("_")[-1][-1] for f in srcfiles]
    
    nrows=len(frames)
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, gridspec_kw={'hspace':0.67})
    
    for i, f in enumerate(frames):
        ANOM = ecio.load_ANOM(srcdir, prodmod_code=prodmod_code, 
                              frame_datetime=frame_datetime, production_datetime=production_datetime, 
                              frame_code=f"{orbit:05d}{f}")

        srcfile_strings.append(ANOM.encoding['source'].split("/")[-1].split(".")[0])
        plot_EC_2D(axes[i], ANOM, d['varname'], d['label'], cmap=d['cmap'], units=d['units'], title=ANOM[d['varname']].attrs['long_name'], 
                         hmax=d['hmax'], plot_scale=d['plot_scale'], plot_range=d['plot_range'], timevar='time', heightvar='sample_altitude')
        ANOM.close()
        
    add_subfigure_labels(axes)

    if dstdir:
        dstfile = f"{srcfile_strings[0]}_{'_'.join(srcfile_strings[-1].split('_')[5:])}_{d['varname']}.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')
        plt.close(fig)


def intercompare_target_classification(ATC, CTC, ACTC, ACMCOM, hmax=20e3, dstdir=None):
    
    nrows=5

    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.67})

    plot_EC_target_classification(axes[0], ATC, 'classification_low_resolution', 
                                    ATC_category_colors, hmax=hmax, title_prefix="", label_fontsize=10)

    plot_EC_target_classification(axes[1], ACTC, 'ATLID_target_classification_low_resolution', 
                                    ATC_category_colors, hmax=hmax, title_prefix="", label_fontsize=10)

    plot_EC_target_classification(axes[2], CTC, 'hydrometeor_classification', 
                                    CTC_category_colors, hmax=hmax, title_prefix="", label_fontsize=10)

    plot_EC_target_classification(axes[3], ACTC, 'CPR_target_classification', 
                                    CTC_category_colors, hmax=hmax, title_prefix="", label_fontsize=10)

    plot_EC_target_classification(axes[4], ACTC, 'synergetic_target_classification_low_resolution', 
                                    ACTC_category_colors, hmax=hmax, title_prefix="", label_fontsize=10)

    for ax in axes:
        add_extras(ax, CTC, ACMCOM, show_surface=False, show_humidity=False)

    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)


def calculate_ACTC_phase_fractions(ds, varname='synergetic_target_classification_low_resolution',
                              ice_classes=[3,6,13,14,15,16,17,19,20,21], 
                              liquid_classes=[4,8,9,16,17,18,20], 
                              rain_classes=[2,5,9,10,11,12], 
                              aerosol_classes=[26,27,28,29,30,31],                              
                              stratospheric_aerosol_classes=[32,33,34],
                              stratospheric_cloud_classes=[22,23,24],
                             separate_fractions=True):
    
    is_aerosol               = ds[varname].isin(aerosol_classes).astype('float')
    is_stratospheric_aerosol = ds[varname].isin(stratospheric_aerosol_classes).astype('float')
    is_stratospheric_cloud   = ds[varname].isin(stratospheric_cloud_classes).astype('float')
    is_liquid                = ds[varname].isin(liquid_classes).astype('float')
    is_ice                   = ds[varname].isin(ice_classes).astype('float')
    is_rain                  = ds[varname].isin(rain_classes).astype('float')

    if separate_fractions:
        attrs = {'long_name':'fraction', 'units':'-', 'plot_scale':'linear', 'plot_range':[0,1]}

        _attrs = attrs.update()
        ds['ice_fraction'] = xr.DataArray(is_ice, dims=['along_track', 'JSG_height'], name='ice_fraction', 
                                          attrs=dict(attrs, long_name='ice cloud & snow fraction'))
        ds['liquid_fraction'] =  xr.DataArray(is_liquid, dims=['along_track', 'JSG_height'], name='liquid_fraction', 
                                              attrs=dict(attrs, long_name='liquid cloud fraction'))
        ds['rain_fraction'] =  xr.DataArray(is_rain, dims=['along_track', 'JSG_height'], name='rain_fraction', 
                                            attrs=dict(attrs, long_name='rain fraction'))

        ds['aerosol_fraction'] =  xr.DataArray(is_aerosol, dims=['along_track', 'JSG_height'], name='aerosol_fraction', 
                                              attrs=dict(attrs, long_name='aerosol fraction'))
        ds['stratospheric_aerosol_fraction'] =  xr.DataArray(is_stratospheric_aerosol, dims=['along_track', 'JSG_height'], name='stratospheric_aerosol_fraction', 
                                            attrs=dict(attrs, long_name='stratospheric aerosol fraction'))
        ds['stratospheric_cloud_fraction'] =  xr.DataArray(is_stratospheric_cloud, dims=['along_track', 'JSG_height'], name='stratospheric_cloud_fraction', 
                                            attrs=dict(attrs, long_name='stratospheric cloud fraction'))
        
    else:
        ds['phase_fractions'] = xr.concat([is_ice, is_liquid, is_rain], 
                                        xr.IndexVariable(['phase'], ['ice', 'liquid', 'rain']))
    return ds


def calculate_ACTC_phase_classification(ds):
    
    ds['phase_classification'] = (ds.phase_fractions.astype('int')*xr.DataArray([1,2,4], coords={'phase':ds.phase})).sum('phase')
    
    ds.phase_classification.attrs['definition']= "\n".join(['0:clear', 
                                                    '1: ice & snow',
                                                    '2: liquid cloud', 
                                                    '3: mixed-phase',
                                                    '4: rain',
                                                    '5: snow & rain',
                                                    '6: liquid cloud & rain'])
    ds.phase_classification.attrs['long_name'] = 'phase class'

    return ds


def calculate_ACTC_simple_classification(ACTC):

    ACTC = calculate_ACTC_phase_fractions(ACTC, separate_fractions=True)
    
    ACTC['simple_classification'] = 1*ACTC.ice_fraction.astype('int') + 2*ACTC.rain_fraction.astype('int') + 3*ACTC.aerosol_fraction.astype('int') + 4*ACTC.stratospheric_aerosol_fraction.astype('int') + 5*ACTC.stratospheric_cloud_fraction.astype('int')
    ACTC['simple_classification'].attrs['definition'] = "\n".join(["0: clear", "1: ice & snow", "2: rain", "3:aerosols", "4:stratospheric aerosols", "5:stratospheric cloud"])
    
    ACTC['liquid_flag'] = ACTC.liquid_fraction.astype('int')
    ACTC['liquid_flag'].attrs['definition'] = "\n".join(["0: clear","1: liquid cloud"])
    return ACTC



def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def plot_ACTC_simple_classification(ax, ACTC, hmax=25e3):
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    
    plot_EC_target_classification(ax, ACTC, 'simple_classification', 
                                        ['w',  sns.xkcd_rgb['light blue'], sns.xkcd_rgb['lightish red'], 
                                         sns.xkcd_rgb['pink'], sns.xkcd_rgb['mauve'], sns.xkcd_rgb['mustard']], 
                                        title="", show_colorbar=False)
    
    _x, _y, _z = xr.broadcast(ACTC.time, ACTC.height, ACTC.liquid_flag)
    cs = ax.contourf(_x, _y, _z, levels=[-0.5, 0.5, 1.5],
                     colors=[[1,1,1,0], list(np.array(hex_to_rgb(sns.xkcd_rgb['dark aqua']))/255) + [0.5]], 
                     edgecolor='k', lw=0, hatches=['','////'])
    for i, collection in enumerate(cs.collections):
        collection.set_edgecolor('k')
        collection.set_linewidth(0.0)
    
    import matplotlib.patches as mpatches
    liquid_patch = mpatches.Patch(color=list(np.array(hex_to_rgb(sns.xkcd_rgb['dark aqua']))/255) + [0.67],  
                                 label='liquid cloud', hatch='////')
    liquid_patch.set_edgecolor('k')
    liquid_patch.set_linewidth(0.0)
    
    rain_patch = mpatches.Patch(color=sns.xkcd_rgb['lightish red'], 
                                 label='rain & drizzle')
    ice_patch = mpatches.Patch(color=sns.xkcd_rgb['light blue'], 
                                 label='ice & snow')
    aerosol_patch = mpatches.Patch(color=sns.xkcd_rgb['pink'], 
                                 label='aerosol')
    strat_aerosol_patch = mpatches.Patch(color=sns.xkcd_rgb['mauve'], 
                                 label='stratospheric aerosol')
    strat_patch = mpatches.Patch(color=sns.xkcd_rgb['mustard'], 
                                 label='stratospheric cloud')
    
    plt.legend(handles=[ ice_patch, rain_patch, liquid_patch, aerosol_patch, strat_aerosol_patch, strat_patch], 
               frameon=False, fontsize='x-small', bbox_to_anchor=(1,1), loc='upper left')
    ax.set_title("Simplified hydrometeor classification")
    
    ax.set_ylim(0,hmax)



def plot_ACTC_example(ACTC, CCLD=None, ACMCOM=None, hmax=20e3, dstdir=None):
    
    nrows=3

    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.67})

    plot_EC_target_classification(axes[0], ACTC, 'ATLID_target_classification_low_resolution', 
                                    ATC_category_colors, hmax=hmax, title_prefix="", label_fontsize=10)

    plot_EC_target_classification(axes[1], ACTC, 'CPR_target_classification', 
                                    CTC_category_colors, hmax=hmax, title_prefix="", label_fontsize=10)

    plot_EC_target_classification(axes[2], ACTC, 'synergetic_target_classification_low_resolution', 
                                    ACTC_category_colors, hmax=hmax, title_prefix="", label_fontsize=10)

    if (CCLD is not None) & (ACMCOM is not None):
        for ax in axes:
            add_extras(ax, CCLD, ACMCOM, show_surface=False, show_humidity=False)

    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)
    
    
    
def intercompare_ice_water_content(AICE, CCLD, ACMCOM, ACMCAP,
                                  dstdir=None, hmax=20e3):

    cmap=colormaps.chiljet2()
    units = 'kg$~$m$^{-3}$'

    nrows=4
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.67})

    i=0
    plot_EC_2D(axes[i], AICE, 'ice_water_content', "IWC", scale_factor=1e-6,
              cmap=cmap, plot_scale=ACMCAP.ice_water_content.attrs['plot_scale'], 
              hmax=hmax, units=units,
              plot_range=ACMCAP.ice_water_content.attrs['plot_range'])

    i+=1
    plot_EC_2D(axes[i], CCLD, 'water_content', "IWC", 
              plot_where=CCLD.hydrometeor_classification.isin(CCLD_ice_classes), 
              cmap=cmap, plot_scale=ACMCAP.ice_water_content.attrs['plot_scale'], 
              hmax=hmax, units=units,
              plot_range=ACMCAP.ice_water_content.attrs['plot_range'])

    i+=1
    plot_EC_2D(axes[i], ACMCOM, 'ice_water_content', "IWC", 
              cmap=cmap, plot_scale=ACMCAP.ice_water_content.attrs['plot_scale'], 
              hmax=hmax, units=units,heightvar='height_layer',
              plot_range=ACMCAP.ice_water_content.attrs['plot_range'])

    i+=1
    plot_EC_2D(axes[i], ACMCAP, 'ice_water_content', "IWC", 
              cmap=cmap, plot_scale=ACMCAP.ice_water_content.attrs['plot_scale'], 
              hmax=hmax, units=units,
              plot_range=ACMCAP.ice_water_content.attrs['plot_range'])

    for ax in axes:
        add_extras(ax, CCLD, ACMCOM)
    
    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)


def intercompare_snow_rate(CCLD, ACMCAP, hmax=20e3):
    label = "S"
    cmap='chiljet2'
    units = 'mm$~$h$^{-1}$'
    plot_scale=ACMCAP.ice_mass_flux.attrs['plot_scale']
    plot_range=3600.*ACMCAP.ice_mass_flux.attrs['plot_range']

    nrows=2
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.67})

    i=0
    plot_EC_2D(axes[i], CCLD, 'mass_flux', label, scale_factor=3600.,
              plot_where=CCLD.hydrometeor_classification.isin(CCLD_ice_classes), 
              cmap=cmap, plot_scale=plot_scale, title_prefix="ice ",
              hmax=hmax, units=units,
              plot_range=plot_range)

    i+=1
    plot_EC_2D(axes[i], ACMCAP, 'ice_mass_flux', label, scale_factor=3600.,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    for ax in axes:
        add_extras(ax, CCLD, ACMCOM)
        
    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)

    
def intercompare_ice_effective_radius(AICE, ACMCOM, ACMCAP, CCLD,
                                      hmax=20e3):
    
    cmap='chiljet2'
    label = "$r_{eff}$"
    units = "$\mu$m"
    plot_scale=ACMCAP.ice_effective_radius.attrs['plot_scale']
    plot_range=1e6*ACMCAP.ice_effective_radius.attrs['plot_range']

    nrows=3
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.67})

    i=0
    plot_EC_2D(axes[i], AICE, 'ice_effective_radius', label,
              plot_where=AICE.ice_water_content > 0,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    i+=1
    plot_EC_2D(axes[i], ACMCOM, 'ice_effective_radius', label,
              plot_where=ACMCOM.ice_water_content > 0, scale_factor=1e6,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units, heightvar='height_layer',
              plot_range=plot_range)

    i+=1
    plot_EC_2D(axes[i], ACMCAP, 'ice_effective_radius', label,
              plot_where=ACMCAP.ice_water_content > 0, scale_factor=1e6,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    for ax in axes:
        add_extras(ax, CCLD, ACMCOM)
        
    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)

    
def intercompare_rain_water_content(CCLD, ACMCOM, ACMCAP,
                                    hmax=20e3):

    cmap='chiljet2'
    label="RWC"
    units = 'kg$~$m$^{-3}$'
    
    plot_scale=ACMCAP.rain_water_content.attrs['plot_scale']
    plot_range=ACMCAP.rain_water_content.attrs['plot_range']

    nrows=3
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.67})

    i=0
    plot_EC_2D(axes[i], CCLD, 'water_content', label,
              plot_where=CCLD.hydrometeor_classification.isin(CCLD_rain_classes), 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units, title_prefix="rain ",
              plot_range=plot_range)

    i+=1
    plot_EC_2D(axes[i], ACMCOM, 'rain_water_content', label,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,heightvar='height_layer',
              plot_range=plot_range)

    i+=1
    plot_EC_2D(axes[i], ACMCAP, 'rain_water_content', label,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    for ax in axes:
        add_extras(ax, CCLD, ACMCOM)
        
    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)

    
def intercompare_rain_rate(CCLD, ACMCAP, ACMCOM, hmax=20e3):
    cmap=colormaps.chiljet2()
    units = 'mm$~$h$^{-1}$'
    plot_scale=ACMCAP.rain_rate.attrs['plot_scale']
    plot_range=3600.*ACMCAP.rain_rate.attrs['plot_range']

    nrows=2
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.67})
    
    i=0
    plot_EC_2D(axes[i], CCLD, 'mass_flux', "R", scale_factor=3600.,
              plot_where=CCLD.hydrometeor_classification.isin(CCLD_rain_classes), 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units, title_prefix="rain ",
              plot_range=plot_range)
    
    i+=1
    plot_EC_2D(axes[i], ACMCAP, 'rain_rate', "R", scale_factor=3600.,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    for ax in axes:
        add_extras(ax, CCLD, ACMCOM)
        
    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)
    

    
def intercompare_liquid_water_content(CCLD, ACMCOM, ACMCAP, hmax=20e3):

    cmap=colormaps.chiljet2()
    label="LWC"
    units = 'kg$~$m$^{-3}$'
        
    plot_range=ACMCAP.liquid_water_content.attrs['plot_range']
    plot_scale=ACMCAP.liquid_water_content.attrs['plot_scale']

    nrows=3
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.67})

    i=0
    plot_EC_2D(axes[i], CCLD, 'liquid_water_content', label,
              cmap=cmap, plot_scale=plot_scale,
              hmax=hmax, units=units, 
              plot_range=plot_range)

    i+=1
    plot_EC_2D(axes[i], ACMCOM, 'liquid_water_content', label,
              cmap=cmap, plot_scale=plot_scale,
              hmax=hmax, units=units,heightvar='height_layer',
              plot_range=plot_range)

    i+=1
    plot_EC_2D(axes[i], ACMCAP, 'liquid_water_content', label,
              cmap=cmap, plot_scale=plot_scale,
              hmax=hmax, units=units,
              plot_range=plot_range)

    for ax in axes:
        add_extras(ax, CCLD, ACMCOM)
        
    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)

    
def intercompare_liquid_effective_radius(CCLD, ACMCOM, ACMCAP, hmax=20e3):
    
    cmap=colormaps.chiljet2()
    label = "$r_{eff}$"
    units = "$\mu$m"
    
    plot_scale=ACMCAP.liquid_effective_radius.attrs['plot_scale']
    plot_range=[0,20]

    nrows=2
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.67})

    i=0
    plot_EC_2D(axes[i], ACMCOM, 'liquid_effective_radius', label,
              plot_where=ACMCOM.liquid_water_content > 0, scale_factor=1e6,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units, heightvar='height_layer',
              plot_range=plot_range)

    i+=1
    plot_EC_2D(axes[i], ACMCAP, 'liquid_effective_radius', label,
              plot_where=ACMCAP.liquid_water_content > 0, scale_factor=1e6,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    for ax in axes:
        add_extras(ax, CCLD, ACMCOM)
        
    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)

    
def intercompare_aerosol_extinction(AAER, AEBD, ACMCOM, ACMCAP, CCLD, hmax=20e3):
    cmap=colormaps.chiljet2()
    label=r"$\alpha$"
    units = 'm$^{-1}$'
    
    plot_scale=ACMCAP.aerosol_extinction.attrs['plot_scale']
    plot_range=[1e-6,1e-3]

    nrows=4
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.67})

    i=0
    plot_EC_2D(axes[i], AAER, 'particle_extinction_coefficient_355nm', label,
              plot_where=AAER.classification.isin(AAER_aerosol_classes),
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    i+=1
    plot_EC_2D(axes[i], AEBD, 'particle_extinction_coefficient_355nm', label,
                plot_where=AEBD.simple_classification.isin([3]),
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    i+=1
    plot_EC_2D(axes[i], ACMCOM, 'aerosol_extinction', label,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units, 
              heightvar='height_layer',
              plot_range=plot_range)

    i+=1
    plot_EC_2D(axes[i], ACMCAP, 'aerosol_extinction', label,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    for ax in axes:
        add_extras(ax, CCLD, ACMCOM)
        
    
    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)
    

def intercompare_lidar_ratio(AAER, AEBD, ACMCAP, CCLD, hmax=20e3):
    cmap=colormaps.chiljet2()
    label=r"$s$"
    units = 'sr'
    
    plot_scale=ACMCAP.ATLID_bscat_extinction_ratio.attrs['plot_scale']
    plot_range=[0,100]

    #Inverting bscat-extinction ratio to get lidar ratio
    ACMCAP['ATLID_lidar_ratio'] = (1/ACMCAP.ATLID_bscat_extinction_ratio).where(ACMCAP.ATLID_bscat_extinction_ratio > 1e-9)
    ACMCAP['ATLID_lidar_ratio'].attrs['long_name'] = "forward-modelled ATLID extinction to backscatter ratio"
    
    nrows=3
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.67})

    i=0
    plot_EC_2D(axes[i], AAER, 'lidar_ratio_355nm', label,
              plot_where=AAER.classification > 0,
              #plot_where=AAER.classification.isin([10,11,12,13,14,15]),
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    i+=1
    plot_EC_2D(axes[i], AEBD, 'lidar_ratio_355nm', label,
              plot_where=AEBD.simple_classification > 0,
              #plot_where=AEBD.simple_classification.isin([3]),
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    i+=1
    plot_EC_2D(axes[i], ACMCAP, 'ATLID_lidar_ratio', label,
              plot_where=ACMCAP.ATLID_lidar_ratio > 0,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    for ax in axes:
        add_extras(ax, CCLD, ACMCOM)
        
    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)

    
def intercompare_rain_median_diameter(CCLD, ACMCAP, ACMCOM, hmax=20e3):
    
    cmap=colormaps.chiljet2()
    label = "$D_{0}$"
    units = "m"
    
    plot_scale=ACMCAP.rain_median_volume_diameter.attrs['plot_scale']
    plot_range=[1e-5,2e-3]

    nrows=2
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.67})


    i=0
    plot_EC_2D(axes[i], CCLD, 'characteristic_diameter', label,
              plot_where=CCLD.hydrometeor_classification.isin(CCLD_rain_classes),
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units, 
              plot_range=plot_range)

    i+=1
    plot_EC_2D(axes[i], ACMCAP, 'rain_median_volume_diameter', label,
              plot_where=ACMCAP.rain_water_content > 0, 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    for ax in axes:
        add_extras(ax, CCLD, ACMCOM)
        
    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)

    
def intercompare_ice_median_diameter(CCLD, ACMCAP, ACMCOM, hmax=20e3):
    
    cmap=colormaps.chiljet2()
    label = "$D_{0}$"
    units = "m"
    
    plot_scale=ACMCAP.ice_median_volume_diameter.attrs['plot_scale']
    plot_range=[1e-5,2e-3]

    nrows=2
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.67})


    i=0
    plot_EC_2D(axes[i], CCLD, 'characteristic_diameter', label,
              plot_where=CCLD.hydrometeor_classification.isin(CCLD_ice_classes),
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units, 
              plot_range=plot_range)

    i+=1
    plot_EC_2D(axes[i], ACMCAP, 'ice_median_volume_diameter', label,
              plot_where=ACMCAP.ice_water_content > 0, 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    for ax in axes:
        add_extras(ax, CCLD, ACMCOM)
        
    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)
    

    
    
def plot_ACMCAP_ice(ACMCAP, hmax=20e3, show_surface=True):
    
    cmap=colormaps.chiljet2

    nrows=9
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.67})

    
    i=0    
    var = 'ice_water_content'
    label = "IWC"
    units = "kg$~$m$^{-1}$"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=[1e-7,1e-2]
    plot_EC_2D(axes[i], ACMCAP, var, label, 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)
    
    i+=1    
    var = 'ice_mass_flux'
    label = "S"
    units = "mm$~$h$^{-1}$"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=[1e-4,2e1]
    plot_EC_2D(axes[i], ACMCAP, var, label, scale_factor=3600.,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    i+=1    
    var = 'ice_extinction'
    label = r"$\alpha$"
    units = "m$^{-1}$"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=[1e-5,1e-1]
    plot_EC_2D(axes[i], ACMCAP, var, label, scale_factor=1.,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)
    
    i+=1    
    var = 'ice_N0prime'
    label = "N'$_{0}$"
    units = "m$^{3.67}$"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=[1e4,1e15]
    plot_EC_2D(axes[i], ACMCAP, var, label, scale_factor=1.,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    i+=1    
    var = 'ice_normalized_number_concentration'
    label = "N*$_{0}$"
    units = "m$^{-4}$"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=[1e1,1e12]
    plot_EC_2D(axes[i], ACMCAP, var, label, scale_factor=1.,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)
    
    i+=1
    var = 'ice_effective_radius'
    label = "r$_{eff}$"
    units = "$\mu$m"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=[0,120]
    plot_EC_2D(axes[i], ACMCAP, var, label,
              plot_where=ACMCAP.ice_water_content > 0, scale_factor=1e6,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)
    
    i+=1
    var = 'ice_median_volume_diameter'
    label = "$D_{0}$"
    units = ACMCAP[var].attrs['units']
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=[1e-5,1e-2]
    plot_EC_2D(axes[i], ACMCAP, var, label,
              plot_where=ACMCAP.ice_water_content > 0, 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)
    
    i+=1
    var = 'ice_lidar_bscat_extinction_ratio'
    label = "$s$"
    units = "$sr^{-1}$"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=ACMCAP[var].attrs['plot_range']
    plot_EC_2D(axes[i], ACMCAP, var, label,
              plot_where=ACMCAP.ice_water_content > 0, 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)
    
    i+=1
    var = 'ice_riming_factor'
    label = "r"
    units = ACMCAP[var].attrs['units']
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=[-0.15, 1.0]
    litmus_riming = colormaps.define_truncated_cm('litmus', v=[-0.15,0,1.0], suffix='doppler')
    plot_EC_2D(axes[i], ACMCAP, var, label,
              plot_where=ACMCAP.ice_water_content > 0, 
              cmap=litmus_riming, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)
        
    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)

    if show_surface:
        for ax in axes:
            add_surface(ax, ACMCAP, elevation_var='elevation')

    return fig, axes


def plot_ACMCAP_rain(ACMCAP, hmax=20e3, show_surface=True):
    
    cmap=colormaps.chiljet2

    nrows=11
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.67})

    i=0 
    var = 'CPR_reflectivity_factor'
    label = "Z"
    units = "dBZ"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=ACMCAP[var].attrs['plot_range']
    plot_EC_2D(axes[i], ACMCAP, var, label, 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    i+=1
    var = 'CPR_reflectivity_factor_forward'
    label = "Z"
    units = "dBZ"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=ACMCAP[var].attrs['plot_range']
    plot_EC_2D(axes[i], ACMCAP, var, label, 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    i+=1 
    var = 'CPR_doppler_velocity'
    label = "V$_D$"
    units = "ms$^{-1}$"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=[-6,6]
    plot_EC_2D(axes[i], ACMCAP, var, label, 
              cmap='litmus', plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    i+=1 
    var = 'CPR_doppler_velocity_forward'
    label = "V$_D$"
    units = "ms$^{-1}$"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=[-6,6]
    plot_EC_2D(axes[i], ACMCAP, var, label, 
              cmap='litmus', plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    i+=1
    var = 'rain_water_content'
    label = "RWC"
    units = "kg$~$m$^{-1}$"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=ACMCAP[var].attrs['plot_range']
    plot_EC_2D(axes[i], ACMCAP, var, label, 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)
    
    i+=1    
    var = 'rain_rate'
    label = "R"
    units = "mm$~$h$^{-1}$"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=[1e-4,2e1]
    plot_EC_2D(axes[i], ACMCAP, var, label, scale_factor=3600.,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)


    i+=1    
    var = 'rain_number_concentration_scaling'
    label = "N_s"
    units = "-"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=[0.01,100]
    plot_EC_2D(axes[i], ACMCAP, var, label, scale_factor=1.,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)
    
    i+=1    
    var = 'rain_normalized_number_concentration'
    label = "N*$_{0}$"
    units = "m$^{-4}$"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=[1e4,1e12]
    plot_EC_2D(axes[i], ACMCAP, var, label, scale_factor=1.,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    i+=1
    var = 'rain_median_volume_diameter'
    label = "$D_{0}$"
    units = ACMCAP[var].attrs['units']
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=[1e-5,1e-2]
    plot_EC_2D(axes[i], ACMCAP, var, label,
              plot_where=ACMCAP.rain_water_content > 0, 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    i+=1
    var = 'liquid_water_content'
    label = "LWC"
    units = "kg$~$m$^{-1}$"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=ACMCAP[var].attrs['plot_range']
    plot_EC_2D(axes[i], ACMCAP, var, label, 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)

    i+=1
    var = 'liquid_number_concentration'
    label = "N"
    units = "m$^{-3}$"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=ACMCAP[var].attrs['plot_range']
    plot_EC_2D(axes[i], ACMCAP, var, label, 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)
    
    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)
    
    if show_surface:
        for ax in axes:
            add_surface(ax, ACMCAP, elevation_var='elevation')

    return fig, axes
    

def plot_ACMCAP_example(ACMCAP, hmax=20e3, show_surface=True):
    
    cmap=colormaps.chiljet2()

    nrows=7
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.67})
    
    
    i=0    
    var = 'ice_water_content'
    label = "IWC"
    units = "kg$~$m$^{-3}$"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=ACMCAP[var].attrs['plot_range']
    plot_EC_2D(axes[i], ACMCAP, var, label, 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)
    
    i+=1
    var = 'ice_effective_radius'
    label = "r$_{eff}$"
    units = "$\mu$m"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=[0,120]
    plot_EC_2D(axes[i], ACMCAP, var, label,
              plot_where=ACMCAP.ice_water_content > 0, scale_factor=1e6,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)
    
    i+=1
    var = 'ice_extinction'
    label = r"$\alpha$"
    units = "m$^{-1}$"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=ACMCAP[var].attrs['plot_range']
    plot_EC_2D(axes[i], ACMCAP, var, label,
              plot_where=ACMCAP.ice_water_content > 0, 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)
    
    i+=1
    var = 'liquid_water_content'
    label = "LWC"
    units = "kg$~$m$^{-3}$"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=ACMCAP[var].attrs['plot_range']
    plot_EC_2D(axes[i], ACMCAP, var, label, 
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)
    
    i+=1
    var = 'liquid_effective_radius'
    label = "r$_{eff}$"
    units = "$\mu$m"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=[0,30]
    plot_EC_2D(axes[i], ACMCAP, var, label,
              plot_where=ACMCAP.liquid_water_content > 0, scale_factor=1e6,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)
    
    i+=1
    var = 'rain_rate'
    label = "R"
    units = "mm$~$h$^{-1}$"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=[1e-3,1e1]
    plot_EC_2D(axes[i], ACMCAP, var, label, 
              cmap=cmap, plot_scale=plot_scale, scale_factor=3600.,
              hmax=hmax, units=units,
              plot_range=plot_range)
    
    i+=1
    var = 'aerosol_extinction'
    label = r"$\alpha$"
    units = "m$^{-1}$"
    plot_scale=ACMCAP[var].attrs['plot_scale']
    plot_range=[1e-6,1e-3]
    plot_EC_2D(axes[i], ACMCAP, var, label,
              cmap=cmap, plot_scale=plot_scale, 
              hmax=hmax, units=units,
              plot_range=plot_range)
        
    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)


    if show_surface:
        for ax in axes:
            add_surface(ax, ACMCAP, elevation_var='elevation')

    

##############################################################################################################################
# Level 1
##############################################################################################################################

# plot VNS bands
def plot_ECL1_MSI_VNS(MRGR, reflectance = False, use_localtime=True):
    from . import ecio
    if reflectance:
        sza,vza,raa,sca = ecio.obs_angle_conv(MRGR['solar_elevation_angle'].values,MRGR['sensor_elevation_angle'].values,
                                         MRGR['solar_azimuth_angle'].values,  MRGR['sensor_azimuth_angle'].values)
        vns_reflectance = ecio.specRad2Refl(MRGR['pixel_values'].values,MRGR['solar_spectral_irradiance'].values,sza)
        MRGR['VIS'  ].values = vns_reflectance[0]
        MRGR['NIR'  ].values = vns_reflectance[1]
        MRGR['SWIR1'].values = vns_reflectance[2]
        MRGR['SWIR2'].values = vns_reflectance[3]
        title = 'MSI-RGR reflectance (VNS)'
        units = '-'
        plot_range   = [[0,1],[0,1],[0,1],[0,1]]
    else:
        title = "MSI-RGR radiance (VNS)"
        units = 'Wm$^{-2}$sr$^{-1}\mu$m$^{-1}$'
        plot_range   = [[0,200],[0,150],[0,40],[0,10]]
    
    
    varname      = ['VIS','NIR','SWIR1','SWIR2']
    cbarlabel    = varname
    cmap         = cmap_grey_r
    plot_scale   = 'linear'
    ymax         = 383 # hmax = yaxis upper limit (ymax)
    dx           = 1000
    d0           = 200
    x0           = 200
    across_track = True
    heightvar    = 'across_track'
    latvar       = 'selected_latitude'
    lonvar       = 'selected_longitude'
    ruler_y0     = 0.75
    
    nrows=4
    fig, axes = plt.subplots(figsize=(25,4.5*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':1.5})
    
    for i in range(4):
        plot_EC_2D(axes[i], MRGR, varname[i], cbarlabel[i], units=units, title=title,
                  cmap=cmap, plot_scale=plot_scale, hmax=ymax, plot_range=plot_range[i],
                  heightvar=heightvar, latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime)
        add_land_sea_border(axes[i], MRGR, col='khaki')
        add_ruler(axes[i], MRGR, timevar='time', dx=dx, d0=d0, x0=x0, pixel_scale_km=0.5, y0=ruler_y0, dark_mode=False)
    
    for ax in axes:
        add_nadir_track(ax)
        
    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)
    
    return fig, axes
    
    
# plot TIR bands
def plot_ECL1_MSI_TIR(MRGR, btdiff=False, use_localtime=True):
    
    if btdiff:
        MRGR = MRGR.assign(TIR2minusTIR1 = MRGR["pixel_values"].isel({'band':5}) - MRGR["pixel_values"].isel({'band':4}),
                           TIR2minusTIR3 = MRGR["pixel_values"].isel({'band':5}) - MRGR["pixel_values"].isel({'band':6}),
                           TIR3minusTIR1 = MRGR["pixel_values"].isel({'band':6}) - MRGR["pixel_values"].isel({'band':4}))
        
        title      = 'MSI-RGR BT difference (TIR)'
        plot_range = [[-6,6]                           ,[-7,7]                            ,[-5,5]                           ]
        varname    = [ 'TIR2minusTIR1'                 , 'TIR2minusTIR3'                  , 'TIR3minusTIR1'                 ]
        cbarlabel  = [r'BT$_{10.8 \mu m}$ - BT$_{8.8 \mu m}$',r'BT$_{10.8 \mu m}$ - BT$_{12.0 \mu m}$',r'BT$_{12.0 \mu m}$ - B$_{8.8 \mu m}$']
        cmap       = cmap_smc
        lmcol      = 'black'
    else:
        title      = 'MSI-RGR brightness temperature (TIR)'
        plot_range = [[210,310]       ,[210,310]        ,[210,310]        ]
        varname    = [ 'TIR1'         , 'TIR2'          , 'TIR3'          ]
        cbarlabel  = [r"BT$_{8.8 \mu m}$",r"BT$_{10.8 \mu m}$",r"BT$_{12.0 \mu m}$"]
        cmap       = cmap_grey
        lmcol      = 'khaki'
    
    units        = '$K$'
    plot_scale   = 'linear'
    ymax         = 383 # hmax = yaxis upper limit (ymax)
    dx           = 1000
    d0           = 200
    x0           = 200
    across_track = True
    heightvar    = 'across_track'
    latvar       = 'selected_latitude'
    lonvar       = 'selected_longitude'
    ruler_y0     = 0.75
    
    nrows=3
    fig, axes = plt.subplots(figsize=(25,4.5*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':1.5})
    
    for i in range(3):
        plot_EC_2D(axes[i], MRGR, varname[i], cbarlabel[i], units=units, title=title, 
                  cmap=cmap, plot_scale=plot_scale, hmax=ymax, plot_range=plot_range[i],
                  heightvar=heightvar, latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime)
        add_land_sea_border(axes[i], MRGR, col=lmcol)
        add_ruler(axes[i], MRGR, timevar='time', dx=dx, d0=d0, x0=x0, pixel_scale_km=0.5, y0=ruler_y0, dark_mode=False)
    
    for ax in axes:
        add_nadir_track(ax)
        
    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)    

    return fig, axes

# plot obs. angles
def plot_ECL1_MSI_extras(MRGR, use_localtime=True):
    
    cmap         = cmap_rnbw
    units        = '$^{\circ}$'
    plot_scale   = 'linear'
    ymax         = 383 # hmax = yaxis upper limit (ymax)
    dx           = 1000
    d0           = 200
    x0           = 200
    plot_range   = [[-10,90],[70,90],[0,360],[0,360]]
    across_track = True
    heightvar    = 'across_track'
    latvar       = 'selected_latitude'
    lonvar       = 'selected_longitude'
    ruler_y0 = 0.75
    
    nrows=4
    fig, axes = plt.subplots(figsize=(25,4.5*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':1.5})
    
    i=0
    plot_EC_2D(axes[i], MRGR, 'solar_elevation_angle', "sea", units=units, 
              cmap=cmap, plot_scale=plot_scale, hmax=ymax, plot_range=plot_range[0],
              heightvar=heightvar, latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime)
    add_ruler(axes[i], MRGR, timevar='time', dx=dx, d0=d0, x0=x0,pixel_scale_km=0.5, y0=ruler_y0, dark_mode=False)
    
    i=1
    plot_EC_2D(axes[i], MRGR, 'sensor_elevation_angle', "vea", units=units, 
              cmap=cmap, plot_scale=plot_scale, hmax=ymax, plot_range=plot_range[1],
              heightvar=heightvar, latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime)
    add_ruler(axes[i], MRGR, timevar='time', dx=dx, d0=d0, x0=x0,pixel_scale_km=0.5, y0=ruler_y0, dark_mode=False)
    
    i=2
    plot_EC_2D(axes[i], MRGR, 'solar_azimuth_angle', "saa", units=units, 
              cmap=cmap, plot_scale=plot_scale, hmax=ymax, plot_range=plot_range[2],
              heightvar=heightvar, latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime)
    add_ruler(axes[i], MRGR, timevar='time', dx=dx, d0=d0, x0=x0,pixel_scale_km=0.5, y0=ruler_y0,dark_mode=False)
    
    i=3
    plot_EC_2D(axes[i], MRGR, 'sensor_azimuth_angle', "vaa", units=units, 
              cmap=cmap, plot_scale=plot_scale, hmax=ymax, plot_range=plot_range[3],
              heightvar=heightvar, latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime)
    add_ruler(axes[i], MRGR, timevar='time', dx=dx, d0=d0, x0=x0, pixel_scale_km=0.5, y0=ruler_y0,dark_mode=False)
    
    for ax in axes:
        add_nadir_track(ax)
        
    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)   
    
    return fig, axes 

def calculate_RGB_MRGR(MRGR, select=slice(None)):
    
    MRGR['along_track'] = MRGR.along_track

    Rmin, Gmin, Bmin = [MRGR.SWIR1.where(MRGR.SWIR1 < 1e36).isel(along_track=select).quantile(0.01),  
                        MRGR.NIR.where(MRGR.NIR < 1e36).isel(along_track=select).quantile(0.01),
                       MRGR.VIS.where(MRGR.VIS < 1e36).isel(along_track=select).quantile(0.01)]
    Rmin, Gmin, Bmin = Rmin if Rmin > 0 else 0.0, Gmin if Gmin > 0 else 0.0, Bmin if Bmin > 0 else 0.0
    
    Rw, Gw, Bw = [0.9,1.0,1.0]
    Rs, Gs, Bs = [1,1,1]
    
    Rmax, Gmax, Bmax = [MRGR.SWIR1.where(MRGR.SWIR1 < 1e36).isel(along_track=select).quantile(0.99),  
                        MRGR.NIR.where(MRGR.NIR < 1e36).isel(along_track=select).quantile(0.99),
                        MRGR.VIS.where(MRGR.VIS < 1e36).isel(along_track=select).quantile(0.99)]
    Rmax, Gmax, Bmax = Rmax if Rmax > 1.0 else 1.0, Gmax if Gmax > 1.0 else 1.0, Bmax if Bmax > 1.0 else 1.0
    
    RGB = xr.concat([Rw*(MRGR.SWIR1.where(MRGR.SWIR1 < 1e36) - Rmin)/(Rs*(Rmax-Rmin)), 
                     Gw*(MRGR.NIR.where(MRGR.NIR < 1e36) - Gmin)/(Gs*(Gmax-Gmin)), 
                     Bw*(MRGR.VIS.where(MRGR.VIS < 1e36) - Bmin)/(Bs*(Bmax-Bmin))], 
                    dim=xr.IndexVariable('band', [0,1,2])).isel(along_track=select)
    RGB['time'] = MRGR.isel(along_track=select).time

    return RGB


def plot_RGB_MRGR(ax, RGB, MRGR, select=slice(None), use_localtime=True, short_timestep=False):
    RGB.swap_dims({'along_track':'time'}).plot.imshow(ax=ax, x='time', y='across_track', rgb='band', yincrease=False)
    format_plot(ax, MRGR.isel(along_track=select), "M-RGR SWIR-NIR-VIS natural colour image", RGB.across_track.max(), 
                          across_track=True, use_localtime=use_localtime, 
                          lonvar='selected_longitude', latvar='selected_latitude',
               short_timestep=short_timestep)
    ax.set_ylim(len(MRGR.across_track)-15, 12)

    
def plot_TIR_MRGR(ax, MRGR, select=slice(None), use_localtime=True, tmin=190, tmax=320, cmap='LW', short_timestep=False):
    _MRGR = MRGR.isel(along_track=select).swap_dims({'along_track':'time'})
    _cm = _MRGR.TIR1.where(_MRGR.TIR1 < 1e36).plot.imshow(ax=ax, x='time', y='across_track', 
                                                           cmap=cmap, norm=Normalize(tmin,tmax), add_colorbar=False, yincrease=False)
    format_plot(ax, MRGR, "M-RGR thermal infrared", MRGR.across_track.max(), 
                          across_track=True, use_localtime=use_localtime, 
                          lonvar='selected_longitude', latvar='selected_latitude', 
               short_timestep=short_timestep)
    ax.set_ylim(len(MRGR.across_track)-15,12)
    cb = add_colorbar(ax, _cm, r"BT$_{8.5\mu\mathrm{m}}$ [K]", horz_buffer=0.01, width_ratio='1%')
    cb.set_ticks([190,230,270,310])
    cb.ax.invert_yaxis()

    
def format_MRGR(ax, MRGR):
    dx           = 1000
    d0           = 200
    x0           = 200
    ruler_y0     = 0.75
    lmcol        = 'khaki'
    
    add_land_sea_border(ax, MRGR, col=lmcol)
    add_nadir_track(ax, label_below=False)


def quicklook_MRGR(MRGR, select=slice(None,None), dh=3, hspace=2.5, use_localtime=True, with_marble=False):

    RGB = calculate_RGB_MRGR(MRGR, select=select)
    
    nrows=2
    fig, axes = plt.subplots(figsize=(25,nrows*dh), nrows=nrows, gridspec_kw={'hspace':hspace})
    
    plot_RGB_MRGR(axes[0], RGB, MRGR, select=select, use_localtime=use_localtime)
    format_MRGR(axes[0], MRGR)
    
    plot_TIR_MRGR(axes[1], MRGR, select=select, use_localtime=use_localtime)
    format_MRGR(axes[1], MRGR)
            
    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)    

    if with_marble:
        add_marble(axes[0], MRGR, timevar='time', lonvar='selected_longitude', latvar='selected_latitude')

    return fig, axes

##############################################################################################################################
# Level 2a
##############################################################################################################################

# plot M-AOT
def plot_MAOT_example(MAOT, MRGR=None, use_localtime=True):
    
    units        = '$1$'
    plot_scale   = 'linear'
    ymax         = 383 # hmax = yaxis upper limit (ymax)
    dx           = 1000
    d0           = 200
    x0           = 200
    
    precision = 2
    maxi=np.true_divide(np.ceil(np.max(np.array([np.nanquantile(MAOT['aerosol_optical_thickness_670nm'].values,0.99),
                                                 np.nanquantile(MAOT['aerosol_optical_thickness_865nm'].values,0.99)]))* 10**precision) \
                                , 10**precision )
    plot_range   = [[0,maxi],[0,maxi],[0,4]]
    across_track = True
    heightvar    = 'across_track'
    latvar       = 'selected_latitude'
    lonvar       = 'selected_longitude'
    ruler_y0 = 0.85
    
    nrows=3
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.67})
    
    i=0
    plot_EC_2D(axes[i], MAOT, 'aerosol_optical_thickness_670nm', "AOT (670 nm)", units=units, 
              cmap=cmap_org, plot_scale=plot_scale, hmax=ymax, plot_range=plot_range[0],
              heightvar=heightvar, latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime)
    if MRGR is not None: add_land_sea_border(axes[i], MRGR)
    add_ruler(axes[i], MAOT, dx, d0, x0, pixel_scale_km=0.5, y0=ruler_y0, dark_mode=False)
    
    i=1
    plot_EC_2D(axes[i], MAOT, 'aerosol_optical_thickness_865nm', "AOT (865 nm)", units=units, 
              cmap=cmap_org, plot_scale=plot_scale, hmax=ymax, plot_range=plot_range[1],
              heightvar=heightvar, latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime)
    if MRGR is not None: add_land_sea_border(axes[i], MRGR)
    add_ruler(axes[i], MAOT, dx, d0, x0, pixel_scale_km=0.5, y0=ruler_y0,dark_mode=False)
    
    i=2
    plot_EC_target_classification(axes[i], MAOT, 'quality_status', 
                                    MAOT_qstat_category_colors, hmax=ymax, title_prefix="", label_fontsize=10, \
                                    heightvar=heightvar, latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime, line_break=35)
    if MRGR is not None: add_land_sea_border(axes[i], MRGR)
    add_ruler(axes[i], MAOT, dx, d0, x0, pixel_scale_km=0.5, y0=ruler_y0,dark_mode=False)
    
    for ax in axes:
        add_nadir_track(ax)
        
    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)
 
    return fig, axes
      
# plot M-COP
def plot_MCOP_example(MCOP, MRGR=None, use_localtime=True):
    
    plot_scale   = 'linear'
    ymax         = 383 # hmax = yaxis upper limit (ymax)
    dx           = 1000
    d0           = 200
    x0           = 200
    
    plot_range   = [[0,255],[0,30],[0,2000],[200,1030],[150,300],[0,20]]
    across_track = True
    heightvar    = 'across_track'
    latvar       = 'selected_latitude'
    lonvar       = 'selected_longitude'
    cmap         = colormaps.chiljet2()# cmap_org
    ruler_y0 = 0.85
    
    nrows=7
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.75})
    
    i=0
    plot_EC_2D(axes[i], MCOP, 'cloud_optical_thickness', "M-COT", units='$1$', 
              cmap=cmap, plot_scale=plot_scale, hmax=ymax, plot_range=plot_range[i],
              heightvar=heightvar, latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime)
    if MRGR is not None: add_land_sea_border(axes[i], MRGR)
    add_ruler(axes[i], MCOP, dx, d0, x0, pixel_scale_km=0.5, y0=ruler_y0,dark_mode=False)
    
    i=1
    plot_EC_2D(axes[i], MCOP, 'cloud_effective_radius', "M-Reff", units='$\mu$m', 
              cmap=cmap, plot_scale=plot_scale, scale_factor=1e6, hmax=ymax, plot_range=plot_range[i],
              heightvar=heightvar, latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime)
    if MRGR is not None: add_land_sea_border(axes[i], MRGR)     
    add_ruler(axes[i], MCOP, dx, d0, x0, pixel_scale_km=0.5, y0=ruler_y0,dark_mode=False)
    
    i=2
    plot_EC_2D(axes[i], MCOP, 'cloud_water_path', "M-CWP", units='$g m^{-2}$', 
              cmap=cmap, plot_scale=plot_scale, scale_factor=1e3, hmax=ymax, plot_range=plot_range[i],
              heightvar=heightvar, latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime)
    if MRGR is not None: add_land_sea_border(axes[i], MRGR)
    add_ruler(axes[i], MCOP, dx, d0, x0, pixel_scale_km=0.5, y0=ruler_y0,dark_mode=False)
    
    i=3
    plot_EC_2D(axes[i], MCOP, 'cloud_top_pressure', "M-CTP", units='$hPa$', 
              cmap=cmap, plot_scale=plot_scale, scale_factor=1e-2, hmax=ymax, plot_range=plot_range[i],
              heightvar=heightvar, latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime)
    if MRGR is not None: add_land_sea_border(axes[i], MRGR)     
    add_ruler(axes[i], MCOP, dx, d0, x0, pixel_scale_km=0.5, y0=ruler_y0,dark_mode=False)
    
    i=4
    plot_EC_2D(axes[i], MCOP, 'cloud_top_temperature', "M-CTT", units='$K$', 
              cmap=cmap, plot_scale=plot_scale, hmax=ymax, plot_range=plot_range[i],
              heightvar=heightvar, latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime)
    if MRGR is not None: add_land_sea_border(axes[i], MRGR)
    add_ruler(axes[i], MCOP, dx, d0, x0, pixel_scale_km=0.5, y0=ruler_y0,dark_mode=False)
    
    i=5
    plot_EC_2D(axes[i], MCOP, 'cloud_top_height', "M-CTH", units='$km$', 
              cmap=cmap, plot_scale=plot_scale, scale_factor=1e-3, hmax=ymax, plot_range=plot_range[i],
              heightvar=heightvar, latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime)
    if MRGR is not None: add_land_sea_border(axes[i], MRGR)     
    add_ruler(axes[i], MCOP, dx, d0, x0, pixel_scale_km=0.5, y0=ruler_y0,dark_mode=False)
    
    i=6
    plot_EC_target_classification(axes[i], MCOP, 'quality_status', 
                                    MCOP_qstat_category_colors, hmax=ymax, title_prefix="", label_fontsize=10, \
                                    heightvar=heightvar, latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime, line_break=30)
    if MRGR is not None: add_land_sea_border(axes[i], MRGR)
    add_ruler(axes[i], MCOP, dx, d0, x0, pixel_scale_km=0.5, y0=ruler_y0,dark_mode=False)
    
    for ax in axes:
        add_nadir_track(ax)
        
    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)
  
    return fig, axes


# plot M-CM
def plot_MCM_example(MCM, MRGR=None, use_localtime=True):
    
    plot_scale   = 'linear'
    ymax         = 383 # hmax = yaxis upper limit (ymax)
    dx           = 1000
    d0           = 200
    x0           = 200
    
    across_track = True
    heightvar    = 'across_track'
    latvar       = 'selected_latitude'
    lonvar       = 'selected_longitude'
    
    ruler_y0 = 0.85
    
    nrows=6
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, 
                             sharex=False, sharey=False, gridspec_kw={'hspace':0.75})
    
    i=0
    plot_EC_target_classification(axes[i], MCM, 'cloud_mask', 
                                    MCM_maskphase_category_colors, hmax=ymax, title_prefix="", label_fontsize=10, \
                                    heightvar=heightvar, latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime, \
                                    fillna=-127, line_break=30)
    if MRGR is not None: add_land_sea_border(axes[i], MRGR)
    add_ruler(axes[i], MCM, dx, d0, x0, pixel_scale_km=0.5, y0=ruler_y0,dark_mode=False)
    
    i=1
    plot_EC_target_classification(axes[i], MCM, 'cloud_mask_quality_status', 
                                    MCM_qstat_category_colors, hmax=ymax, title_prefix="", label_fontsize=10, \
                                    heightvar=heightvar, latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime, line_break=30)
    if MRGR is not None: add_land_sea_border(axes[i], MRGR)
    add_ruler(axes[i], MCM, dx, d0, x0, pixel_scale_km=0.5, y0=ruler_y0,dark_mode=False)
    
    i=2
    plot_EC_target_classification(axes[i], MCM, 'cloud_phase', 
                                    MCM_maskphase_category_colors, hmax=ymax, title_prefix="", label_fontsize=10, \
                                    heightvar=heightvar, latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime, \
                                    fillna=-127, line_break=30)
    if MRGR is not None: add_land_sea_border(axes[i], MRGR)
    add_ruler(axes[i], MCM, dx, d0, x0, pixel_scale_km=0.5, y0=ruler_y0,dark_mode=False)
    
    i=3
    plot_EC_target_classification(axes[i], MCM, 'cloud_phase_quality_status', 
                                    MCM_qstat_category_colors, hmax=ymax, title_prefix="", label_fontsize=10, \
                                    heightvar=heightvar, latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime, line_break=30)
    if MRGR is not None: add_land_sea_border(axes[i], MRGR)
    add_ruler(axes[i], MCM, dx, d0, x0, pixel_scale_km=0.5, y0=ruler_y0,dark_mode=False)
    
    i=4
    plot_EC_target_classification(axes[i], MCM, 'cloud_type', 
                                    MCM_type_category_colors, hmax=ymax, title_prefix="", label_fontsize=10, \
                                    heightvar=heightvar, latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime, \
                                    fillna=-127, line_break=30)
    if MRGR is not None: add_land_sea_border(axes[i], MRGR)
    add_ruler(axes[i], MCM, dx, d0, x0, pixel_scale_km=0.5, y0=ruler_y0,dark_mode=False)
    
    i=5
    plot_EC_target_classification(axes[i], MCM, 'cloud_type_quality_status', 
                                    MCM_qstat_category_colors, hmax=ymax, title_prefix="", label_fontsize=10, \
                                    heightvar=heightvar, latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime, line_break=30)
    if MRGR is not None: add_land_sea_border(axes[i], MRGR)
    add_ruler(axes[i], MCM, dx, d0, x0, pixel_scale_km=0.5, y0=ruler_y0,dark_mode=False)
    
    for ax in axes:
        add_nadir_track(ax)
        
    add_subfigure_labels(axes, yloc=1.2)
    snap_xlims(axes)   
    
    return fig, axes
### end ---


def format_latlon_ticks(ax, _ax, ds, timevar, lonvar, latvar, dim_name):
    
    #Trim to frame
    get_frame_edge = lambda n: min([67.5,22.5,-22.5,-67.5], key=lambda x:abs(x-n))

    is_polar = np.diff(ds[latvar][[0,-1]].values) < 1
    is_nh = (ds[latvar].mean() > 0)
    
    nticks = 9
    
    format_lat = lambda lat: "${:.1f}^\circ$S".format(-1*lat) if lat < 0 else "${:.1f}^\circ$N".format(lat)
    format_lon = lambda lon: "${:.1f}^\circ$W".format(-1*lon) if lon < 0 else "${:.1f}^\circ$E".format(lon)
    
    _ds = ds.set_coords(timevar).swap_dims({dim_name:timevar})
    
    #To the pole
    time_ticks = pd.date_range(ds[timevar][0].values, ds[timevar][-1].values, periods=1*nticks+1)
    lat_ticks = _ds[latvar].sel({timevar:time_ticks}, method='nearest').values
    _ax.set_xticks(time_ticks)

    #Lat/lon ticks snap to frame boundaries, then intervals of 5deg latitude
    time_ticks_minor = pd.date_range(ds[timevar][0].values, ds[timevar][-1].values, periods=5*nticks+1)
    lat_ticks_minor = _ds[latvar].sel({timevar:time_ticks_minor}, method='nearest').values
    _ax.set_xticks(time_ticks_minor, minor=True)

    #Nice formatting for coordinates
    lon_ticks = _ds[lonvar].sel({timevar:time_ticks}, method='nearest').values
    
    latlon_ticks = ["" + format_lat(ll[0]) + '\n' + format_lon(ll[1]) for ll in zip(lat_ticks, lon_ticks)]
    _ax.set_xticklabels(latlon_ticks, fontsize='xx-small', color='0.5')
    _ax.tick_params(axis='x', which='both', color='0.5')

    ax.set_xlim(ds[timevar][0], ds[timevar][-1])
    _ax.set_xlim(ds[timevar][0], ds[timevar][-1])


def format_time_ticks(ax, ds, timevar, lonvar, dim_name, major_step='60s', minor_step='15s', use_localtime=True):
    if use_localtime:
        localtime = ds[timevar] + [np.timedelta64(int(l), 's') for l in np.round(ds[lonvar].values[:]/15*60*60)]

        if major_step < '60s':
            #Major ticks
            time_ticks = pd.date_range(ds[timevar].to_index().ceil(major_step)[0], ds[timevar].to_index().floor(major_step)[-1], freq=major_step)
            ax.set_xticks(time_ticks)
            ax.set_xticklabels([f"{t:%d %H:%M:%S}" for t in time_ticks])
        else:
            #Major ticks at 3-minute intervals
            time_ticks = pd.date_range(ds[timevar].to_index().ceil(major_step)[0], ds[timevar].to_index().floor(major_step)[-1], freq=major_step)
            ax.set_xticks(time_ticks)
            ax.set_xticklabels([f"{t:%d %H:%M}" for t in time_ticks])
        xticks = [t for t in ax.get_xticklabels()]

        ds[dim_name] = ds[dim_name]
        time_idx = [np.argmin(np.abs(ds[timevar].values - np.datetime64(f"{pd.to_datetime(time_ticks[0]):%Y-%m}-{t.get_text()}"))) for t in xticks if t.get_text() != '']
        
        xticks_time = time_ticks
        xticks_localtime = localtime[time_idx].values
        if major_step < '60s':
            xticklabels = [f"{pd.to_datetime(t):%H:%M:%S}\n{pd.to_datetime(xticks_localtime[i]):%H:%M:%S}" for i,t in enumerate(xticks_time)]       
        else:
            xticklabels = [f"{pd.to_datetime(t):%H:%M}\n{pd.to_datetime(xticks_localtime[i]):%H:%M}" for i,t in enumerate(xticks_time)]       
        ax.set_xticklabels(xticklabels, fontsize='x-small')

        #Minor ticks at 1-minute intervals
        time_ticks_minor = pd.date_range(ds[timevar].to_index().round(minor_step)[0], ds[timevar].to_index().round(minor_step)[-1], freq=minor_step)
        ax.set_xticks(time_ticks_minor, minor=True)

        frame = ds.encoding['source'].split("/")[-1].split(".")[0].split("_")[-1]
        if len(xticks_time) > 0:
            ax.set_xlabel(f"Time, {pd.to_datetime(xticks_time[0]):%Y-%m-%d}", 
                          fontsize='small', loc='center')
        ax.set_title(f"frame {frame}", loc='right', fontsize='small')

        xtickslabels = ax.get_xticklabels()
        
        #Adding UTC/LST to the final time tick
        if len(xtickslabels) > 0:
            _l = xticklabels[-1].split('\n')
            xticklabels[-1] = f"        {_l[0]} (UTC)\n        {_l[1]} (LST)"
        
        ax.set_xticklabels(xticklabels, fontsize='x-small')

    else:
        #Major ticks at 3-minute intervals
        time_ticks = pd.date_range(ds[timevar].to_index().round(major_step)[0], ds[timevar].to_index().round(major_step)[-1], freq=major_step)
        ax.set_xticks(time_ticks)

        #Minor ticks at 1-minute intervals
        time_ticks_minor = pd.date_range(ds[timevar].to_index().round(minor_step)[0], ds[timevar].to_index().round(minor_step)[-1], freq=minor_step)
        ax.set_xticks(time_ticks_minor, minor=True)

        #Nice formatting for time
        frame = ds.encoding['source'].split("/")[-1].split(".")[0].split("_")[-1]
        format_time(ax, format_string="%H:%M", label=f"Time (UTC) {pd.to_datetime(ds[timevar][0].values):%Y-%m-%d}")
        ax.set_title(f"frame {frame}", loc='right', fontsize='small')

        
        
###For 1D/scalar timeseries plots

def format_plot_1D(ax, ds, title, use_latitude=False, dark_mode=False, 
                timevar='time', latvar='latitude', lonvar='longitude', 
                dim_name='along_track', use_localtime=True, y0_ruler=0.9, 
                include_ruler=True):
    
    #Set title
    ax.set_title(title)

    format_time_ticks(ax, ds, timevar, lonvar, dim_name, use_localtime=use_localtime)
    
    #Complement time axis ticks with lat/lon information
    _ax = ax.twiny()
    _ax.set_xlim(ax.get_xlim())

    format_latlon_ticks(ax, _ax, ds, timevar, lonvar, latvar, dim_name)
    
    #Specify the product filename
    product_code = ds.encoding['source'].split('/')[-1].split('.')[0]        

    if dark_mode:
        text_color = 'w'
        text_shading= 'k'
        shading_alpha = 0.5
        shading_lw = 5
    else:
        text_color = 'k'
        text_shading = 'w'
        shading_alpha = 0.5
        shading_lw = 5
        
    shade_around_text(ax.text(0.9975,0.98, product_code, ha='right', va='top', 
                              fontsize='xx-small', color=text_color, transform=ax.transAxes), 
                      lw=shading_lw, alpha=shading_alpha, fg=text_shading)
    
    if include_ruler:
        add_ruler(ax, ds, timevar=timevar, dx=500, x0=100, y0=y0_ruler)
    

def plot_EC_1D(ax, ds, plot1D_dict, title, ylabel, 
               yscale='linear', y0_ruler=0.9, ylim=[None,None],
              timevar='time', latvar='latitude', lonvar='longitude', dim_name='along_track',
              include_ruler=False):

    for l, d in plot1D_dict.items():
        color = None if ('color' not in d) else d['color']
        markersize = 3 if ('markersize' not in d) else d['markersize']
        marker = '.' if ('marker' not in d) else d['marker']
        zorder = None if ('zorder' not in d) else d['zorder']
        scale  = 1 if ('scale' not in d) else d['scale']
                
        ax.plot(d['xdata'], d['ydata'].where(d['ydata'] < 1e30),
                label=l, lw=2.5, marker=marker, markersize=markersize, color=color, zorder=zorder)

    ax.set_yscale(yscale)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.legend(frameon=False, fontsize='xx-small', markerscale=10, bbox_to_anchor=(1,1), loc='upper left')

    format_plot_1D(ax, ds, title, y0_ruler=y0_ruler,
                   timevar=timevar, latvar=latvar, lonvar=lonvar, dim_name=dim_name, include_ruler=include_ruler)
    
    

def plot_CFAD(ax, H, n, v_dict, y_dict, norm=LogNorm(1e-3,2), 
              surface_clutter_height=500, overlay_median=False):
    
    _cm = H.plot.pcolormesh(ax=ax, y=f"{y_dict['varname']}_bin", cmap='chiljet2', norm=norm, 
                                                                   add_colorbar=False)
        
    shade_around_text(ax.text(0.99,0.99,"n$_\mathrm{frames}$" + f"={n}", 
                             va='top', ha='right', 
                             fontsize='x-small', color='k', transform=ax.transAxes), 
                             alpha=0.5, lw=3, fg='w')
            
    ax.set_xlabel(f"{v_dict['label']}" + f" [{v_dict['units']}]")
    ax.set_xscale(v_dict['xscale'])
    ax.set_xticks(v_dict['xticks'])
    ax.set_xlim(v_dict['xlim'])

    if 'height' in y_dict['varname']:
        ax.fill_between(H[f"{v_dict['alias']}_bin"].values, 
                        ax.get_ylim()[0], 
                        surface_clutter_height,
                        hatch='////', edgecolor='k', lw=0, facecolor='none')

    if overlay_median:
        _min = H[f"{v_dict['alias']}_bin"].min().values
        _median = H[f"{v_dict['alias']}_bin"].isel({f"{v_dict['alias']}_bin":H.argmax(f"{v_dict['alias']}_bin")})
        _median.where(_median > _min).plot(ax=ax, y=f"{y_dict['varname']}_bin", color='k')
        
    return _cm
    

def plot_CFAD_frames(H, CFAD_vars, y_dict, norm=LogNorm(1e-3,2), nadir_pixel=None, suptitle=""):
    
    import string
    import xhistogram as histogram
    
    frame_labels = {"A":'equatorial night', 
                    "B":'NH ex.tropical night', 
                    "C":'NH polar', 
                    "D":'NH ex.tropical day', 
                    "E":'equatorial day', 
                    "F":'SH ex.tropical day', 
                    "G":'SH polar', 
                    "H":'SH ex.tropical night'}    

    figs_and_axes = {}
    
    n_frames = len(H.frame)
    for v_label, v_dict in CFAD_vars.items():
        if v_label in H.data_vars:
            fig, axes = plt.subplots(figsize=(5*n_frames, 8), ncols=n_frames, sharex=True, sharey=True)

            for i, f in enumerate(H.frame.values):

                n = (~H.datafile.sel(frame=f).isnull()).sum().values

                if n > 0:
                    _cm = plot_CFAD(axes[i], (100.*counts_to_frequency(H[v_label].sel(frame=f).sum('orbit'))), 
                                    n, v_dict, y_dict, norm)
                else:
                    plot_CFAD(axes[i], xr.zeros_like(H.isel(orbit=0, frame=0))[v_label], 0, 
                              v_dict, y_dict, norm)

                axes[i].set_title(f"frame {f}\n{frame_labels[f]}")

                if i > 0:
                    axes[i].set_ylabel("")

            if nadir_pixel:
                for ax in axes:
                    ax.axhline(y=nadir_pixel, color='k', ls='--', lw=3)
                
            _title = v_dict['varname'].replace("_"," ")
            fig.suptitle(f"{suptitle}\n{H.attrs['product_name']} {_title}", va='bottom', y=1.01)
            
            if 'height' in y_dict['varname']:
                format_height(axes[0])   
            elif 'temperature' in y_dict['varname']:
                format_temperature(axes[0])
            else:
                axes[0].set_ylabel(y_dict['varname'])
                if hasattr(y_dict, 'scale'):
                    if 'log' in y_dict['yscale']:
                        axes[0].set_yscale('log')
            
            if _cm:
                cb = add_colorbar(axes[-1], _cm, "$f$ [%]", width_ratio='5%', horz_buffer=0.1)
                cb.set_ticks([1e-3,1e-2,1e-1,1])
                cb.set_ticklabels([0.001,0.01, 0.1, 1.0])

        figs_and_axes[v_label] = (fig, axes)
    
    return figs_and_axes


def counts_to_frequency(H):
    return H/H.sum()


def intercompare_CFADs(frames, HS, v, v_dict):

    for f in frames:

        ncols=len(HS)
        fig, axes = plt.subplots(figsize=(ncols*5, 7), ncols=ncols, sharex=True, sharey=True, gridspec_kw={'wspace':0.25})

        for i, H in enumerate(HS):

            n = (~H.sel(frame=f).datafile.isnull()).sum().values

            _cm = plot_CFAD(axes[i], (100.*counts_to_frequency(H[v].sel(frame=f).sum('orbit'))),
                            n, v_dict, norm=LogNorm(1e-3,5))
            axes[i].set_title(H.attrs['product_name'])

        add_colorbar(axes[-1], _cm, "$f$ [%]", width_ratio='5%', horz_buffer=0.1)
        axes[-1].set_xscale(v_dict['xscale'])

        format_height(axes[0])
        for ax in axes[1:]:
            ax.set_ylabel("")

        for ax in axes:
            ax.set_xlabel(f"{v_dict['label']} [{v_dict['units']}]")
            ax.set_xticks(v_dict['xticks'])

        fig.suptitle(f"{v_dict['title']}"+f"\nFrame {f}", y=1.05)
        

def quicklook_ANOM(ANOM, hmax=30e3, dstdir=None, total_backscatter=False, 
                   heightvar='sample_altitude', tempvar='layer_temperature', 
                   smoother=None, strato_smoother=False,
                   show_temperature=True, with_marble=False, with_surface=False):
    
    cmap_elastic=colormaps.calipso_smooth
    cmap_inelastic=colormaps.chiljet3
    
    units = 'sr$^{-1}$m$^{-1}$'
    plot_scale='logarithmic'
    plot_range_elastic=[1e-8,1e-5] 
    plot_range_inelastic=[1e-8,1e-5] 

    if total_backscatter:

        nrows=1
        fig, ax = plt.subplots(figsize=(25,5*nrows), nrows=nrows)

        ANOM['total_attenuated_backscatter'] = ANOM.mie_attenuated_backscatter
        ANOM['total_attenuated_backscatter'].values = ANOM.mie_attenuated_backscatter + \
                                        ANOM.rayleigh_attenuated_backscatter + ANOM.crosspolar_attenuated_backscatter
        
        if strato_smoother:
            ANOM['total_attenuated_backscatter_ss'] = ANOM['total_attenuated_backscatter'].where(ANOM.sample_altitude > 20250, 
                                                  ANOM['total_attenuated_backscatter'].rolling(height=5, center=True).mean())
        
            cmap=colormaps.calipso
            plot_EC_2D(ax, ANOM, 'total_attenuated_backscatter_ss', r"$\beta_{\mathrm{tot}}$", cmap=cmap_inelastic, units=units, title="A-NOM total attenuated backscatter", hmax=hmax, smoother=smoother, plot_scale=plot_scale, min_value=1e-9, fill_value=1e-9, plot_range=plot_range_inelastic, heightvar=heightvar, latvar='latitude', lonvar='longitude')
        
        else:
            cmap=colormaps.calipso
            plot_EC_2D(ax, ANOM, 'total_attenuated_backscatter', r"$\beta_{\mathrm{tot}}$", cmap=cmap_inelastic, units=units, title="A-NOM total attenuated backscatter", hmax=hmax, smoother=smoother, plot_scale=plot_scale, min_value=1e-9, fill_value=1e-9, plot_range=plot_range_inelastic, heightvar=heightvar, latvar='latitude', lonvar='longitude')
        
        dx           = 1000
        d0           = 200
        x0           = 200
        ruler_y0     = 0.9
        add_ruler(ax, ANOM, timevar='time', dx=dx, d0=d0, x0=x0, pixel_scale_km=0.5, y0=ruler_y0, dark_mode=False)
        
    else:
    
        nrows=3
        fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, gridspec_kw={'hspace':0.67})

        
        if strato_smoother:
            ANOM['mie_attenuated_backscatter_ss'] = ANOM['mie_attenuated_backscatter'].where(ANOM.sample_altitude > 20250, 
                                                  ANOM['mie_attenuated_backscatter'].rolling(height=5, center=True).mean())
            plot_EC_2D(axes[0], ANOM, 'mie_attenuated_backscatter_ss', r"$\beta_{\mathrm{mie}}$", 
                       cmap=cmap_elastic, units=units, title="A-NOM mie attenuated backscatter", 
                       hmax=hmax,smoother=smoother, min_value=1e-9, fill_value=1e-9,
                              plot_scale=plot_scale,  plot_range=plot_range_elastic, 
                   heightvar=heightvar, latvar='latitude', lonvar='longitude')
        else:
            plot_EC_2D(axes[0], ANOM, 'mie_attenuated_backscatter', r"$\beta_{\mathrm{mie}}$", 
                       cmap=cmap_elastic, units=units, title="A-NOM mie attenuated backscatter", 
                       hmax=hmax,smoother=smoother, min_value=1e-9, fill_value=1e-9,
                              plot_scale=plot_scale,  plot_range=plot_range_elastic, 
                   heightvar=heightvar, latvar='latitude', lonvar='longitude')
        
        if strato_smoother:
            ANOM['crosspolar_attenuated_backscatter_ss'] = ANOM['crosspolar_attenuated_backscatter'].where(ANOM.sample_altitude > 20250, 
                                                  ANOM['crosspolar_attenuated_backscatter'].rolling(height=5, center=True).mean())
        
            plot_EC_2D(axes[1], ANOM, 'crosspolar_attenuated_backscatter_ss', r"$\beta_{\mathrm{xpol}}$", cmap=cmap_elastic, units=units, title="A-NOM cross-polar attenuated backscatter", hmax=hmax,smoother=smoother, min_value=1e-9, fill_value=1e-9,
                              plot_scale=plot_scale,  plot_range=plot_range_elastic, 
                   heightvar=heightvar, latvar='latitude', lonvar='longitude')
            
        else:
            plot_EC_2D(axes[1], ANOM, 'crosspolar_attenuated_backscatter', r"$\beta_{\mathrm{xpol}}$", cmap=cmap_elastic, units=units, title="A-NOM cross-polar attenuated backscatter", hmax=hmax,smoother=smoother, min_value=1e-9, fill_value=1e-9,
                              plot_scale=plot_scale,  plot_range=plot_range_elastic, 
                   heightvar=heightvar, latvar='latitude', lonvar='longitude')

        
        if strato_smoother:
            ANOM['rayleigh_attenuated_backscatter_ss'] = ANOM['rayleigh_attenuated_backscatter'].where(ANOM.sample_altitude > 20250, 
                                                  ANOM['rayleigh_attenuated_backscatter'].rolling(height=5, center=True).mean())
            
            plot_EC_2D(axes[2], ANOM, 'rayleigh_attenuated_backscatter_ss', r"$\beta_{\mathrm{ray}}$", cmap=cmap_inelastic, units=units, title="A-NOM rayleigh attenuated backscatter", hmax=hmax, smoother=smoother, min_value=1e-9, fill_value=1e-9,
                              plot_scale=plot_scale,  plot_range=plot_range_inelastic, 
                   heightvar=heightvar, latvar='latitude', lonvar='longitude')

        else:
            plot_EC_2D(axes[2], ANOM, 'rayleigh_attenuated_backscatter', r"$\beta_{\mathrm{ray}}$", cmap=cmap_inelastic, units=units, title="A-NOM rayleigh attenuated backscatter", hmax=hmax, smoother=smoother, min_value=1e-9, fill_value=1e-9,
                              plot_scale=plot_scale,  plot_range=plot_range_inelastic, 
                   heightvar=heightvar, latvar='latitude', lonvar='longitude')
        

        for ax in axes:
            if show_temperature & (tempvar in ANOM.data_vars):
                add_temperature(ax, ANOM, heightvar=heightvar, tempvar=tempvar)

    if with_marble:
        add_marble(axes[0], ANOM, timevar='time', lonvar='longitude', latvar='latitude')

    if with_surface:
        for ax in axes:
            add_surface(ax, ANOM, 
                    elevation_var='surface_elevation', 
                    land_var='land_flag', hmin=-1e3)
        
    if dstdir:
        srcfile_string = ANOM.encoding['source'].split("/")[-1].split(".")[0]
        dstfile = f"{srcfile_string}_quicklook.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')


def quicklook_CNOM(CNOM, hmax=20e3, dstdir=None, Z_only=False,
                  show_temperature=False, with_marble=False, with_surface=False, 
                  tempvar='temperature'):

    if Z_only:

        nrows=1
        fig, ax = plt.subplots(figsize=(25,5*nrows), nrows=nrows)
        
        cmap = colormaps.chiljet2
        units = 'dBZ'
        plot_scale='linear'
        plot_range=[-40,20] 
        label=r"Z"
        plot_EC_2D(ax, CNOM, 'radarReflectivityFactor', label, cmap=cmap, units=units, title="C-NOM radar reflectivity", hmax=hmax,
                              plot_scale=plot_scale, plot_range=plot_range, timevar='profileTime', heightvar='binHeight')

        if show_temperature & (tempvar in CNOM.data_vars):
                add_temperature(ax, CNOM, heightvar=heightvar, tempvar=tempvar)
    else:
    
        nrows=5
        fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, gridspec_kw={'hspace':0.67})
        
        cmap = colormaps.chiljet2
        units = 'dBZ'
        plot_scale='linear'
        plot_range=[-40,20] 
        label=r"Z"
        plot_EC_2D(axes[0], CNOM, 'radarReflectivityFactor', label, cmap=cmap, units=units, title="C-NOM radar reflectivity", hmax=hmax,
                              plot_scale=plot_scale, plot_range=plot_range, timevar='profileTime', heightvar='binHeight')
        
        plot_EC_1D(axes[1], CNOM, {'sigma0':{'xdata':CNOM['profileTime'], 'ydata':CNOM['sigmaZero']}}, 
                         "surface signal", "$\sigma_0$ [dB]", timevar='profileTime', include_ruler=False)
        
        plot_EC_1D(axes[2], CNOM, {'NF':{'xdata':CNOM['profileTime'], 'ydata':CNOM['noiseFloorPower']}}, 
                         "noise floor", "$F$ [dBW]", timevar='profileTime', include_ruler=False)
        
        cmap = colormaps.litmus
        units = 'ms$^{-1}$'
        plot_scale='linear'
        plot_range=[-6,6] 
        label=r"V$_D$"
        plot_EC_2D(axes[3], CNOM, 'dopplerVelocity', label, scale_factor=-1., cmap=cmap, units=units, 
                   title="C-NOM mean Doppler velocity", hmax=hmax,
                   plot_scale=plot_scale,  plot_range=plot_range, timevar='profileTime', heightvar='binHeight')
        
        cmap = colormaps.chiljet2
        units = 'ms$^{-1}$'
        plot_scale='linear'
        plot_range=[0,10] 
        label=r"$\sigma_D$"
        plot_EC_2D(axes[4], CNOM, 'spectrumWidth', label, cmap=cmap, units=units, title="C-NOM Doppler spectrum width", hmax=hmax,
                              plot_scale=plot_scale,  plot_range=plot_range, timevar='profileTime', heightvar='binHeight')
        
        add_subfigure_labels(axes)

        for ax in axes:
            if show_temperature & (tempvar in CNOM.data_vars):
                add_temperature(ax, CNOM, timevar='profileTime', heightvar='binHeight', tempvar=tempvar)

    if with_marble:
        add_marble(axes[0], CNOM, timevar='time', lonvar='longitude', latvar='latitude')

    if with_surface:
        for ax in axes:
            add_surface(ax, CNOM, 
                    elevation_var='surfaceElevation', 
                    land_var='navigationLandWaterFlg', hmin=-1e3)

    if dstdir:
        srcfile_string = CNOM.encoding['source'].split("/")[-1].split(".")[0]
        dstfile = f"{srcfile_string}_quicklook.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')


def quicklook_BSNG(BSNG, dstdir=None, hspace=1.5, dh=4, show_TW=True):
    
    nrows=2
    fig, axes = plt.subplots(figsize=(25,dh*nrows), nrows=nrows, gridspec_kw={'hspace':hspace})
    
    cmap = 'Blues_r'
    units = 'Wm$^{-2}$sr$^{-1}$'
    label=r"$L_{\mathrm{SW}}$"
    plot_EC_2D(axes[0], BSNG.sel(band='SW'), 'radiance', label, cmap=cmap, units=units, title=f"B-SNG SW radiance, {str(BSNG.view.values).lower()} view", hmax=30,
                          plot_scale='linear', plot_range=[0,200], timevar='time', lonvar='selected_longitude', latvar='selected_latitude', heightvar='across_track', across_track=True)
    add_nadir_track(axes[0], idx_across_track=14, label_below=False, label_offset=0)
    axes[0].set_yticks([0,15,30])
    axes[0].set_ylim(0,30)

    if show_TW:
        cmap = 'Reds'
        units = 'Wm$^{-2}$sr$^{-1}$'
        label=r"$L_{\mathrm{TW}}$"
        plot_EC_2D(axes[1], BSNG.sel(band='TW'), 'radiance', label, cmap=cmap, units=units, title=f"B-SNG TW radiance, {str(BSNG.view.values).lower()} view", hmax=30,
                              plot_scale='linear', plot_range=[np.max([0,BSNG.sel(band='TW').radiance.min()]), 
                                                               BSNG.sel(band='TW').radiance.max()], 
                   timevar='time', lonvar='selected_longitude', latvar='selected_latitude', heightvar='across_track', across_track=True)
        add_nadir_track(axes[1], idx_across_track=14, label_below=False, label_offset=0)
        axes[1].set_yticks([0,15,30])
        axes[1].set_ylim(0,30)
        
    else:
        cmap = 'Reds'
        units = 'Wm$^{-2}$sr$^{-1}$'
        label=r"$L_{\mathrm{LW}}$"
        
        _BSNG = BSNG.copy()
        _BSNG.radiance.sel(band='TW').values = _BSNG.radiance.sel(band='TW') - _BSNG.radiance.sel(band='SW')
        
        plot_EC_2D(axes[1], _BSNG.sel(band='TW'), 'radiance', label, cmap=cmap, units=units, title=f"B-SNG LW radiance, {str(BSNG.view.values).lower()} view", hmax=30,
                              plot_scale='linear', plot_range=[np.max([0,_BSNG.sel(band='TW').radiance.min()]), 
                                                               _BSNG.sel(band='TW').radiance.max()], 
                   timevar='time', lonvar='selected_longitude', latvar='selected_latitude', heightvar='across_track', across_track=True)
        add_nadir_track(axes[1], idx_across_track=14, label_below=False, label_offset=0)
        axes[1].set_yticks([0,15,30])
        axes[1].set_ylim(0,30)

    if dstdir:
        srcfile_string = BSNG.encoding['source'].split("/")[-1].split(".")[0]
        dstfile = f"{srcfile_string}_quicklook.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')


def add_marble(ax, ds, vert_buffer=0.5, timevar='time', lonvar='longitude', latvar='latitude', 
              add_arrows=False, annotate=True):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    import cartopy
    from cartopy.mpl.geoaxes import GeoAxes
    from cartopy import crs as ccrs
    from cartopy import feature as cfeat
    
    idx_mid = len(ds.along_track)//2

    proj = ccrs.Orthographic(central_longitude = ds[lonvar].isel(along_track=idx_mid).values,
                                                  central_latitude  = ds[latvar].isel(along_track=idx_mid).values)
    cax = inset_axes(ax,
                     width="33%",  # percentage of parent_bbox width
                     height="100%",  
                     loc='upper left',
                     bbox_to_anchor=(0.02,1,0.5,1),
                     bbox_transform=ax.transAxes,
                     borderpad=1.0,
                     axes_class=GeoAxes, 
                     axes_kwargs=dict(map_projection=proj)
                     )       

    cax.set_global()
    cax.coastlines(lw=1, color='0.2')
    cax.add_feature(cfeat.LAND, facecolor='0.95')
    cax.gridlines(color='0.5', lw=0.5, xlocs=np.arange(-180,180,15), ylocs=[-67.5,-45,-22.5,0,22.5,45,67.5])
    #cax.plot(ds[lonvar], ds[latvar], marker='.', lw=0, markersize=10, color=sns.color_palette()[3], transform=ccrs.PlateCarree())
    cax.plot(ds[lonvar][::100], ds[latvar][::100], lw=0, markersize=5, marker='.', color=sns.color_palette()[3], solid_capstyle='butt', 
             transform=ccrs.PlateCarree())

    if annotate:
        t0, t1 = ds[timevar][[0,-1]].values
        
        frame = ds.encoding['source'].split("/")[-1].split(".")[0].split("_")[-1][-1]
        if frame in "ABH":
            start_text = f"{pd.to_datetime(t0):%H:%M}" 
            start_ha = "center"
            start_va = "top"
            stop_text = f"{pd.to_datetime(t1):%H:%M}"
            stop_ha = "center"
            stop_va = "bottom"
        elif frame in "CG":
            start_text = f"{pd.to_datetime(t0):%H:%M}"
            start_ha = "center"
            start_va = "bottom"
            stop_text = f"{pd.to_datetime(t1):%H:%M}"
            stop_ha = "center"
            stop_va = "top"
        else:
            start_text = f"{pd.to_datetime(t0):%H:%M}"
            start_ha = "center"
            start_va = "bottom"
            stop_text = f"{pd.to_datetime(t1):%H:%M}" 
            stop_ha = "center"
            stop_va = "top"
            
        cax.scatter(ds.longitude[[0,-1]].values, ds.latitude[[0,-1]].values, marker='.', s=50, color='k', transform=ccrs.PlateCarree(), 
               zorder=11)
        shade_around_text(cax.text(ds.longitude[0].values, ds.latitude[0].values, start_text,
                                         ha=start_ha, va=start_va, fontsize='small', color='k', transform=ccrs.PlateCarree()), 
                         fg='w', lw=4, alpha=0.5)
        shade_around_text(cax.text(ds.longitude[-1].values, ds.latitude[-1].values, stop_text,
                                   ha=stop_ha, va=stop_va, fontsize='small', color='k', transform=ccrs.PlateCarree()), 
                         fg='w', lw=4, alpha=0.5)
    
    if add_arrows:
        idx = len(ds.longitude)//2
    
        cax.quiver(ds.longitude[[idx]].values, ds.latitude[[idx]].values, 
                   ds.longitude[idx:idx+100].diff('along_track')[:1].values, ds.latitude[idx:idx+100].diff('along_track')[:1].values, 
                   transform=ccrs.PlateCarree(), zorder=10, 
                   scale=1/10,
                   width=0.01, headlength=10, headaxislength=10, lw=0, headwidth=10, 
                   facecolor=sns.color_palette()[3], pivot='mid')
    
    from cartopy.feature.nightshade import Nightshade
    cax.add_feature(Nightshade(pd.to_datetime(ds[timevar].isel(along_track=idx_mid).values, utc='UTC'), color=sns.xkcd_rgb['dark blue'], alpha=0.2))

    return cax
    


def quicklook_ACMB(frame, dstdir=None, 
                  time_lims=(None, None), with_marble=True, 
                  ANOM_srcdir="/perm/pasm/CARDINAL/commissioning_phase/first_ATL_data/ATL_NOM_1B/",
                  CNOM_srcdir="/perm/pasm/CARDINAL/commissioning_phase/first_CPR_data/CPR_NOM_1B/",
                  MRGR_srcdir="/perm/pasm/CARDINAL/commissioning_phase/first_MSI_data/MSI_RGR_1C/",
                  BSNG_srcdir="/perm/pasm/CARDINAL/commissioning_phase/first_BBR_data/BBR_SNG_1B/"):

    if dstdir:
        plt.ioff()
    
    from . import ecio
    ANOM = ecio.load_ANOM(ANOM_srcdir, frame_code=frame, trim=True)
    CNOM = ecio.load_CNOM(CNOM_srcdir, frame_code=frame)
    MRGR = ecio.load_MRGR(MRGR_srcdir, frame_code=frame)
    BSNG = ecio.load_BSNG(BSNG_srcdir, frame_code=frame)

    if ANOM: print("A-NOM available")
    if CNOM: print("C-NOM available")
    if MRGR: print("M-RGR available")
    if BSNG: print("B-SNG available")

    srcfile_string = "ECA_EXAA_ALL_NOM_1B_"
    if ANOM:
        srcfile_string += "_".join(ANOM.encoding['source'].split("/")[-1].split(".")[0].split("_")[-3:])
    elif CNOM: 
        srcfile_string += "_".join(CNOM.encoding['source'].split("/")[-1].split(".")[0].split("_")[-3:])
    elif MRGR:
        srcfile_string += "_".join(CNOM.encoding['source'].split("/")[-1].split(".")[0].split("_")[-3:])
    elif BSNG:
        srcfile_string += "_".join(BSNG.encoding['source'].split("/")[-1].split(".")[0].split("_")[-3:])
    
    nrows = (2 if ANOM else 0) + \
            (2 if CNOM else 0) + \
            (2 if MRGR else 0) + \
            (2 if BSNG else 0) 

    if nrows == 0:
        print("No L1 data available for that frame")
        return None, None
    elif nrows == 1:
        print("Only one instrument available; need to re-write plotting function for this case")
        return None, None
    else:
        if time_lims[0] or time_lims[1]:
            tstart, tstop = time_lims
            if ANOM:
                ANOM['along_track'] = ANOM.along_track
                ANOM = ANOM.swap_dims({'along_track':'time'}).sel(time=slice(tstart, tstop)).swap_dims({'time':'along_track'})
    
            if CNOM:
                CNOM['along_track'] = CNOM.along_track
                CNOM = CNOM.swap_dims({'along_track':'profileTime'}).sel(profileTime=slice(tstart, tstop)).swap_dims({'profileTime':'along_track'})
    
            if MRGR:
                MRGR['along_track'] = MRGR.along_track
                MRGR = MRGR.swap_dims({'along_track':'time'}).sel(time=slice(tstart, tstop)).swap_dims({'time':'along_track'})
    
            if BSNG:
                BSNG['along_track'] = BSNG.along_track
                BSNG['time'] = BSNG.time.max('band').squeeze()
                BSNG = BSNG.swap_dims({'along_track':'time'}).sel(time=slice(tstart, tstop)).swap_dims({'time':'along_track'})

        figure_height=(20 if ANOM else 0) + \
                      (14 if CNOM else 0) + \
                      (8 if MRGR else 0) + \
                      (6 if BSNG else 0)
        height_ratios = ([1.5,1.5] if ANOM else []) + \
                        ([1,1] if CNOM else []) + \
                        ([0.4,0.4] if MRGR else []) + \
                        ([0.2,0.2] if BSNG else [])

        if figure_height > 45:
            hspace=1.25
            suptitle_yloc = 0.95
        elif figure_height > 35:
            hspace=1.5
            suptitle_yloc = 0.975
        elif figure_height > 30:
            hspace=1.15
            suptitle_yloc = 1.0
        else:
            hspace=1.2
            suptitle_yloc = 1.0
        
        fig, axes = plt.subplots(figsize=(25,figure_height), nrows=nrows, 
                                 gridspec_kw={'height_ratios':height_ratios,
                                              'hspace':hspace})
    
        i=0
    
        if ANOM:
            units = 'sr$^{-1}$m$^{-1}$'
            plot_scale='logarithmic'
            plot_range=[1e-7,1e-4]
            cmap=colormaps.calipso
        
            ANOM['mie_total_attenuated_backscatter'] = ANOM.mie_attenuated_backscatter
            ANOM['mie_total_attenuated_backscatter'].values = ANOM.mie_attenuated_backscatter + ANOM.crosspolar_attenuated_backscatter
            ANOM['mie_total_attenuated_backscatter']  = ANOM['mie_total_attenuated_backscatter'].where(ANOM['mie_total_attenuated_backscatter'] >= 1e-8, 1e-8)
        
            plot_EC_2D(axes[i], ANOM, 'mie_total_attenuated_backscatter', r"$\beta_{\mathrm{mie}}$", cmap=cmap, units=units, title="A-NOM mie total attenuated backscatter", hmax=30e3, plot_scale=plot_scale, plot_range=plot_range, heightvar='sample_altitude', latvar='ellipsoid_latitude', lonvar='ellipsoid_longitude')
        
            dx           = 1000
            d0           = 200
            x0           = 200
            ruler_y0     = 0.9
            if False: add_ruler(axes[i], ANOM, timevar='time', dx=dx, d0=d0, x0=x0, pixel_scale_km=0.5, y0=ruler_y0, dark_mode=False)
            i+=1

            
            units = 'sr$^{-1}$m$^{-1}$'
            plot_scale='logarithmic'
            plot_range=[1e-7,5e-6]
            cmap=colormaps.chiljet2
            
            ANOM['rayleigh_attenuated_backscatter']  = ANOM['rayleigh_attenuated_backscatter'].where(ANOM['rayleigh_attenuated_backscatter'] >= 1e-8, 1e-8)
        
            plot_EC_2D(axes[i], ANOM, 'rayleigh_attenuated_backscatter', r"$\beta_{\mathrm{rayleigh}}$", cmap=cmap, units=units, title="A-NOM Rayleigh attenuated backscatter", hmax=30e3, plot_scale=plot_scale, plot_range=plot_range, heightvar='sample_altitude', latvar='ellipsoid_latitude', lonvar='ellipsoid_longitude')
        
            dx           = 1000
            d0           = 200
            x0           = 200
            ruler_y0     = 0.9
            if False: add_ruler(axes[i], ANOM, timevar='time', dx=dx, d0=d0, x0=x0, pixel_scale_km=0.5, y0=ruler_y0, dark_mode=False)
            i+=1

            if with_marble:
                add_marble(axes[0], ANOM, timevar='time', lonvar='ellipsoid_longitude', latvar='ellipsoid_latitude')
            
            ANOM.close()
    
        if CNOM:
            cmap = colormaps.chiljet2
            units = 'dBZ'
            plot_scale='linear'
            plot_range=[-40,20] 
            label=r"Z"
            plot_EC_2D(axes[i], CNOM, 'radarReflectivityFactor', label, cmap=cmap, units=units, title="C-NOM radar reflectivity", hmax=20e3,
                                  plot_scale=plot_scale, plot_range=plot_range, timevar='profileTime', heightvar='binHeight')
            i+=1

            cmap = colormaps.litmus
            units = 'm/s'
            plot_scale='linear'
            plot_range=[-6,6] 
            label=r"V"
            plot_EC_2D(axes[i], CNOM, 'dopplerVelocity', label, cmap=cmap, units=units, title="C-NOM Doppler velocity", hmax=20e3,
                                  plot_scale=plot_scale, plot_range=plot_range, timevar='profileTime', heightvar='binHeight')
            i+=1

            CNOM.close()
    
        if MRGR:
            RGB = calculate_RGB_MRGR(MRGR)
            plot_RGB_MRGR(axes[i], RGB, MRGR)
            format_MRGR(axes[i], MRGR)
            i+=1
            
            plot_TIR_MRGR(axes[i], MRGR)
            format_MRGR(axes[i], MRGR)
            i+=1

            MRGR.close()
    
        if BSNG:
            cmap = 'SW_r'
            units = 'Wm$^{-2}$sr$^{-1}$'
            label=r"$L_{\mathrm{SW}}$"
            plot_EC_2D(axes[i], BSNG.sel(band='SW'), 'radiance', label, cmap=cmap, units=units, plot_where=(BSNG.sel(band='SW').radiance > 0), 
                       title=f"B-SNG SW radiance, {str(BSNG.view.values).lower()} view", hmax=30, fill_value=None, 
                       plot_scale='linear', plot_range=[0,200], timevar='time', lonvar='selected_longitude', latvar='selected_latitude',
                       heightvar='across_track', across_track=True)
            add_nadir_track(axes[i], idx_across_track=14, label_below=False, label_offset=0)
            axes[i].set_yticks([0,15,30])
            axes[i].set_ylim(0,30)
            i+=1
        
            cmap = 'LW'
            units = 'Wm$^{-2}$sr$^{-1}$'
            label=r"$L_{\mathrm{LW}}$"
            
            _BSNG = BSNG.copy()
            _BSNG.radiance.sel(band='TW').values = _BSNG.radiance.sel(band='TW').where(_BSNG.radiance.sel(band='TW') > 0).fillna(0.).squeeze() - _BSNG.radiance.sel(band='SW').where(_BSNG.radiance.sel(band='SW') > 0).fillna(0.)
            
            plot_EC_2D(axes[i], _BSNG.sel(band='TW'), 'radiance', label, cmap=cmap, units=units, title=f"B-SNG LW radiance, {str(BSNG.view.values).lower()} view", hmax=30, plot_scale='linear', fill_value=0., plot_range=[50,150], timevar='time', lonvar='selected_longitude', latvar='selected_latitude', heightvar='across_track', across_track=True)
            add_nadir_track(axes[i], idx_across_track=14, label_below=False, label_offset=0)
            axes[i].set_yticks([0,15,30])
            axes[i].set_ylim(0,30)

            BSNG.close()
    
        add_subfigure_labels(axes, yloc=1.2)
        snap_xlims(axes)  
       
        fig.suptitle("EarthCARE L1 synergy", fontsize='xx-large', y=suptitle_yloc, x=0.5125, va='top')
    
        if ANOM: ANOM.close()
        if CNOM: CNOM.close()
        if MRGR: MRGR.close()
        if BSNG: BSNG.close()
    
    if dstdir:
        print(srcfile_string)
        dstfile = f"{srcfile_string}_ACMB_quicklook.png"
        print(dstfile)
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')



def quicklook_ACM(frame, dstdir=None, 
                  time_lims=(None, None),
                  ANOM_srcdir="/perm/pasm/CARDINAL/commissioning_phase/first_ATL_data/ATL_NOM_1B/",
                  CNOM_srcdir="/perm/pasm/CARDINAL/commissioning_phase/first_CPR_data/CPR_NOM_1B/",
                  MRGR_srcdir="/perm/pasm/CARDINAL/commissioning_phase/first_MSI_data/MSI_RGR_1C/", 
                  XMET_srcdir="/perm/pasm/CARDINAL/commissioning_phase/first_AUX_data/AUX_MET_1D/",
                  CNOM_baseline="[A-Z][A-Z]", ANOM_baseline="[A-Z][A-Z]", MRGR_baseline="[A-Z][A-Z]",
                  show_polar_channel=False, hmax_CNOM=20e3, hmax_ANOM=30e3, hmin=-0.5e3, ANOM_smoother=dict(along_track=21),
                  strato_smoother=False, show_temperature=True, short_timestep=False,
                  nested_directory_structure=False):

    if False:
        if dstdir:
            plt.ioff()
        else:
            plt.ion()

    sns.set_context('poster')
    sns.set_style('ticks')
    
    from . import ecio
    XMET = ecio.load_XMET(XMET_srcdir, prodmod_code=f"ECA_EX[A-Z][A-Z]", frame_code=frame, 
                          nested_directory_structure=nested_directory_structure)
    if XMET:
        ANOM = ecio.get_XMET(XMET, ecio.load_ANOM(ANOM_srcdir, prodmod_code=f"ECA_EX{ANOM_baseline}", frame_code=frame, trim=True, 
                          nested_directory_structure=nested_directory_structure),
                             grid_altvar='sample_altitude', grid_heightdim='height')
        CNOM = ecio.get_XMET(XMET, ecio.load_CNOM(CNOM_srcdir, prodmod_code=f"ECA_JX{CNOM_baseline}", frame_code=frame, trim=True, 
                          nested_directory_structure=nested_directory_structure), 
                             grid_altvar='binHeight', grid_heightdim='CPR_height', grid_time='profileTime')
    else:
        ANOM = ecio.load_ANOM(ANOM_srcdir, prodmod_code=f"ECA_EX{ANOM_baseline}", frame_code=frame, trim=True, 
                          nested_directory_structure=nested_directory_structure)
        CNOM = ecio.load_CNOM(CNOM_srcdir, prodmod_code=f"ECA_JX{CNOM_baseline}", frame_code=frame, trim=True, 
                          nested_directory_structure=nested_directory_structure)
    MRGR = ecio.load_MRGR(MRGR_srcdir, prodmod_code=f"ECA_EX{MRGR_baseline}", frame_code=frame, trim=True, 
                          nested_directory_structure=nested_directory_structure)

    if ANOM: 
        print("A-NOM available")
        ANOM['time'].values -= np.timedelta64(3, 's') + np.timedelta64(100, 'ms')
    if CNOM: 
        print("C-NOM available")
    if MRGR: print("M-RGR available")

    if show_polar_channel:
        nrows = (3 if ANOM else 0) + \
                (2 if CNOM else 0) + \
                (2 if MRGR else 0)
    else:
        nrows = (2 if ANOM else 0) + \
                (2 if CNOM else 0) + \
                (2 if MRGR else 0)

    if nrows == 0:
        print("No L1 data available for that frame")
        return None, None
    elif nrows == 1:
        print("Only one instrument available; need to re-write plotting function for this case")
        return None, None
    else:
        if time_lims[0] or time_lims[1]:
            tstart, tstop = time_lims
            if ANOM:
                ANOM['along_track'] = ANOM.along_track
                ANOM = ANOM.swap_dims({'along_track':'time'}).sel(time=slice(tstart, tstop)).swap_dims({'time':'along_track'})
                
            if CNOM:
                CNOM['along_track'] = CNOM.along_track
                CNOM = CNOM.swap_dims({'along_track':'profileTime'}).sel(profileTime=slice(tstart, tstop)).swap_dims({'profileTime':'along_track'})
    
            if MRGR:
                MRGR['along_track'] = MRGR.along_track
                MRGR = MRGR.swap_dims({'along_track':'time'}).sel(time=slice(tstart, tstop)).swap_dims({'time':'along_track'})

            short_timestep=True

        if short_timestep:
            if hmax_CNOM < 10e3:
                figure_height=(15 if ANOM else 0) + \
                              (15 if CNOM else 0) + \
                              (15 if MRGR else 0)
                height_ratios = ([1.,1.] if ANOM else []) + \
                                ([1.,1.] if CNOM else []) + \
                                ([1.,1.] if MRGR else [])
            else:
                figure_height=(20 if ANOM else 0) + \
                              (20 if CNOM else 0) + \
                              (20 if MRGR else 0)
                height_ratios = ([1.,1.] if ANOM else []) + \
                                ([1.,1.] if CNOM else []) + \
                                ([1.,1.] if MRGR else [])
            
        elif show_polar_channel:
            figure_height=(30 if ANOM else 0) + \
                          (14 if CNOM else 0) + \
                          (8 if MRGR else 0)
            height_ratios = ([1.5,1.5,1.5] if ANOM else []) + \
                            ([1, 1] if CNOM else []) + \
                            ([0.4,0.4] if MRGR else [])
            
        else:
            figure_height=(20 if ANOM else 0) + \
                          (14 if CNOM else 0) + \
                          (8 if MRGR else 0)
            height_ratios = ([1.5,1.5] if ANOM else []) + \
                            ([1, 1] if CNOM else []) + \
                            ([0.4,0.4] if MRGR else [])

        if (figure_height > 50) & short_timestep:
            hspace=0.4
            suptitle_yloc=0.92
        elif (figure_height < 50) & short_timestep:
            hspace=0.75
            suptitle_yloc=0.95
        elif figure_height > 30:
            hspace=0.75
            suptitle_yloc = 0.95
        elif figure_height > 20:
            hspace=0.8
            suptitle_yloc = 0.975
        elif figure_height > 16:
            hspace=1.0
            suptitle_yloc = 1.0
        else:
            hspace=1.1
            suptitle_yloc = 1.0

        if ANOM or CNOM or MRGR:

            if short_timestep:
                figure_width=25
            else:
                figure_width=25
                
            fig, axes = plt.subplots(figsize=(figure_width,figure_height), nrows=nrows, 
                                 gridspec_kw={'height_ratios':height_ratios,
                                              'hspace':hspace})
        
            i=0
        
            if ANOM:
                hmax = hmax_ANOM
                units = 'sr$^{-1}$m$^{-1}$'
                plot_scale='logarithmic'
                plot_range=[1e-8,1e-5]
                cmap=colormaps.calipso_smooth
            
                if not show_polar_channel:
                    ANOM['mie_total_attenuated_backscatter'] = ANOM.mie_attenuated_backscatter
                    ANOM['mie_total_attenuated_backscatter'].values = ANOM.mie_attenuated_backscatter + ANOM.crosspolar_attenuated_backscatter

                    if strato_smoother:
                        ANOM['mie_total_attenuated_backscatter_ss'] = ANOM['mie_total_attenuated_backscatter'].where(ANOM.sample_altitude > 20250, ANOM['mie_total_attenuated_backscatter'].rolling(height=5, center=True).mean())
                        
                        plot_EC_2D(axes[i], ANOM, 'mie_total_attenuated_backscatter_ss', r"$\beta_{\mathrm{mie}}$", cmap=cmap, 
                                   units=units, title="A-NOM mie total attenuated backscatter", hmax=hmax, hmin=hmin, 
                                   smoother=ANOM_smoother, min_value=1e-9, fill_value=1e-9, 
                                   plot_scale=plot_scale, plot_range=plot_range, heightvar='sample_altitude', 
                                   latvar='ellipsoid_latitude', lonvar='ellipsoid_longitude', short_timestep=short_timestep)
                    else:
                        
                        plot_EC_2D(axes[i], ANOM, 'mie_total_attenuated_backscatter', r"$\beta_{\mathrm{mie}}$", cmap=cmap, 
                                   units=units, title="A-NOM mie total attenuated backscatter", hmax=hmax, hmin=hmin, 
                                   smoother=ANOM_smoother, min_value=1e-9, fill_value=1e-9, 
                                   plot_scale=plot_scale, plot_range=plot_range, heightvar='sample_altitude', 
                                   latvar='ellipsoid_latitude', lonvar='ellipsoid_longitude', short_timestep=short_timestep)
                        
                else:
                    if strato_smoother:
                        ANOM['mie_attenuated_backscatter_ss'] = ANOM['mie_attenuated_backscatter'].where(ANOM.sample_altitude > 20250, ANOM['mie_attenuated_backscatter'].rolling(height=5, center=True).mean())
                        ANOM['crosspolar_attenuated_backscatter_ss'] = ANOM['crosspolar_attenuated_backscatter'].where(ANOM.sample_altitude > 20250, ANOM['crosspolar_attenuated_backscatter'].rolling(height=5, center=True).mean())
                        
                        plot_EC_2D(axes[i], ANOM, 'mie_attenuated_backscatter_ss', r"$\beta_{\mathrm{mie}}$", cmap=cmap, 
                                   units=units, title="A-NOM mie co-polar attenuated backscatter", hmax=hmax, hmin=hmin, 
                                   smoother=ANOM_smoother, min_value=1e-9, fill_value=1e-9, 
                                   plot_scale=plot_scale, plot_range=plot_range, heightvar='sample_altitude', 
                                   latvar='ellipsoid_latitude', lonvar='ellipsoid_longitude', short_timestep=short_timestep)
                        
                        if show_temperature & ('temperature' in ANOM.data_vars):
                             add_temperature(axes[i],ANOM, heightvar='sample_altitude')
                             
                        #Draw surface elevation
                        axes[i].plot(ANOM.time, ANOM.surface_elevation, color='k', lw=2.5)
                         
                        dx           = 1000
                        d0           = 200
                        x0           = 200
                        ruler_y0     = 0.9
                        i+=1

                        plot_EC_2D(axes[i], ANOM, 'crosspolar_attenuated_backscatter_ss', r"$\beta_{\mathrm{mie}}$", cmap=cmap, 
                                   units=units, title="A-NOM mie cross-polar attenuated backscatter", hmax=hmax, hmin=hmin, 
                                   smoother=ANOM_smoother, min_value=1e-9, fill_value=1e-9, 
                                   plot_scale=plot_scale, plot_range=plot_range, heightvar='sample_altitude', 
                                   latvar='ellipsoid_latitude', lonvar='ellipsoid_longitude', short_timestep=short_timestep)
                    else:
    
                        plot_EC_2D(axes[i], ANOM, 'mie_attenuated_backscatter', r"$\beta_{\mathrm{mie}}$", cmap=cmap, 
                                   units=units, title="A-NOM mie co-polar attenuated backscatter", hmax=hmax, hmin=hmin, 
                                   smoother=ANOM_smoother, min_value=1e-9, fill_value=1e-9, 
                                   plot_scale=plot_scale, plot_range=plot_range, heightvar='sample_altitude', 
                                   latvar='ellipsoid_latitude', lonvar='ellipsoid_longitude', short_timestep=short_timestep)

                        if show_temperature & ('temperature' in ANOM.data_vars):
                            add_temperature(axes[i],ANOM, heightvar='sample_altitude')

                        #Draw surface elevation
                        axes[i].plot(ANOM.time, ANOM.surface_elevation, color='k', lw=2.5)
                         
                        dx           = 1000
                        d0           = 200
                        x0           = 200
                        ruler_y0     = 0.9
                        i+=1
                        
                        plot_EC_2D(axes[i], ANOM, 'crosspolar_attenuated_backscatter', r"$\beta_{\mathrm{mie}}$", cmap=cmap, 
                                   units=units, title="A-NOM mie cross-polar attenuated backscatter", hmax=hmax, hmin=hmin, 
                                   smoother=ANOM_smoother, min_value=1e-9, fill_value=1e-9, 
                                   plot_scale=plot_scale, plot_range=plot_range, heightvar='sample_altitude', 
                                   latvar='ellipsoid_latitude', lonvar='ellipsoid_longitude', short_timestep=short_timestep)

                if show_temperature & ('temperature' in ANOM.data_vars):
                    add_temperature(axes[i],ANOM, heightvar='sample_altitude')
                    
                #Draw surface elevation
                axes[i].plot(ANOM.time, ANOM.surface_elevation, color='k', lw=2.5)
                
                dx           = 1000
                d0           = 200
                x0           = 200
                ruler_y0     = 0.9
                if False:add_ruler(axes[i], ANOM, timevar='time', dx=dx, d0=d0, x0=x0, pixel_scale_km=0.5, y0=ruler_y0, dark_mode=False)
                i+=1
            
                units = 'sr$^{-1}$m$^{-1}$'
                plot_scale='logarithmic'
                plot_range=[1e-8,1e-5]
                cmap=colormaps.chiljet3

                if strato_smoother:
                    ANOM['rayleigh_attenuated_backscatter_ss'] = ANOM['rayleigh_attenuated_backscatter'].where(ANOM.sample_altitude > 20300, 
                                                  ANOM['rayleigh_attenuated_backscatter'].rolling(height=5, center=True).mean())
                
                    plot_EC_2D(axes[i], ANOM, 'rayleigh_attenuated_backscatter_ss', r"$\beta_{\mathrm{rayleigh}}$", cmap=cmap, 
                           units=units, title="A-NOM Rayleigh attenuated backscatter", hmax=hmax, hmin=hmin, 
                           smoother=ANOM_smoother, min_value=1e-9, fill_value=1e-9, 
                           plot_scale=plot_scale, plot_range=plot_range, heightvar='sample_altitude', 
                           latvar='ellipsoid_latitude', lonvar='ellipsoid_longitude', short_timestep=short_timestep)
                else:
                    plot_EC_2D(axes[i], ANOM, 'rayleigh_attenuated_backscatter', r"$\beta_{\mathrm{rayleigh}}$", cmap=cmap, 
                           units=units, title="A-NOM Rayleigh attenuated backscatter", hmax=hmax, hmin=hmin, 
                           smoother=ANOM_smoother, min_value=1e-9, fill_value=1e-9, 
                           plot_scale=plot_scale, plot_range=plot_range, heightvar='sample_altitude', 
                           latvar='ellipsoid_latitude', lonvar='ellipsoid_longitude', short_timestep=short_timestep)

                if show_temperature & ('temperature' in ANOM.data_vars):
                    add_temperature(axes[i], ANOM, heightvar='sample_altitude')

                #Draw surface elevation
                axes[i].plot(ANOM.time, ANOM.surface_elevation, color='k', lw=2.5)
                
                dx           = 1000
                d0           = 200
                x0           = 200
                ruler_y0     = 0.9
                if False: add_ruler(axes[i], ANOM, timevar='time', dx=dx, d0=d0, x0=x0, pixel_scale_km=0.5, y0=ruler_y0, dark_mode=False)
                i+=1
            
            if CNOM:
                hmax = hmax_CNOM
                cmap = colormaps.chiljet2
                units = 'dBZ'
                plot_scale='linear'
                plot_range=[-40,20] 
                label=r"Z"
                plot_EC_2D(axes[i], CNOM, 'radarReflectivityFactor', label, cmap=cmap, units=units, 
                           title="C-NOM radar reflectivity", hmax=hmax, hmin=hmin,
                           plot_scale=plot_scale, plot_range=plot_range, timevar='profileTime', 
                           heightvar='binHeight', short_timestep=short_timestep)
                if show_temperature & ('temperature' in CNOM.data_vars):
                    add_temperature(axes[i], CNOM, timevar='profileTime', heightvar='binHeight')
                
                #Draw surface elevation
                axes[i].plot(CNOM.profileTime, CNOM.surfaceElevation, color='k', lw=2.5)
                
                i+=1

                cmap = colormaps.litmus
                units = 'm/s'
                plot_scale='linear'
                plot_range=[-5,5] 
                label=r"V$_D$"
                plot_EC_2D(axes[i], CNOM, 'dopplerVelocity', label, cmap=cmap, units=units, title="C-NOM Doppler velocity", 
                           hmax=hmax, hmin=hmin, plot_scale=plot_scale, plot_range=plot_range, 
                           timevar='profileTime', heightvar='binHeight', short_timestep=short_timestep)
                if show_temperature & ('temperature' in CNOM.data_vars):
                    add_temperature(axes[i], CNOM, timevar='profileTime', heightvar='binHeight')

                #Draw surface elevation
                axes[i].plot(CNOM.profileTime, CNOM.surfaceElevation, color='k', lw=2.5)
                
                i+=1
            
            if MRGR:
                RGB = calculate_RGB_MRGR(MRGR)
                plot_RGB_MRGR(axes[i], RGB, MRGR, short_timestep=short_timestep)
                format_MRGR(axes[i], MRGR)
                i+=1
            
                plot_TIR_MRGR(axes[i], MRGR, short_timestep=short_timestep)
                format_MRGR(axes[i], MRGR)
                i+=1
    
            add_subfigure_labels(axes, yloc=1.2)
            snap_xlims(axes)    

            if not short_timestep:
                if CNOM:
                    add_marble(axes[0], CNOM, timevar='profileTime', lonvar='longitude', latvar='latitude', add_arrows=False, annotate=True)
                elif ANOM:
                    add_marble(axes[0], ANOM, timevar='time', lonvar='longitude', latvar='latitude', add_arrows=False, annotate=True)
                
            fig.suptitle("EarthCARE L1 synergy", fontsize='xx-large', y=suptitle_yloc, x=0.5125, va='top')

        datasets_available = []
        
        if XMET: XMET.close()
        if ANOM: datasets_available.append("ANOM"); ANOM.close()
        if CNOM: datasets_available.append("CNOM"); CNOM.close()
        if MRGR: datasets_available.append("MRGR"); MRGR.close()
        
        datasets_available = "_".join(datasets_available)
    
    if dstdir:
        import datetime
        now = datetime.datetime.now()
        dstfile = f"EarthCARE_{datasets_available}_L1_quicklook_{frame}_{now:%H%M%S}.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')
        



def quicklook_AFM(AFM, hmax=30e3, dstdir=None, with_marble=False, show_temperature=False):
    
    fig, ax = plt.subplots(figsize=(25,5))

    plot_EC_2D(ax, AFM, 'featuremask', "class", cmap=colormaps.litmus_doppler, plot_scale='linear', plot_range=[-3,10], units='', hmax=hmax)

    if show_temperature and ('temperature' in AFM.data_vars):
        add_temperature(ax, AFM)
    
    if with_marble:
        add_marble(ax, AFM, timevar='time', lonvar='longitude', latvar='latitude', add_arrows=False, annotate=True)
    
    if dstdir:
        srcfile_string = AFM.encoding['source'].split("/")[-1].split(".")[0]
        dstfile = f"{srcfile_string}_quicklook.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')
    else:
        return fig, ax


def quicklook_AEBD(AEBD, resolution='high', hmax=30e3, dstdir=None, with_marble=False, show_temperature=False,):
        
    if 'med' in resolution:
        suffix='_medium_resolution'
    elif 'low' in resolution:
        suffix='_low_resolution'
    else:
        suffix=''
    
    nrows=5
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, gridspec_kw={'hspace':0.75})
    
    plot_EC_target_classification(axes[0], AEBD, 'simple_classification', colormaps.chiljet2(np.linspace(0, 1, 9)), hmax=hmax)
    
    plot_EC_2D(axes[1], AEBD, 'particle_backscatter_coefficient_355nm' + suffix, r"$\beta_\mathrm{mie}$", cmap=colormaps.calipso_smooth, plot_scale='log', plot_range=[1e-8,1e-4], units='sr$^{-1}$m$^{-1}$', hmax=hmax)
    
    plot_EC_2D(axes[2], AEBD, 'particle_linear_depol_ratio_355nm' + suffix, r"$\delta$", cmap=colormaps.chiljet2, plot_scale='linear', plot_range=[0,0.5], units='-', hmax=hmax)
    
    plot_EC_2D(axes[3], AEBD, 'particle_extinction_coefficient_355nm' + suffix, r"$\alpha$", cmap=colormaps.chiljet2, plot_scale='log', plot_range=[1e-6,1e-2], units='$m^{-1}$', hmax=hmax)
    
    plot_EC_2D(axes[4], AEBD, 'lidar_ratio_355nm' + suffix, r"$S$", cmap=colormaps.chiljet2, plot_scale='linear', plot_range=[0,100], units='-', hmax=hmax)

    if show_temperature and ('temperature' in AEBD.data_vars):
        for ax in axes:
            add_temperature(ax, AEBD)
    
    add_subfigure_labels(axes)
    
    if with_marble:
        add_marble(axes[0], AEBD, timevar='time', lonvar='longitude', latvar='latitude', add_arrows=False, annotate=True)

    if dstdir:
        srcfile_string = AEBD.encoding['source'].split("/")[-1].split(".")[0]
        dstfile = f"{srcfile_string}_quicklook{suffix}.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')
    else:
        return fig, axes


def quicklook_AICE(AICE, hmax=20e3, dstdir=None, with_marble=False, show_temperature=False,):
        
    nrows=2
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, gridspec_kw={'hspace':0.75})
    
    hmax=20e3
    
    plot_EC_2D(axes[0], AICE, 'ice_water_content', r"IWC", scale_factor=1e-6, cmap=colormaps.chiljet2, plot_scale='log', plot_range=[1e-7,1e-4], units='kg m$^{-3}$', hmax=hmax)
    
    plot_EC_2D(axes[1], AICE, 'ice_effective_radius', r"$r_\mathrm{eff}$", plot_where=AICE.ice_water_content > 0, cmap=colormaps.chiljet2, plot_scale='linear', plot_range=[0,100], units='$\mu$m', hmax=hmax)
    
    add_subfigure_labels(axes)

    if show_temperature and ('temperature' in AICE.data_vars):
        for ax in axes:
            add_temperature(ax, AICE)
    
    if with_marble:
        add_marble(axes[0], AICE, timevar='time', lonvar='longitude', latvar='latitude', add_arrows=False, annotate=True)
    
    if dstdir:
        srcfile_string = AICE.encoding['source'].split("/")[-1].split(".")[0]
        dstfile = f"{srcfile_string}_quicklook.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')
    else:
        fig, axes


def quicklook_ATC(ATC, hmax=20e3, resolution='high', dstdir=None, with_marble=False, show_temperature=False, with_hatching=True, timevar='time'):
        
    if 'med' in resolution:
        suffix='_medium_resolution'
    elif 'low' in resolution:
        suffix='_low_resolution'
    else:
        suffix=''
    
    nrows=1
    fig, ax = plt.subplots(figsize=(25,7*nrows), nrows=nrows, gridspec_kw={'hspace':0.75})
    
    plot_EC_target_classification(ax, ATC, 'classification' + suffix, ATC_category_colors, hmax=hmax)

    if with_hatching:

        ATC_aerosol_classes = [10,11,12,13,14,15,25,26,27]
        ATC_no_data_classes = [-2,-1]
        ATC_unknown_classes = [101,102,103,104,105,106,107]
        
        _x, _y, is_aerosol, is_no_data, is_unknown = xr.broadcast(ATC[timevar], ATC.height, 
                                                                  ATC['classification' + suffix].isin(ATC_aerosol_classes),
                                                                  ATC['classification' + suffix].isin(ATC_no_data_classes),
                                                                  ATC['classification' + suffix].isin(ATC_unknown_classes))
        overlay = -2*is_no_data -1*is_unknown + 1*is_aerosol 
        hatches = ['//////', '\\\\\\', '', '....']
        
        cs = ax.contourf(_x, _y, overlay,
                           [-2.5, -1.5, -0.5, 0.5, 1.5], colors=['none', 'none', 'none', 'none'], hatches=hatches)
        
        import matplotlib
        # For each level, we set the color of its hatch 
        for i, collection in enumerate(cs.collections):
            collection.set_edgecolor('k')
        # Doing this also colors in the box around each level
        # We can remove the colored line around the levels by setting the linewidth to 0
        for collection in cs.collections:
            collection.set_linewidth(0.)
        # ------------------------------

        import matplotlib.patches as mpatches
        p1 = mpatches.Patch( facecolor='1.0', edgecolor='k', linewidth=0, hatch=r'//////', label='no data')
        p2 = mpatches.Patch( facecolor='0.9', edgecolor='k', linewidth=0, hatch='\\\\\\',  label='unknown')
        p3 = mpatches.Patch( facecolor='0.8', edgecolor='k', linewidth=0, hatch='',        label='hydrometeors')
        p4 = mpatches.Patch( facecolor='0.7', edgecolor='k', linewidth=0, hatch='....',    label='aerosol')

        patches = [p1, p2, p3, p4]
            
        ax.legend(handles=patches, frameon=False, loc='lower right', bbox_to_anchor=(1.33,0), fontsize='xx-small', 
                     labelspacing=0.0, handlelength=1.5)
        
        ax.plot(ATC.time, ATC.elevation, color='k', lw=2.5)

    
    if show_temperature and ('temperature' in ATC.data_vars):
        add_temperature(ax, ATC)
    
    if with_marble:
        add_marble(ax, ATC, timevar='time', lonvar='longitude', latvar='latitude', add_arrows=False, annotate=True)
        
    if dstdir:
        srcfile_string = ATC.encoding['source'].split("/")[-1].split(".")[0]
        dstfile = f"{srcfile_string}_quicklook{suffix}.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')
    else:
        return fig, ax


def quicklook_CTC(CTC, hmax=20e3, dstdir=None, with_marble=False, show_temperature=False):
            
    nrows=1
    fig, ax = plt.subplots(figsize=(25,7*nrows), nrows=nrows, gridspec_kw={'hspace':0.75})
    
    plot_EC_target_classification(ax, CTC, 'hydrometeor_classification', CTC_category_colors, hmax=hmax)
    
    ax.plot(CTC.time, CTC.surface_elevation, color='k', lw=2.5)

    if show_temperature and ('temperature' in CTC.data_vars):
        add_temperature(ax, CTC)
            
    if with_marble:
        add_marble(ax, CTC, timevar='time', lonvar='longitude', latvar='latitude', add_arrows=False, annotate=True)
        
    if dstdir:
        srcfile_string = CTC.encoding['source'].split("/")[-1].split(".")[0]
        dstfile = f"{srcfile_string}_quicklook.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')
    else:
        return fig, ax


def quicklook_CFMR(CFMR, hmax=20e3, dstdir=None, with_marble=False, show_temperature=False):
            
    nrows=3
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, gridspec_kw={'hspace':0.67})
    
    plot_EC_2D(axes[0], CFMR, 'reflectivity_no_attenuation_correction', "Z", units="dBZ", plot_scale='linear', plot_range=[-40,20], hmax=hmax)
    
    plot_EC_1D(axes[1], CFMR, {'C-FMR':{'xdata':CFMR['time'], 'ydata':CFMR['path_integrated_attenuation']}}, 
                     "CPR path-integrated attenuation", r"$Z$ [dBZ]", timevar='time', include_ruler=False)
    
    plot_EC_1D(axes[2], CFMR, {'C-FMR':{'xdata':CFMR['time'], 'ydata':CFMR['brightness_temperature']}}, 
                     "CPR brightness temperature", r"$T_B$ [K]", timevar='time', include_ruler=False)
    
    add_subfigure_labels(axes)

    if show_temperature and ('temperature' in CFMR.data_vars):
        for ax in axes:
            add_temperature(ax, CFMR)
    
    if with_marble:
        add_marble(axes[0], CFMR, timevar='time', lonvar='longitude', latvar='latitude', add_arrows=False, annotate=True)

    if dstdir:
        srcfile_string = CFMR.encoding['source'].split("/")[-1].split(".")[0]
        dstfile = f"{srcfile_string}_quicklook.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')
    else:
        return fig, axes


def quicklook_CCD(CCD, hmax=20e3, dstdir=None, with_marble=False, show_temperature=False):
            
    nrows=3
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, gridspec_kw={'hspace':0.67})
    
    plot_EC_2D(axes[0], CCD, 'doppler_velocity_best_estimate', "$V_D$", units="ms$^{-1}$", plot_scale='linear', plot_range=[-6,6], cmap=colormaps.litmus, hmax=hmax)
    
    plot_EC_2D(axes[1], CCD, 'sedimentation_velocity_best_estimate', "$V_s$", units="ms$^{-1}$", plot_scale='linear', plot_range=[-6,6], cmap=colormaps.litmus, hmax=hmax)
    
    plot_EC_2D(axes[2], CCD, 'spectrum_width_integrated', "$\sigma_D$", units="ms$^{-1}$", 
               scale_factor=-1, plot_scale='linear', plot_range=[0,10], cmap=colormaps.chiljet2, hmax=hmax)
    
    add_subfigure_labels(axes)

    if show_temperature and ('temperature' in CCD.data_vars):
        for ax in axes:
            add_temperature(ax, CCD)
    
    if with_marble:
        add_marble(axes[0], CCD, timevar='time', lonvar='longitude', latvar='latitude', add_arrows=False, annotate=True)

    if dstdir:
        srcfile_string = CCD.encoding['source'].split("/")[-1].split(".")[0]
        dstfile = f"{srcfile_string}_quicklook.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')
    else:
        return fig, axes


def quicklook_CPRO(CFMR, CCD, CTC, hmax=20e3, dstdir=None, with_marble=False, show_temperature=False):
            
    nrows=3
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, gridspec_kw={'hspace':0.67})
    
    plot_EC_2D(axes[0], CFMR, 'reflectivity_corrected', "Z", units="dBZ", plot_scale='linear', plot_range=[-40,20], hmax=hmax)
    
    plot_EC_2D(axes[1], CCD, 'doppler_velocity_best_estimate', "$V_D$", units="ms$^{-1}$", plot_scale='linear', plot_range=[-6,6], cmap=colormaps.litmus, hmax=hmax)
    
    plot_EC_target_classification(axes[2], CTC, 'hydrometeor_classification', CTC_category_colors, hmax=hmax)
    
    for ax in axes:
        ax.plot(CTC.time, CTC.surface_elevation, color='k', lw=2.5)

    add_subfigure_labels(axes)

    if show_temperature and ('temperature' in CFMR.data_vars):
        for ax in axes:
            add_temperature(ax, CFMR)
    
    if with_marble:
        add_marble(axes[0], CFMR, timevar='time', lonvar='longitude', latvar='latitude', add_arrows=False, annotate=True)

    if dstdir:
        srcfile_string = CFMR.encoding['source'].split("/")[-1].split(".")[0]
        dstfile = f"{srcfile_string}_quicklook.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')
    else:
        return fig, axes


def quicklook_CCLD(CCLD, hmax=20e3, dstdir=None, with_marble=False, show_temperature=False):
            
    nrows=4
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, gridspec_kw={'hspace':0.67})
    
    plot_EC_2D(axes[0], CCLD, 'water_content', "WC", units="kgm$^{-3}$", title="CPR-CLD ice, snow & rain water content", plot_scale='log', plot_range=[1e-6,1e-3], cmap=colormaps.chiljet2, hmax=hmax)
    
    plot_EC_2D(axes[1], CCLD, 'mass_flux', "$R$", scale_factor=3600., units="mmh$^{-1}$", plot_scale='log', plot_range=[1e-3,1e1], cmap=colormaps.chiljet2, hmax=hmax)
    
    plot_EC_2D(axes[2], CCLD, 'characteristic_diameter', "$D_0$", units="m", plot_scale='log', plot_range=[1e-4,2e-3], cmap=colormaps.chiljet2, hmax=hmax)

    plot_EC_2D(axes[3], CCLD, 'liquid_water_content', "$L$", units="kgm$^{-3}$", plot_scale='log', plot_range=[1e-6,1e-3], cmap=colormaps.chiljet2, hmax=hmax)
    
    add_subfigure_labels(axes)

    if "surface_elevation" in CCLD.data_vars:
        for ax in axes:
            ax.plot(CCLD.time, CCLD.surface_elevation, color='k', lw=2.5)
        
    if with_marble:
        add_marble(axes[0], CCLD, timevar='time', lonvar='longitude', latvar='latitude', add_arrows=False, annotate=True)

    if show_temperature and ('temperature' in CCLD.data_vars):
        for ax in axes:
            add_temperature(ax, CCLD)
    
    if dstdir:
        srcfile_string = CCLD.encoding['source'].split("/")[-1].split(".")[0]
        dstfile = f"{srcfile_string}_quicklook.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')
    else: 
        return fig, axes


def quicklook_ACTC(ACTC, resolution='high', hmax=20e3, dstdir=None, with_marble=False, 
                   with_inputs=False, with_hatching=True, show_temperature=False):
              
    if 'med' in resolution:
        suffix='_medium_resolution'
    elif 'low' in resolution:
        suffix='_low_resolution'
    else:
        suffix=''

    if with_inputs:
        nrows=3
        fig, axes = plt.subplots(figsize=(25,8*nrows), nrows=nrows, gridspec_kw={'hspace':0.5})
        
        plot_EC_target_classification(axes[0], ACTC, 'ATLID_target_classification'+suffix, ATC_category_colors, hmax=hmax)
        if show_temperature:
            add_temperature(axes[0], ACTC)
        
        if with_hatching:

            ATC_aerosol_classes = [10,11,12,13,14,15,25,26,27]
            ATC_no_data_classes = [-3, -2,-1]
            ATC_unknown_classes = [101,102,103,104,105,106,107]
            
            _x, _y, is_aerosol, is_no_data, is_unknown = xr.broadcast(ACTC['time'], ACTC.height, 
                                                                      ACTC['ATLID_target_classification'+suffix].isin(ATC_aerosol_classes),
                                                                      ACTC['ATLID_target_classification'+suffix].isin(ATC_no_data_classes),
                                                                      ACTC['ATLID_target_classification'+suffix].isin(ATC_unknown_classes))
            overlay = -2*is_no_data -1*is_unknown + 1*is_aerosol 
            hatches = ['//////', '\\\\\\', '', '....']
            
            _ax = axes[0]
            cs = _ax.contourf(_x, _y, overlay,
                               [-2.5, -1.5, -0.5, 0.5, 1.5], colors=['none', 'none', 'none', 'none'], hatches=hatches)
            
            import matplotlib
            # For each level, we set the color of its hatch 
            for i, collection in enumerate(cs.collections):
                collection.set_edgecolor('k')
            # Doing this also colors in the box around each level
            # We can remove the colored line around the levels by setting the linewidth to 0
            for collection in cs.collections:
                collection.set_linewidth(0.)
            # ------------------------------

            #Draw surface elevation
            _ax.plot(ACTC.time, ACTC.elevation, color='k', lw=2.5)

        plot_EC_target_classification(axes[1], ACTC, 'CPR_target_classification', CTC_category_colors, hmax=hmax)
        if show_temperature:
            add_temperature(axes[1], ACTC)

        if with_hatching:
            
            CTC_no_data_classes = [-1,0]
            CTC_unknown_classes = [20]
            _x, _y, is_no_data, is_unknown = xr.broadcast(ACTC['time'], ACTC.height, 
                                                                          ACTC['CPR_target_classification'].isin(CTC_no_data_classes),
                                                                          ACTC['CPR_target_classification'].isin(CTC_unknown_classes))
            overlay = -2*is_no_data -1*is_unknown 
            hatches = ['//////', '\\\\\\', '']
            
            _ax = axes[1]
            cs = _ax.contourf(_x, _y, overlay,
                               [-2.5, -1.5, -0.5, 0.5], colors=['none', 'none', 'none'], hatches=hatches)
            
            import matplotlib
            # For each level, we set the color of its hatch 
            for i, collection in enumerate(cs.collections):
                collection.set_edgecolor('k')
            # Doing this also colors in the box around each level
            # We can remove the colored line around the levels by setting the linewidth to 0
            for collection in cs.collections:
                collection.set_linewidth(0.)
            # ------------------------------
            
            _ax.plot(ACTC.time, ACTC.elevation, color='k', lw=2.5)
        
        plot_EC_target_classification(axes[2], ACTC, 'synergetic_target_classification'+suffix, ACTC_category_colors, hmax=hmax)
        
        if show_temperature:
            add_temperature(axes[2], ACTC)
            
        if with_hatching:
            
            ACTC_aerosol_classes = [26,27,28,29,30,31,32,33,34]
            ACTC_no_data_classes = [-1, 0]
            ACTC_unknown_classes = [25,7]
            _x, _y, is_aerosol, is_no_data, is_unknown = xr.broadcast(ACTC['time'], ACTC.height, 
                                                                      ACTC['synergetic_target_classification'+suffix].isin(ACTC_aerosol_classes),
                                                                      ACTC['synergetic_target_classification'+suffix].isin(ACTC_no_data_classes),
                                                                      ACTC['synergetic_target_classification'+suffix].isin(ACTC_unknown_classes))
            overlay = -2*is_no_data -1*is_unknown + 1*is_aerosol 
            hatches = ['//////', '\\\\\\', '', '....']
            
            _ax = axes[2]
            cs = _ax.contourf(_x, _y, overlay,
                               [-2.5, -1.5, -0.5, 0.5, 1.5], colors=['none', 'none', 'none', 'none'], hatches=hatches)
            
            import matplotlib
            # For each level, we set the color of its hatch 
            for i, collection in enumerate(cs.collections):
                collection.set_edgecolor('k')
            # Doing this also colors in the box around each level
            # We can remove the colored line around the levels by setting the linewidth to 0
            for collection in cs.collections:
                collection.set_linewidth(0.)
            # ------------------------------
            
            _ax.plot(ACTC.time, ACTC.elevation, color='k', lw=2.5)

        if with_hatching:
            import matplotlib.patches as mpatches
            p1 = mpatches.Patch( facecolor='1.0', edgecolor='k', linewidth=0, hatch=r'//////', label='no data')
            p2 = mpatches.Patch( facecolor='0.9', edgecolor='k', linewidth=0, hatch='\\\\\\',  label='unknown')
            p3 = mpatches.Patch( facecolor='0.8', edgecolor='k', linewidth=0, hatch='',        label='hydrometeors')
            p4 = mpatches.Patch( facecolor='0.7', edgecolor='k', linewidth=0, hatch='....',    label='aerosol')
    
            patches = [p1, p2, p3, p4]
                
            for ax in axes:
                ax.legend(handles=patches, frameon=False, loc='lower right', bbox_to_anchor=(1.33,0), fontsize='xx-small', 
                         labelspacing=0.0, handlelength=1.5)
        
        if with_marble:
            add_marble(axes[0], ACTC, timevar='time', lonvar='longitude', latvar='latitude', add_arrows=False, annotate=True)

        add_subfigure_labels(axes)
    
    else:
        nrows=1
        fig, ax = plt.subplots(figsize=(25,7*nrows), nrows=nrows, gridspec_kw={'hspace':0.75})
        
        plot_EC_target_classification(ax, ACTC, 'synergetic_target_classification'+suffix, ACTC_category_colors, hmax=hmax)
        if show_temperature:
            add_temperature(ax, ACTC)
            
        if with_hatching:
            
            ACTC_aerosol_classes = [26,27,28,29,30,31,32,33,34]
            ACTC_no_data_classes = [0]
            ACTC_unknown_classes = [25,7,-1]
            _x, _y, is_aerosol, is_no_data, is_unknown = xr.broadcast(ACTC['time'], ACTC.height, 
                                                                      ACTC['synergetic_target_classification'+suffix].isin(ACTC_aerosol_classes),
                                                                      ACTC['synergetic_target_classification'+suffix].isin(ACTC_no_data_classes),
                                                                      ACTC['synergetic_target_classification'+suffix].isin(ACTC_unknown_classes))
            overlay = -2*is_no_data -1*is_unknown + 1*is_aerosol 
            hatches = ['//////', '\\\\\\', '', '....']
            
            _ax = ax
            cs = _ax.contourf(_x, _y, overlay,
                               [-2.5, -1.5, -0.5, 0.5, 1.5], colors=['none', 'none', 'none', 'none'], hatches=hatches)
            
            import matplotlib
            # For each level, we set the color of its hatch 
            for i, collection in enumerate(cs.collections):
                collection.set_edgecolor('k')
            # Doing this also colors in the box around each level
            # We can remove the colored line around the levels by setting the linewidth to 0
            for collection in cs.collections:
                collection.set_linewidth(0.)
            # ------------------------------
            
            _ax.plot(ACTC.time, ACTC.elevation, color='k', lw=2.5)
            
            import matplotlib.patches as mpatches
            p1 = mpatches.Patch( facecolor='1.0', edgecolor='0.5', linewidth=0, hatch=r'//////', label='no data')
            p2 = mpatches.Patch( facecolor='0.9', edgecolor='0.5', linewidth=0, hatch='\\\\\\',  label='unknown')
            p3 = mpatches.Patch( facecolor='0.8', edgecolor='0.5', linewidth=0, hatch='',        label='hydrometeors')
            p4 = mpatches.Patch( facecolor='0.7', edgecolor='0.5', linewidth=0, hatch='....',    label='aerosol')
    
            patches = [p1, p2, p3, p4]
            
            _ax.legend(handles=patches, frameon=False, loc='lower right', bbox_to_anchor=(1.33,0), fontsize='xx-small', 
                         labelspacing=0.0, handlelength=1.5)
        
        if with_marble:
            add_marble(ax, ACTC, timevar='time', lonvar='longitude', latvar='latitude', add_arrows=False, annotate=True)

        
    if dstdir:
        srcfile_string = ACTC.encoding['source'].split("/")[-1].split(".")[0]
        dstfile = f"{srcfile_string}_quicklook{suffix}.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')
    else:
        if with_inputs:
            return fig, axes
        else:
            return fig, ax


def quicklook_ACMCAP(ACMCAP, hmax=20e3, dstdir=None, with_marble=False, show_retrievals=True, show_temperature=True, show_surface=True):
    if show_retrievals:
        return quicklook_ACMCAP_retrievals(ACMCAP, hmax=hmax, dstdir=dstdir, with_marble=with_marble, show_temperature=show_temperature, show_surface=show_surface)
    else:
        return quicklook_ACMCAP_forward(ACMCAP, hmax=hmax, dstdir=dstdir, with_marble=with_marble, show_temperature=show_temperature, show_surface=show_surface)


def quicklook_ACMCAP_retrievals(ACMCAP, hmax=20e3, dstdir=None, with_marble=False, show_temperature=True, show_surface=True):
            
    nrows=5
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, gridspec_kw={'hspace':0.67})
        
    plot_EC_2D(axes[0], ACMCAP, 'ice_water_content', "IWC", units="kgm$^{-3}$", plot_scale='log', plot_range=[1e-7,10e-3], cmap=colormaps.chiljet2, hmax=hmax)
    
    plot_EC_2D(axes[1], ACMCAP, 'ice_effective_radius', "$r_\mathrm{eff}$", scale_factor=1e6, units="m", plot_scale='linear', plot_range=[0,200], cmap=colormaps.chiljet2, hmax=hmax)
    
    plot_EC_2D(axes[2], ACMCAP, 'rain_rate', "$R$", scale_factor=3600., units="mmh$^{-1}$", plot_scale='log', plot_range=[1e-3,2e1], cmap=colormaps.chiljet2, hmax=hmax)

    if 'liquid_water_content' in ACMCAP.data_vars:
        plot_EC_2D(axes[3], ACMCAP, 'liquid_water_content', "$L$", units="kgm$^{-3}$", plot_scale='log', plot_range=[1e-7,2e-3], cmap=colormaps.chiljet2, hmax=hmax)
    else:
        plot_EC_2D(axes[3], ACMCAP, 'liquid_detected_by_lidar_water_content', "$L$", units="kgm$^{-3}$", plot_scale='log', plot_range=[1e-7,2e-3], cmap=colormaps.chiljet2, hmax=hmax)
        
    
    plot_EC_2D(axes[4], ACMCAP, 'aerosol_extinction', r"$\alpha$", units="m$^{-1}$", plot_scale='log', plot_range=[1e-7,1e-3], cmap=colormaps.chiljet2, hmax=hmax)
    
    add_subfigure_labels(axes)
    
    if with_marble:
        add_marble(axes[0], ACMCAP, timevar='time', lonvar='longitude', latvar='latitude', add_arrows=False, annotate=True)

    if show_temperature and ('temperature' in ACMCAP.data_vars):
        for ax in axes:
            add_temperature(ax, ACMCAP)

    if show_surface:
        for ax in axes:
            add_surface(ax, ACMCAP, elevation_var='elevation')
    
    if dstdir:
        srcfile_string = ACMCAP.encoding['source'].split("/")[-1].split(".")[0]
        dstfile = f"{srcfile_string}_retrieval_quicklook.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')
        return fig, axes
    else:
        return fig, axes


def quicklook_ACMCAP_forward(ACMCAP, hmax=20e3, dstdir=None, with_marble=False, show_temperature=True, show_surface=True):

    if "CPR_brightness_temperature" in ACMCAP.data_vars:
        nrows=12
    else:    
        nrows=11
    fig, axes = plt.subplots(figsize=(25,7*nrows), nrows=nrows, gridspec_kw={'hspace':0.75})
    
    i=0
    plot_EC_2D(axes[i], ACMCAP, 'ATLID_backscatter_mie', r"$\beta_\mathrm{Mie}$", units="sr$^{-1}$m$^{-1}$", 
               plot_scale='log', plot_range=[1e-8,1e-5], fill_value=1e-9, min_value=1e-9, smoother=dict(along_track=5),
               cmap=colormaps.calipso_smooth, hmax=hmax)
    
    i+=1
    plot_EC_2D(axes[i], ACMCAP, 'ATLID_backscatter_mie_forward', r"$\beta_\mathrm{Mie}$", units="sr$^{-1}$m$^{-1}$", 
               plot_scale='log', plot_range=[1e-8,1e-5], 
               cmap=colormaps.calipso_smooth, hmax=hmax)
    
    i+=1
    plot_EC_2D(axes[i], ACMCAP, 'ATLID_backscatter_rayleigh', r"$\beta_\mathrm{Ray}$", units="sr$^{-1}$m$^{-1}$", 
               plot_scale='log', plot_range=[1e-8,1e-5], fill_value=1e-9, min_value=1e-9, smoother=dict(along_track=5),
               cmap=colormaps.chiljet3, hmax=hmax)
    
    i+=1
    plot_EC_2D(axes[i], ACMCAP, 'ATLID_backscatter_rayleigh_forward',r"$\beta_\mathrm{Ray}$", units="sr$^{-1}$m$^{-1}$", 
               plot_scale='log', plot_range=[1e-8,1e-5], 
               cmap=colormaps.chiljet3, hmax=hmax)
    
    i+=1
    plot_EC_2D(axes[i], ACMCAP, 'CPR_reflectivity_factor', "Z", units="dBZ", plot_scale='linear', plot_range=[-40,20], 
               cmap=colormaps.chiljet2, hmax=hmax)
    
    i+=1
    plot_EC_2D(axes[i], ACMCAP, 'CPR_reflectivity_factor_forward', "Z", units="dBZ", plot_scale='linear', plot_range=[-40,20], 
               cmap=colormaps.chiljet2, hmax=hmax)

    i+=1
    plot_EC_2D(axes[i], ACMCAP, 'CPR_doppler_velocity', "V$_{D}$", units="m/s", plot_scale='linear', plot_range=[-6,6], 
               cmap='litmus', hmax=hmax)
    
    i+=1
    plot_EC_2D(axes[i], ACMCAP, 'CPR_doppler_velocity_forward', "V$_{D}$", units="m/s", plot_scale='linear', plot_range=[-6,6], 
               cmap='litmus', hmax=hmax)
    
    i+=1
    plot_EC_1D(axes[i], ACMCAP, {'PIA (obs)':{'xdata':ACMCAP['time'], 
                                              'ydata':ACMCAP['CPR_path_integrated_attenuation']},
                                 'PIA (fwd)':{'xdata':ACMCAP['time'], 
                                              'ydata':ACMCAP['CPR_path_integrated_attenuation_forward']}}, 
               "CPR path-integrated attenuation", r"PIA [dB]", timevar='time', include_ruler=False)

    if "CPR_brightness_temperature" in ACMCAP.data_vars:
        i+=1
        plot_EC_1D(axes[i], ACMCAP, {'BT (obs)':{'xdata':ACMCAP['time'], 
                                                  'ydata':ACMCAP['CPR_brightness_temperature']},
                                     'BT (fwd)':{'xdata':ACMCAP['time'], 
                                                  'ydata':ACMCAP['CPR_brightness_temperature_forward']}}, 
                   "CPR brightness temperature", r"BT [K]", timevar='time', include_ruler=False)
        _y0, _y1 = axes[i].get_ylim()
        if _y0 <= 0:
            _y0 = 200
        axes[i].set_ylim(_y1, _y0)
        
    i+=1
    plot_EC_1D(axes[i], ACMCAP, {'TIR1 (obs)':{'xdata':ACMCAP['time'], 
                                               'ydata':ACMCAP['MSI_longwave_brightness_temperature'].isel(MSI_longwave_channel=0)},
                                 'TIR1 (fwd)':{'xdata':ACMCAP['time'], 
                                               'ydata':ACMCAP['MSI_longwave_brightness_temperature_forward'].isel(MSI_longwave_channel=0)}}, 
               "MSI thermal IR brightness temperature", r"BT [K]", timevar='time', include_ruler=False)
    _y0, _y1 = axes[i].get_ylim()
    if _y0 <= 0:
        _y0 = 200
    axes[i].set_ylim(_y1, _y0)

    i+=1
    if (ACMCAP['MSI_shortwave_albedo'] == 0).mean() == 1:
        plot_EC_1D(axes[i], ACMCAP, {'TIR3 (obs)':{'xdata':ACMCAP['time'], 
                                                   'ydata':ACMCAP['MSI_longwave_brightness_temperature'].isel(MSI_longwave_channel=-1)},
                                     'TIR3 (fwd)':{'xdata':ACMCAP['time'], 
                                                   'ydata':ACMCAP['MSI_longwave_brightness_temperature_forward'].isel(MSI_longwave_channel=-1)}}, 
               "MSI thermal IR brightness temperature", r"BT [K]", timevar='time', include_ruler=False)
        _y0, _y1 = axes[i].get_ylim()
        axes[i].set_ylim(_y1, _y0)

    else:
        plot_EC_1D(axes[i], ACMCAP, {'A (obs)':{'xdata':ACMCAP['time'], 
                                                  'ydata':np.pi*ACMCAP['MSI_shortwave_albedo']},
                                     'A (fwd)':{'xdata':ACMCAP['time'], 
                                                  'ydata':np.pi*ACMCAP['MSI_shortwave_albedo_forward']}}, 
                   "MSI solar albedo", r"$A$ [-]", timevar='time', include_ruler=False)
        
    add_subfigure_labels(axes)
    
    if with_marble:
        add_marble(axes[0], ACMCAP, timevar='time', lonvar='longitude', latvar='latitude', add_arrows=False, annotate=True)

    if show_temperature and ('temperature' in ACMCAP.data_vars):
        for ax in axes[:8]:
            add_temperature(ax, ACMCAP)

    if show_surface:
        for ax in axes[:8]:
            add_surface(ax, ACMCAP, elevation_var='elevation')
    
    if dstdir:
        srcfile_string = ACMCAP.encoding['source'].split("/")[-1].split(".")[0]
        dstfile = f"{srcfile_string}_forward_quicklook.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')
        return fig, axes
    else:
        return fig, axes


def quicklook_MCM(MCM, dstdir=None, with_marble=False, 
                  figure_width=25, panel_height=5):
            
    nrows=3
    fig, axes = plt.subplots(figsize=(figure_width,panel_height*nrows), nrows=nrows, 
                             gridspec_kw={'hspace':1.33})
    
    hmax=384
    
    plot_EC_2D(axes[0], MCM, 'cloud_mask', "confidence", plot_range=[0,3], plot_scale='linear', cmap='Reds', hmax=hmax, units='-', across_track=True, 
                heightvar='across_track', latvar='selected_latitude', lonvar='selected_longitude')
    add_nadir_track(axes[0])
    
    plot_EC_target_classification(axes[1], MCM, 'cloud_type', MCM_type_category_colors, hmax=hmax, across_track=True, 
                heightvar='across_track', latvar='selected_latitude', lonvar='selected_longitude')
    add_nadir_track(axes[1])
    
    plot_EC_target_classification(axes[2], MCM, 'cloud_phase', MCM_maskphase_category_colors, hmax=hmax, across_track=True, 
                heightvar='across_track', latvar='selected_latitude', lonvar='selected_longitude')
    add_nadir_track(axes[2])

    add_subfigure_labels(axes)

    if with_marble:
        add_marble(axes[0], ACMCAP, timevar='time', lonvar='selected_longitude', latvar='selected_latitude', add_arrows=False, annotate=True)
        
    if dstdir:
        srcfile_string = MCM.encoding['source'].split("/")[-1].split(".")[0]
        dstfile = f"{srcfile_string}_quicklook.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')
    else:
        return fig, axes

def plot_EC_target_classificationxw(ax, ds, varname, category_colors, 
                                  hmax=15e3, label_fontsize='xx-small', 
                                  processor=None, title=None, title_prefix=None,
                                  savefig=False, dstdir="./", show_latlon=True,
                                  use_latitude=False, dark_mode=False, use_localtime=True,
                                  timevar='time', heightvar='height', latvar='latitude', lonvar='longitude', 
                                  across_track=False, line_break=None, fillna=None):
   
   if processor is None:
       processor = "-".join([t.replace("_","") for t in ds.encoding['source'].split("/")[-1][9:16].split('_', maxsplit=1)])
       
   print(ds.keys())
   if title is None:
       long_name = ds[varname].attrs['long_name'].split(" ")
       #Removing capitalizations unless it's an acronym
       for i, l in enumerate(long_name):
           if not l.isupper():
               long_name[i] = l.lower()
       if title_prefix:
           long_name = [title_prefix.strip()] + long_name
       else:
           title_prefix=""
       long_name = " ".join(long_name)
       title = f"{processor} {title_prefix}{long_name}"

       if len(title) > 50:
           title_parts = title.split(" ")
           title = "\n".join([" ".join(title_parts[:4]), " ".join(title_parts[4:])])
   
   cleanup_category = lambda s: s.strip().replace('possible', 'poss.').replace('supercooled', "s\'cooled").replace("stratospheric", 'strat.').replace('extinguished', 'ext.').replace('precipitation', 'precip.').replace('and', '&').replace('unknown', 'unk.').replace('precipitation', 'precip.')
   
   if "\n" in ds[varname].attrs['definition']:
       definitions = ds[varname].attrs['definition']
       if definitions.endswith("\n"):
           definitions = definitions[:-1]
       categories = [cleanup_category(s) for s in definitions.split('\n')]
   else:
       #C-TC uses comma-separated "definition" attribute, but also uses commas within definitions.
       #categories = [cleanup_category(s) for s in ds[varname].attrs['definition'].replace("ground clutter,", "ground clutter;").split(',')]
       #Same for M-COP's quality_status.
       categories = [cleanup_category(s) for s in ds[varname].attrs['definition'].replace("ground clutter,", "ground clutter;").replace('valid, quality','valid; quality').replace('valid, degraded','valid; degraded').split(',')]

   categories = [c.replace("_", " ") for c in categories]
   
   import pandas as pd
   if line_break is not None: categories = [linebreak(c, line_break) for c in categories]
   if ':' in categories[0]:
       try:               # try/except added due to bit# instead of integers in M-CM *_quality_status ... only for ':',
                          #     could be adapted for '=' definitions too, if needed or for completeness...
           first_c = int(categories[0].split(":")[0])
           last_c  = int(categories[-1].split(":")[0])
           u = np.array([int(c.split(':')[0]) for c in categories])

       except ValueError:
           first_c = int(categories[0].split(":")[0].split('bit')[-1])-1
           last_c  = int(categories[-1].split(":")[0].split('bit')[-1])-1
           u = 2**np.array([int(c.split(':')[0].split('bit')[-1])-1 for c in categories])
       categories_formatted = [f"${c.split(':')[0]}$:{c.split(':')[1]}" for c in categories]

   elif '=' in categories[0]:
       first_c = int(categories[0].split("=")[0])
       last_c  = int(categories[-1].split("=")[0])
       u = np.array([int(c.split('=')[0]) for c in categories])
       categories_formatted = [f"${c.split('=')[0]}$:{c.split('=')[1]}" for c in categories]

   else:
       print("category values are not included within categories")

   # due to M-CM categories that are not strictly monotonically increasing in attribute definition: 0,1,2,..., -127
   idx                  = np.argsort(u)
   u                    = u[idx]
   categories_formatted = list(np.array(categories_formatted)[idx]) # end

   bounds = np.concatenate(([u.min()-1], u[:-1]+np.diff(u)/2. ,[u.max()+1]))

   from matplotlib.colors import ListedColormap, BoundaryNorm
   norm = BoundaryNorm(bounds, len(bounds)-1)
   cmap = ListedColormap(sns.color_palette(category_colors[:len(u)]).as_hex())

   if use_latitude:
       if fillna is None:
           _l, _h, _z = xr.broadcast(ds[latvar][::44], ds[heightvar][::44], ds[varname][::44])
       else:
           _l, _h, _z = xr.broadcast(ds[latvar], ds[heightvar], ds[varname].fillna(fillna))

       if (np.isnan(_h).sum() > 0):
           _cm = ax.pcolor(_l, _h, _z, norm=norm, cmap=cmap)
       else:
           _cm = ax.pcolormesh(_l, _h, _z, norm=norm, cmap=cmap)
           
   else:
       if fillna is None:
           _t, _h, _z = xr.broadcast(ds[timevar][::44], ds[heightvar][::44], ds[varname][::44])
           #print('here1, _z=',_z)
           #print('here2, _z=',_z[:])
           #print('here3, _z=',_z.values)
           _z.values = np.where(_z.values<12,_z.values,np.nan)
       else:
           _t, _h, _z = xr.broadcast(ds[timevar], ds[heightvar], ds[varname].fillna(fillna))
       if (np.isnan(_h).sum() > 0):
           _cm = ax.pcolor(_t, _h, _z, norm=norm, cmap=cmap)
       else:
           _cm = ax.pcolormesh(_t, _h, _z, norm=norm, cmap=cmap)
   
   _cb = add_colorbar(ax, _cm, '', horz_buffer=0.01)
   _cb.set_ticks(bounds[:-1]+np.diff(bounds)/2.)
   _cb.ax.set_yticklabels(categories_formatted, fontsize=label_fontsize)
   
   format_plot(ax, ds, title, hmax, dark_mode=dark_mode, use_localtime=use_localtime, latvar=latvar, lonvar=lonvar, across_track=across_track)
   
   if savefig:
       import os
       dstfile = f"{product_code}_{varname}.png"
       fig.savefig(os.path.join(dstdir,dstfile), bbox_inches='tight')
 
from scipy.ndimage import gaussian_filter

def plot_EC_2Dxw(ax, ds, varname,varname2, label, 
             plot_where=True, scale_factor=1,
             hmax=15e3, plot_scale=None, plot_range=None, cmap=None,
             units=None, processor=None, title=None, title_prefix="",
             timevar='time', heightvar='height', latvar='latitude', lonvar='longitude',
    	     across_track=False, dark_mode=False, use_localtime=True, smoothx=0):

    sns.set_style('ticks')
    sns.set_context('poster')
    
    import pandas as pd
    
    if plot_scale is None:
        plot_scale = ds[varname].attrs['plot_scale']
    
    if plot_range is None:
        plot_range = ds[varname].attrs['plot_range']
        
    if 'log' in plot_scale:
        norm=LogNorm(plot_range[0], plot_range[-1])
    else:
        norm=Normalize(plot_range[0], plot_range[-1])
    
    if cmap is None:
        cmap=colormaps.chiljet2
    
    if processor is None:
        processor = "-".join([t.replace("_","") for t in ds.encoding['source'].split("/")[-1][9:16].split('_', maxsplit=1)])
    
    if units is None:
        units = ds[varname].attrs['units']
    
    if title is None:
        long_name = ds[varname].attrs['long_name'].split(" ")
        #Removing capitalizations unless it's an acronym
        for i, l in enumerate(long_name):
            if not l.isupper():
                long_name[i] = l.lower()
        if title_prefix:
            long_name = [title_prefix.strip()] + long_name
        long_name = " ".join(long_name)
    
        title = f"{processor} {long_name}"

        if len(title) > 50:
            title_parts = title.split(" ")
            title = "\n".join([" ".join(title_parts[:4]), " ".join(title_parts[4:])])
    if smoothx >0:
        print("SSSSSS" ,smoothx)
    # White background 
    _t, _h, _z = xr.broadcast(ds[timevar], ds[heightvar].fillna(0.), scale_factor*ds[varname]/ds[varname2])


    sigma_time=1
    sigma_height=0.4
    # Apply a Gaussian smoothing filter to the data
    z_smooth = xr.DataArray(gaussian_filter(_z, sigma=(sigma_time, sigma_height)), dims=_z.dims, coords=_z.coords)

    # Optionally, apply a mask or condition
    _z_filtered = z_smooth.where(plot_where)

    # Create the plot
    #fig, ax = plt.subplots()
    _cm = ax.pcolormesh(_t, _h, _z_filtered, norm=norm, cmap=cmap)

    #z_smooth = gaussian_filter(_z, sigma=1)
#    _z=z_smooth
    #_z_masked = np.ma.masked_where(~plot_where, _z_smooth)

   # _cm = ax.pcolormesh(_t, _h, _z.where(plot_where), norm=norm, cmap=cmap)
   # _cm = ax.pcolormesh(_t, _h, _z_masked, norm=norm, cmap=cmap)

    if len(units) > 0:
        cb_label = f"{label} [{units}]"
        if len(cb_label) > 25:
            add_colorbar(ax, _cm, f"{label}\n[{units}]", horz_buffer=0.01, width_ratio='1%')
        else:
            add_colorbar(ax, _cm, cb_label, horz_buffer=0.01, width_ratio='1%')
    else:
        add_colorbar(ax, _cm, f"{label}", horz_buffer=0.01, width_ratio='1%')

    format_plot(ax, ds, title, hmax, dark_mode=dark_mode, timevar=timevar, heightvar=heightvar,
                latvar=latvar, lonvar=lonvar, across_track=across_track, use_localtime=use_localtime)
 
def plot_EC_1Dxw(ax, ds, plot1D_dict, title, ylabel, 
               yscale='linear', y0_ruler=0.9, ylim=[None,None],
              timevar='time', latvar='latitude', lonvar='longitude', dim_name='along_track',
              include_ruler=False):

    for l, d in plot1D_dict.items():
        color = None if ('color' not in d) else d['color']
        markersize = 3 if ('markersize' not in d) else d['markersize']
        marker = '.' if ('marker' not in d) else d['marker']
        zorder = None if ('zorder' not in d) else d['zorder']
        scale  = 1 if ('scale' not in d) else d['scale']
                
        ax.plot(d['xdata'], d['ydata'],
                label=l, lw=0, marker=marker, markersize=markersize, color='black', zorder=zorder)

    ax.set_yscale(yscale)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.legend(frameon=False, fontsize='xx-small', markerscale=10, bbox_to_anchor=(1,1), loc='upper left')

    format_plot_1D(ax, ds, title, y0_ruler=y0_ruler,
                   timevar=timevar, latvar=latvar, lonvar=lonvar, dim_name=dim_name, include_ruler=include_ruler)
 
def quicklook_varxw(ds, varname, cmap, hmax=20e3, resolution='high', dstdir=None):

    if 'med' in resolution:
        suffix='_medium_resolution' 
    elif 'low' in resolution:
        suffix='_low_resolution'
    else:                           
        suffix=''
    
    nrows=1 
    fig, ax = plt.subplots(figsize=(25,7*nrows), nrows=nrows, gridspec_kw={'hspace':0.75})

    #hmax=20e3
    
    plot_EC_target_classification(ax, ds, varname + suffix, cmap, hmax=hmax)

    if dstdir:
        srcfile_string = ds.encoding['source'].split("/")[-1].split(".")[0]
        dstfile = f"{srcfile_string}_quicklook_{varname}{suffix}.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')


def quicklook_AFM(AFM, hmax=30e3, dstdir=None):
    import matplotlib.colors as mcolors
    fig, ax = plt.subplots(figsize=(25,5))
    rgb= np.matrix([[0,0,0],
                   [255,255,255],
                   [ 0, 23, 190],
                   [ 7,187,250],
                   [ 63 ,242, 254],
                   [ 100, 248, 254],
                   [ 130, 250, 254],
                   [ 150, 246, 254],
                   [ 190, 190, 190],
                   [ 242, 219, 15],
                   [ 245, 173, 26],
                   [ 202, 112, 12],
                   [ 255, 17, 0],
                   [ 165, 19, 23], 
                   ])
    cm = mcolors.ListedColormap(rgb/255.0)

    plot_EC_2D(ax, AFM, 'featuremask', "class", cmap=cm, plot_scale='linear', plot_range=[-3,10], units='', hmax=hmax)

    if dstdir:
        srcfile_string = AFM.encoding['source'].split("/")[-1].split(".")[0]
        dstfile = f"{srcfile_string}_quicklook.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')

def quicklook_ATCxw(ATC, hmax=20e3, resolution='high', dstdir=None):
        
    if 'med' in resolution:
        suffix='_medium_resolution'
    elif 'low' in resolution:
        suffix='_low_resolution'
    else:
        suffix=''
    
    nrows=1
    fig, ax = plt.subplots(figsize=(25,7*nrows), nrows=nrows, gridspec_kw={'hspace':0.75})
    
    #hmax=20e3
    
    plot_EC_target_classificationxw(ax, ATC, 'classification' + suffix, ATC_category_colors, hmax=hmax)
    
    if dstdir:
        srcfile_string = ATC.encoding['source'].split("/")[-1].split(".")[0]
        dstfile = f"{srcfile_string}_quicklook{suffix}.png"
        fig.savefig(f"{dstdir}/{dstfile}", bbox_inches='tight')



