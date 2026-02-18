from ectools.ectools import ecplot as ecplt
import numpy as np
import seaborn as sns
from ectools.ectools import ecplot

line_break = None
def ecplt_cmap(ds,varname,line_break = None):
    cleanup_category = lambda s: s.strip().replace('possible', 'poss.').replace('supercooled', "s\'cooled").replace("stratospheric", 'strat.').replace('extinguished', 'ext.').replace('precipitation', 'precip.').replace('and', '&').replace('unknown', 'unk.').replace('precipitation', 'precip.')

    if "\n" in ds[varname].attrs['definition']:
        definitions = ds[varname].attrs['definition']
        if definitions.endswith("\n"):
            definitions = definitions[:-1]
        categories = [cleanup_category(s) for s in definitions.split('\n')]
    else:
        # C-TC uses comma-separated "definition" attribute, but also uses commas within definitions.
        # categories = [cleanup_category(s) for s in ds[varname].attrs['definition'].replace("ground clutter,", "ground clutter;").split(',')]
        # Same for M-COP's quality_status.
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
    cmap = ListedColormap(sns.color_palette(ecplot.ATC_category_colors[:len(u)]).as_hex())
    return cmap,bounds,categories_formatted,norm
