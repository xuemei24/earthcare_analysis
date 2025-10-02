import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_on_orthographic(lons, lats, fig_name, fig_title,
                        central_longitude=0,
                        central_latitude=0, 
                        ax=None, globe=None):
    """
    Plots latitude and longitude points on an orthographic projection.

    Args:
        lons (list): List of longitudes.
        lats (list): List of latitudes.
        ax (matplotlib.axes._axes.Axes, optional): Axes object to plot on.
            If None, a new figure and axes will be created. Defaults to None.
        globe (cartopy.crs.Globe, optional): Globe object to customize the Earth's shape.
            Defaults to None.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(
                                                        central_longitude=central_longitude, 
                                                        central_latitude=central_latitude,
                                                        globe=globe))

    ax.set_global()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

    ax.plot(lons, lats, linewidth=5, color='red',
               transform=ccrs.PlateCarree(), zorder=10)
    ax.set_title(fig_title)

    fig.savefig(fig_name)
    plt.close(fig)
