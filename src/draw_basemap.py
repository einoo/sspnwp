import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"


def draw_basemap():
    fig = plt.figure(constrained_layout=True, figsize=(3.5, 2.0), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
    ax.set_global()  # Necessary, for the subsequent plot

    m = ax.gridlines(draw_labels=True)

    m.top_labels = False
    m.right_labels = False
    m.xlines = False
    m.ylines = False

    m.xformatter = LONGITUDE_FORMATTER
    m.yformatter = LATITUDE_FORMATTER

    m.xlabel_style = {"size": 5, "color": "k"}
    m.ylabel_style = {"size": 5, "color": "k"}

    return fig, ax
