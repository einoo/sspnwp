import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from cutoff_radius import cutoff_radius
from matplotlib.colors import ListedColormap

from define_speedy_grid import define_speedy_grid
from draw_basemap import draw_basemap
from read_obs_station import read_obs_station

plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"


NLAT = 48  # number of grids in latitude
NLON = 96  # number of grids in longitude
REC = 18  # 4th layer temperature
CENTER = np.array([-31.545, 60.0])
R = 900
OBSN = 4

# Choose colormap
cmap = plt.cm.YlOrRd
# Get the colormap colors
my_cmap = cmap(np.arange(cmap.N))
# Define the alphas in the range from 0 to 1
alphas = np.linspace(0, 1.0, cmap.N)
# Define the background as white
BG = np.asarray([1.0, 1.0, 1.0])
# Mix the colors with the background
for i in range(cmap.N):
    my_cmap[i, :-1] = my_cmap[i, :-1] * alphas[i] + BG * (1.0 - alphas[i])
# Create new colormap which mimics the alpha values
my_cmap = ListedColormap(my_cmap)
# Boundariy
bounds = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 10])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
# (1) Draw speedy grids
lat_spd, lon_spd = define_speedy_grid(NLAT, NLON)
latspd, lonspd = np.meshgrid(lat_spd, lon_spd)

# (2) Draw raob observing network
obsyr, obsxr, obsn = read_obs_station(
    "../data/station_raob.tbl", lat_spd, lon_spd)

# create the mesh in xy direction
x = []
x.append(0)
for i in range(len(lon_spd) - 1):
    x.append((lon_spd[i] + lon_spd[i + 1]) / 2.0)
x.append(360)
y = []
y.append(-90)
for i in range(len(lat_spd) - 1):
    y.append((lat_spd[i] + lat_spd[i + 1]) / 2.0)
y.append(90)
X, Y = np.meshgrid(x, y)

# plot figures
fig = plt.figure(constrained_layout=True, figsize=(3.5, 2.0), dpi=300)
ax1 = fig.add_subplot(
    2, 2, 1, projection=ccrs.PlateCarree(central_longitude=180))
ax1.add_feature(cfeature.COASTLINE, linewidth=0.3)
ax1.set_extent([180, 300, -80, 5])  # Necessary, for the subsequent plot

m1 = ax1.gridlines(draw_labels=True)

m1.top_labels = False
m1.right_labels = False
m1.xlines = False
m1.ylines = False

m1.xformatter = LONGITUDE_FORMATTER
m1.yformatter = LATITUDE_FORMATTER

m1.xlabel_style = {"size": 4, "color": "k"}
m1.ylabel_style = {"size": 4, "color": "k"}

# ax1.plot(lonspd, latspd, "k,", ms=1)  # grids
# center
ax1.plot(CENTER[1], CENTER[0], "o", color="C2", mew=0.5, lw=0.5, ms=2)

if R < 1500:
    lat_radius, lon_radius = cutoff_radius(CENTER, R)
    ax1.plot(lon_radius, lat_radius, "-", color="C0", lw=0.5)
if R > 1500 and R < 5500:
    lat_radius, lon_radius = cutoff_radius(CENTER, R)
    ax1.plot(lon_radius, lat_radius, ".", color="C0", ms=0.5)

# raob
ax1.plot(obsxr, obsyr, "kx", mew=0.5, lw=0.5, ms=2)
# uniform distribution
obsy, obsx, _ = read_obs_station(
    "../data/Unif/unif_R%05d_N%03d.tbl" % (R, OBSN), lat_spd, lon_spd
)
obs = np.zeros((NLON, NLAT))
obs[:] = np.nan

for i in range(OBSN):
    obs[np.where(lon_spd == obsx[i]), np.where(lat_spd == obsy[i])] = 100.0

pcm = ax1.pcolormesh(X, Y, obs.T, cmap=my_cmap, vmax=8, vmin=1)

###################
ax2 = fig.add_subplot(
    2, 2, 2, projection=ccrs.PlateCarree(central_longitude=180))
ax2.add_feature(cfeature.COASTLINE, linewidth=0.3)
ax2.set_extent([180, 300, -80, 5])  # Necessary, for the subsequent plot

m1 = ax2.gridlines(draw_labels=True)

m1.top_labels = False
m1.right_labels = False
m1.xlines = False
m1.ylines = False

m1.xformatter = LONGITUDE_FORMATTER
m1.yformatter = LATITUDE_FORMATTER

m1.xlabel_style = {"size": 4, "color": "k"}
m1.ylabel_style = {"size": 4, "color": "k"}

# center
ax2.plot(CENTER[1], CENTER[0], "o", color="C2", mew=0.5, lw=0.5, ms=2)

if R < 1500:
    lat_radius, lon_radius = cutoff_radius(CENTER, R)
    ax2.plot(lon_radius, lat_radius, "-", color="C0", lw=0.5)
if R > 1500 and R < 5500:
    lat_radius, lon_radius = cutoff_radius(CENTER, R)
    ax2.plot(lon_radius, lat_radius, ".", color="C0", ms=0.5)

# raob
ax2.plot(obsxr, obsyr, "kx", mew=0.5, lw=0.5, ms=2)
# (4) Draw the obs patterns by different methods
MET = "dhmode"
filename = "../data/Network/ssp_%s_R%dN4.txt" % (MET, R)
data = pd.read_csv(filename, header=None, index_col=None, sep=r"\s+")
count = data.groupby(data.columns.tolist(), as_index=False).size()
total_selection = np.sum(count.iloc[:, 3])
obs = np.zeros((NLON, NLAT))
obs[:] = np.nan
for i in range(len(count)):
    obs[count.iloc[i, 1] - 1, count.iloc[i, 2] - 1] = (
        count.iloc[i, 3] / total_selection * 400.0
    )  # percentage

pcm = ax2.pcolormesh(X, Y, obs.T, lw=0, cmap=my_cmap, norm=norm)

###################
ax3 = fig.add_subplot(
    2, 2, 3, projection=ccrs.PlateCarree(central_longitude=180))
ax3.add_feature(cfeature.COASTLINE, linewidth=0.3)
ax3.set_extent([180, 300, -80, 5])  # Necessary, for the subsequent plot

m1 = ax3.gridlines(draw_labels=True)

m1.top_labels = False
m1.right_labels = False
m1.xlines = False
m1.ylines = False

m1.xformatter = LONGITUDE_FORMATTER
m1.yformatter = LATITUDE_FORMATTER

m1.xlabel_style = {"size": 4, "color": "k"}
m1.ylabel_style = {"size": 4, "color": "k"}

# center
ax3.plot(CENTER[1], CENTER[0], "o", color="C2", mew=0.5, lw=0.5, ms=2)

if R < 1500:
    lat_radius, lon_radius = cutoff_radius(CENTER, R)
    ax3.plot(lon_radius, lat_radius, "-", color="C0", lw=0.5)
if R > 1500 and R < 5500:
    lat_radius, lon_radius = cutoff_radius(CENTER, R)
    ax3.plot(lon_radius, lat_radius, ".", color="C0", ms=0.5)

# raob
ax3.plot(obsxr, obsyr, "kx", mew=0.5, lw=0.5, ms=2)
# (4) Draw the obs patterns by different methods
MET = "ahmode"
filename = "../data/Network/ssp_%s_R%dN4.txt" % (MET, R)
data = pd.read_csv(filename, header=None, index_col=None, sep=r"\s+")
count = data.groupby(data.columns.tolist(), as_index=False).size()
total_selection = np.sum(count.iloc[:, 3])
obs = np.zeros((NLON, NLAT))
obs[:] = np.nan
for i in range(len(count)):
    obs[count.iloc[i, 1] - 1, count.iloc[i, 2] - 1] = (
        count.iloc[i, 3] / total_selection * 400.0
    )  # percentage

# pcm = ax3.pcolormesh(X, Y, obs.T, cmap=my_cmap, vmax=20, vmin=0)
pcm = ax3.pcolormesh(X, Y, obs.T, lw=0, cmap=my_cmap, norm=norm)

###################
ax4 = fig.add_subplot(
    2, 2, 4, projection=ccrs.PlateCarree(central_longitude=180))
ax4.add_feature(cfeature.COASTLINE, linewidth=0.3)
ax4.set_extent([180, 300, -80, 5])  # Necessary, for the subsequent plot

m1 = ax4.gridlines(draw_labels=True)

m1.top_labels = False
m1.right_labels = False
m1.xlines = False
m1.ylines = False

m1.xformatter = LONGITUDE_FORMATTER
m1.yformatter = LATITUDE_FORMATTER

m1.xlabel_style = {"size": 4, "color": "k"}
m1.ylabel_style = {"size": 4, "color": "k"}

# center
ax4.plot(CENTER[1], CENTER[0], "o", color="C2", mew=0.5, lw=0.5, ms=2)

if R < 1500:
    lat_radius, lon_radius = cutoff_radius(CENTER, R)
    ax4.plot(lon_radius, lat_radius, "-", color="C0", lw=0.5)
if R > 1500 and R < 5500:
    lat_radius, lon_radius = cutoff_radius(CENTER, R)
    ax4.plot(lon_radius, lat_radius, ".", color="C0", ms=0.5)

# raob
ax4.plot(obsxr, obsyr, "kx", mew=0.5, lw=0.5, ms=2)
# (4) Draw the obs patterns by different methods
MET = "ehmode"
filename = "../data/Network/ssp_%s_R%dN4.txt" % (MET, R)
data = pd.read_csv(filename, header=None, index_col=None, sep=r"\s+")
count = data.groupby(data.columns.tolist(), as_index=False).size()
total_selection = np.sum(count.iloc[:, 3])
obs = np.zeros((NLON, NLAT))
obs[:] = np.nan
for i in range(len(count)):
    obs[count.iloc[i, 1] - 1, count.iloc[i, 2] - 1] = (
        count.iloc[i, 3] / total_selection * 400.0
    )  # percentage

pcm = ax4.pcolormesh(X, Y, obs.T, lw=0, cmap=my_cmap, norm=norm)

plt.savefig("../figure/fig3_obspattern.png")
plt.show()
