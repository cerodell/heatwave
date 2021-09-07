import context
import json
import salem

import xarray as xr
from pathlib import Path
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import string


import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
from datetime import datetime

from wrf import (
    to_np,
    getvar,
    smooth2d,
    get_cartopy,
    cartopy_xlim,
    cartopy_ylim,
    latlon_coords,
    interplevel,
)

from context import wrf_dir, data_dir
from cartopy.io.img_tiles import GoogleTiles
import warnings

warnings.simplefilter("ignore")


class ShadedReliefESRI(GoogleTiles):
    # shaded relief
    def _image_url(self, tile):
        x, y, z = tile
        url = (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg"
        ).format(z=z, y=y, x=x)
        return url


domain = "d02"
ds = xr.open_dataset(
    str(data_dir) + f"/heatwave/wrfout_{domain}_2021062506_2021070100-new.nc"
)

gog_dir = "/Volumes/GoogleDrive/Shared drives/WAN00CG-01/21071900"
ncfile = Dataset(str(gog_dir) + f"/wrfout_{domain}_2021-07-20_21:00:00")
p = getvar(ncfile, "pressure")
cart_proj = get_cartopy(p)


### Open color map json
with open(str(data_dir) + "/json/colormaps-hw.json") as f:
    cmaps = json.load(f)

var, index = "T", 18
vmin, vmax = cmaps[var]["vmin"], cmaps[var]["vmax"]
name, colors, sigma = (
    str(cmaps[var]["name"]),
    cmaps[var]["colors"],
    cmaps[var]["sigma"],
)
levels = cmaps[var]["levels"]
colors = colors[-24:]


ty1, ty2 = 0, -20
tx1, tx2 = 0, -1
levels = np.arange(0, 46.5, 0.5)
skip = 10
cmap = LinearSegmentedColormap.from_list("meteoblue", colors, N=len(levels))
lats, lons = ds.XLAT.values, ds.XLONG.values
# lats, lons = ds.XLAT.values[ty1:ty2, tx1:tx2], ds.XLONG.values[ty1:ty2, tx1:tx2]

# Download and add the states and coastlines
states = NaturalEarthFeature(
    category="cultural",
    scale="50m",
    facecolor="none",
    name="admin_1_states_provinces_shp",
)

sub_name = list(string.ascii_uppercase)

interp_level = ds.interp_level.values
for i in range(len(ds.Time.values)):
    ds1 = ds.isel(Time = i)
    ds1 = ds1.isel(interp_level = 1)
    temp = ds1.temp.values - 273.15
    slp = ds1.slp.values
    smooth_slp = smooth2d(slp, 50, cenweight=4).values
    # ind_high = np.unravel_index(np.argmax(smooth_slp, axis=None), smooth_slp.shape)
    # ind_low = np.unravel_index(np.argmin(smooth_slp, axis=None), smooth_slp.shape)
    ua = ds1.ua.values
    va = ds1.va.values

    Cnorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax + 1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=crs.PlateCarree())
    # ax = fig.add_subplot(1, 1, 1, projection=cart_proj)
    ax.add_feature(states, linewidth=0.5, edgecolor="black", zorder=9)
    ax.coastlines("50m", linewidth=0.8)

    valid = np.array(ds1.Time.dt.strftime("%Y-%m-%dT%H"))
    save_time = valid
    valid = datetime.strptime(str(valid), "%Y-%m-%dT%H").strftime("%H UTC %A %d %B, %Y")
    ax.set_title(f"Valid: {valid}")

    # ax.text(
    #     lons[ind_low[0], ind_low[1]],
    #     lats[ind_low[0], ind_low[1]],
    #     "L",
    #     transform=crs.Geodetic(),
    #     zorder=10,
    #     color="red",
    #     fontweight="bold",
    #     bbox=dict(facecolor="white", edgecolor="k", lw=0.5, boxstyle="round", pad=0.2),
    # )

    # ax.text(
    #     lons[ind_high[0], ind_high[1]],
    #     lats[ind_high[0], ind_high[1]],
    #     "H",
    #     transform=crs.Geodetic(),
    #     zorder=10,
    #     color="blue",
    #     fontweight="bold",
    #     bbox=dict(facecolor="white", edgecolor="k", lw=0.5, boxstyle="round", pad=0.2),
    # )

    CS = ax.contour(
        lons,
        lats,
        smooth_slp,
        levels=np.arange(960, 1032, 1),
        linewidths=0.6,
        colors="black",
        zorder=9,
    )
    cb = ax.clabel(CS, inline=1, fontsize=9, fmt="%1.0f", inline_spacing=-2.4, zorder=9)

    contourf = ax.contourf(
        lons,
        lats,
        temp,
        levels=levels,
        cmap=cmap,
        extend="both",
        zorder=6,
        # transform=crs.PlateCarree(),
    )

    widths = np.linspace(0, 10, ua[::skip, ::skip].size) * 20

    ax.quiver(
        lons[::skip, ::skip],
        lats[::skip, ::skip],
        ua[::skip, ::skip],
        va[::skip, ::skip],
        # length=3.8,
        zorder=8,
        linewidths=widths,
        transform=crs.PlateCarree(),
    )
    ax.set_xlim([-140, -110])
    ax.set_ylim([32.1, 62])

    clb = plt.colorbar(contourf,fraction=0.04, pad=0.04)
    clb.ax.get_yaxis().labelpad = 12
    clb.ax.set_ylabel("Temperature ($^\circ$C)", rotation=270, fontsize=11)

    # plt.figtext(0.1, 0.975, f"UBC WRF-NAM 12km Domain", fontsize=14)
    # plt.figtext(
    #     0.1,
    #     0.92,
    #     "Temperature at 925 hPa ($^\circ$C)  \nSea Level Pressure (hPa)",
    #     fontsize=11,
    # )
    # plt.show()
    print(i)
    plt.savefig(str(data_dir) + f'/images/heatwave/upper-{save_time}.png', dpi = 250)
