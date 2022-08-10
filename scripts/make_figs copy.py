import context
import json
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
)

from context import wrf_dir, data_dir, root_dir
from cartopy.io.img_tiles import GoogleTiles
import warnings

data_dir = "/Volumes/cer/heatwave/data"

warnings.simplefilter("ignore")

lats_study = np.arange(45, 52 + 0.1, 0.1)
lons_study = np.arange(-123.25, -119 + 0.1, 0.1)
## mesh for ploting...could do without
lons_study, lats_study = np.meshgrid(lons_study, lats_study)


# class ShadedReliefESRI(GoogleTiles):
#     # shaded relief
#     def _image_url(self, tile):
#         x, y, z = tile
#         url = (
#             "https://server.arcgisonline.com/ArcGIS/rest/services/"
#             "World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg"
#         ).format(z=z, y=y, x=x)
#         return url


domain = "d03"
ds = xr.open_dataset(str(data_dir) + f"/wrfout_{domain}_2021062506_2021070100.nc")

### Open color map json
with open(str(data_dir) + "/colormaps-hw.json") as f:
    cmaps = json.load(f)

var, index = "T", 18
vmin, vmax = cmaps[var]["vmin"], cmaps[var]["vmax"]
name, colors, sigma = (
    str(cmaps[var]["name"]),
    cmaps[var]["colors"],
    cmaps[var]["sigma"],
)
levels = cmaps[var]["levels"]

# cools = colors[-28:-24] # cools
# warms = colors[-12:-4]  # warms
# colors = cools + warms + [colors[-1]]

# warms = colors[-12:-4]  # warms
# colors =  warms + [colors[-1]]
colors = colors[-12:-1]


# 00ef7c
# d8f7a1
# %%


ds1 = ds.isel(Time=10)
intitalized_a = np.array(ds.isel(Time=7).Time.dt.strftime("%Y-%m-%dT%H"))
intitalized_a = datetime.strptime(str(intitalized_a), "%Y-%m-%dT%H").strftime(
    "%H UTC %A %d %B, %Y"
)

intitalized_cbde = np.array(ds.isel(Time=11).Time.dt.strftime("%Y-%m-%dT%H"))
intitalized_cbde = datetime.strptime(str(intitalized_cbde), "%Y-%m-%dT%H").strftime(
    "%H UTC %A %d %B, %Y"
)

intitalized_fghi = np.array(ds.isel(Time=15).Time.dt.strftime("%Y-%m-%dT%H"))
intitalized_fghi = datetime.strptime(str(intitalized_fghi), "%Y-%m-%dT%H").strftime(
    "%H UTC %A %d %B, %Y"
)

ty1, ty2 = 0, 402
tx1, tx2 = 44, -200
# ty1,ty2 = 0, -1
# tx1, tx2 = 0, -1
levels = np.arange(10, 46.5, 0.5)
skip = 15
cmap = LinearSegmentedColormap.from_list("meteoblue", colors, N=len(levels))
lats, lons = ds.XLAT.values[ty1:ty2, tx1:tx2], ds.XLONG.values[ty1:ty2, tx1:tx2]
# Download and add the states and coastlines
states = NaturalEarthFeature(
    category="cultural",
    scale="50m",
    facecolor="none",
    name="admin_1_states_provinces_shp",
)

sub_name = list(string.ascii_uppercase)

slp = ds1.slp
smooth_slp = smooth2d(slp, 3, cenweight=4)
smooth_slp = smooth_slp.values[ty1:ty2, tx1:tx2]
wsp = ds.wspd_wdir10


def plot_low(lon, lat):
    ax.text(
        lon,
        lat,
        "L",
        transform=crs.Geodetic(),
        zorder=10,
        color="red",
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="k", lw=0.5, boxstyle="round", pad=0.2),
    )
    return


def plot_high(lon, lat):
    ax.text(
        lon,
        lat,
        "H",
        transform=crs.Geodetic(),
        zorder=10,
        color="blue",
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="k", lw=0.5, boxstyle="round", pad=0.2),
    )
    return


# Create a figure
fig = plt.figure(figsize=(14, 10))
for i in range(1, 10):
    ax = fig.add_subplot(3, 3, i, projection=crs.PlateCarree())
    ax.text(
        0.022,
        0.96,
        sub_name[i - 1],
        ha="center",
        va="center",
        transform=ax.transAxes,
        zorder=10,
        color="k",
        bbox=dict(facecolor="white", edgecolor="k", lw=0.5),
    )
    ds1 = ds.isel(Time=10 + i)
    slp = ds1.slp
    smooth_slp = smooth2d(slp, 400, cenweight=4)
    smooth_slp = smooth_slp.values[ty1:ty2, tx1:tx2]
    wsp = ds1.wspd_wdir10.values[ty1:ty2, tx1:tx2]

    if i == 1:
        plot_low(-123.825, 44.23)
        plot_high(-115.065, 50.9)
    elif i == 2:
        plot_low(-124.077, 44.678)
        plot_high(-115.437, 50.68)
    elif i == 3:
        plot_low(-126.2, 43.62)
        plot_high(-116.656, 51.233)
    elif i == 4:
        plot_low(-127.178, 43.58)
        plot_high(-116.968, 51.524)
    elif i == 5:
        plot_low(-128.003, 43.489)
        plot_high(-117.22, 52.189)
    elif i == 6:
        plot_low(-128.436, 43.58)
        plot_low(-126.08, 48.29)
        plot_high(-116.595, 51.796)
    elif i == 7:
        plot_low(-128.356, 43.599)
        plot_low(-127.48, 48.009)
        plot_high(-116.33, 51.392)
    elif i == 8:
        plot_low(-128.034, 43.901)
        plot_low(-128.305, 47.465)
        plot_high(-116.495, 51.463)
    else:
        plot_low(-128.134, 45.079)
        plot_high(-115.931, 51.39)

    t2 = ds1.T2.values - 273.15
    t2 = t2[ty1:ty2, tx1:tx2]
    u10, v10 = (
        ds1.uvmet10.values[0, ty1:ty2, tx1:tx2] * (60 * 60 / 1000),
        ds1.uvmet10.values[1, ty1:ty2, tx1:tx2] * (60 * 60 / 1000),
    )
    # (1.943844)
    valid = np.array(ds1.Time.dt.strftime("%Y-%m-%dT%H"))
    valid = datetime.strptime(str(valid), "%Y-%m-%dT%H").strftime("%H UTC %A %d %B, %Y")
    ax.set_title(f"Valid: {valid}")

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)

    ax.add_feature(states, linewidth=0.5, edgecolor="black", zorder=9)
    ax.coastlines("50m", linewidth=0.8)

    CS = ax.contour(
        lons,
        lats,
        smooth_slp,
        levels=np.arange(960, 1032, 2),
        linewidths=0.4,
        colors="black",
        zorder=9,
    )
    cb = ax.clabel(CS, inline=1, fontsize=7, fmt="%1.0f", inline_spacing=-2.4, zorder=9)

    contourf = ax.contourf(
        lons,
        lats,
        t2,
        levels=levels,
        cmap=cmap,
        extend="both",
        zorder=4,
        alpha=0.9,
    )

    # ax.barbs(
    #     lons[::skip, ::skip],
    #     lats[::skip, ::skip],
    #     u10[::skip, ::skip],
    #     v10[::skip, ::skip],
    #     length=3,
    #     zorder=8,
    #     lw=0.4,
    # )

    widths = np.linspace(0, 10, u10[::skip, ::skip].size)
    ax.quiver(
        lons[::skip, ::skip],
        lats[::skip, ::skip],
        u10[::skip, ::skip],
        v10[::skip, ::skip],
        zorder=8,
        lw=widths,
        # width = 0.5,
        headwidth=9,
        headlength=6,
    )

    # # Set the map bounds
    ax.set_xlim([-129, -114.1])
    ax.set_ylim([43.2, 53.6])
    # Add the gridlines
    color = "black"
    ax.plot(lons_study[0], lats_study[0], color=color, linewidth=2, zorder=10, alpha=1)
    ax.plot(
        lons_study[-1].T, lats_study[-1].T, color=color, linewidth=2, zorder=10, alpha=1
    )
    ax.plot(
        lons_study[:, 0], lats_study[:, 0], color=color, linewidth=2, zorder=10, alpha=1
    )
    ax.plot(
        lons_study[:, -1].T,
        lats_study[:, -1].T,
        color=color,
        linewidth=2,
        zorder=10,
        alpha=1,
        label="Study Area",
    )

    if i == 1:
        print(i)
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.5,
            color="gray",
            alpha=0.5,
            linestyle="dotted",
            zorder=5,
        )
        gl.top_labels = False
        gl.bottom_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(list(np.arange(-180, 180, 4)))

    elif i == 4:
        print(i)
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.5,
            color="gray",
            alpha=0.5,
            linestyle="dotted",
            zorder=5,
        )
        gl.top_labels = False
        gl.bottom_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(list(np.arange(-180, 180, 4)))

    elif i == 7:
        print(i)
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.5,
            color="gray",
            alpha=0.5,
            linestyle="dotted",
            zorder=5,
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = True
        gl.bottom_labels = True
        gl.xlocator = mticker.FixedLocator(list(np.arange(-180, 180, 4)))

    elif i == 8:
        print(i)
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.5,
            color="gray",
            alpha=0.5,
            linestyle="dotted",
            zorder=5,
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = False
        gl.bottom_labels = True
        gl.xlocator = mticker.FixedLocator(list(np.arange(-180, 180, 4)))

    elif i == 9:
        print(i)
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.5,
            color="gray",
            alpha=0.5,
            linestyle="dotted",
            zorder=5,
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = False
        gl.bottom_labels = True
        gl.xlocator = mticker.FixedLocator(list(np.arange(-180, 180, 4)))

    else:
        gl = ax.gridlines(
            draw_labels=False,
            linewidth=0.5,
            color="gray",
            alpha=0.5,
            linestyle="dotted",
            zorder=5,
        )
    plt.subplots_adjust(wspace=0.0)


# plt.title("Sea Level Pressure (hPa)")
# fig.tight_layout(rect=[0, 0.03, 1, 0.9])
# fig.subplots_adjust(right=0.99, wspace=-0.45)
# cbaxes = fig.add_axes([0.88, 0.04, 0.03, 0.8])
# clb = fig.colorbar(contourf, cax=cbaxes, pad=0.2)

## vectors
fig.tight_layout(rect=[0, 0, 0.9, 0.9])  # (left, bottom, right, top)
fig.subplots_adjust(wspace=-0.2)
cbaxes = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # (left, bottom, right, top)
clb = fig.colorbar(contourf, cax=cbaxes, pad=0.01)
clb.ax.get_yaxis().labelpad = 14
clb.ax.set_ylabel("Temperature at 2m ($^\circ$C)", rotation=270, fontsize=11)

# ## barbs
# fig.tight_layout(rect=[0, 0.03, 0.9, 0.9]) #(left, bottom, right, top)
# fig.subplots_adjust(right=0.99, wspace=-0.4)
# cbaxes = fig.add_axes([0.9, 0.15, 0.02, 0.7]) #(left, bottom, right, top)
# clb = fig.colorbar(contourf, cax=cbaxes, pad=0.01)
# clb.ax.get_yaxis().labelpad = 14
# clb.ax.set_ylabel("Temperature at 2m ($^\circ$C)", rotation=270, fontsize=11)

plt.figtext(0.65, 0.97, f"Init [A]:              {intitalized_a}", fontsize=11)
plt.figtext(0.65, 0.95, f"Init [B, C, D, E]: {intitalized_cbde}", fontsize=11)
plt.figtext(0.65, 0.93, f"Init [F, G, H, I]:  {intitalized_fghi}", fontsize=11)

plt.figtext(0.1, 0.975, f"UBC WRF-NAM 4km Domain", fontsize=14)
plt.figtext(
    0.1,
    0.92,
    # "Temperature at 2m ($^\circ$C)  \nSea Level Pressure (hPa) \n10m Wind (full barb = 10$km\,h^{-1}$)",
    "Temperature at 2m ($^\circ$C)  \nSea Level Pressure (hPa) \n10m Wind Vectors",
    fontsize=11,
)

plt.savefig(
    str(root_dir) + f"/img/slp-final-multi-vectors.png",
    dpi=250,
    bbox_inches="tight",
)

plt.show()
# %%
