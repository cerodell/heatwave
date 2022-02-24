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

warnings.simplefilter("ignore")
data_dir = "/Volumes/cer/heatwave/data"


lats_study = np.arange(45, 52 + 0.1, 0.1)
lons_study = np.arange(-123.25, -119 + 0.1, 0.1)
## mesh for ploting...could do without
lons_study, lats_study = np.meshgrid(lons_study, lats_study)


class ShadedReliefESRI(GoogleTiles):
    # shaded relief
    def _image_url(self, tile):
        x, y, z = tile
        url = (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg"
        ).format(z=z, y=y, x=x)
        return url


domain = "d03"
ds = xr.open_dataset(str(data_dir) + f"/wrfout_{domain}_2021062506_2021070100.nc")


gog_dir = "/Users/rodell/fwf/data/wrf"
ncfile = Dataset(str(gog_dir) + f"/wrfout_{domain}_2021-01-14_00:00:00")
p = getvar(ncfile, "pressure")


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

colors = colors[-24:]


# %%


ty1, ty2 = 0, 400
tx1, tx2 = 0, -180
# ty1,ty2 = 0, 20
# tx1, tx2 = 0, 20
levels = np.arange(0, 46.5, 0.5)
skip = 20
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


cart_proj = get_cartopy(p)
# fig = plt.figure(figsize=(7, 9))


# # ax = fig.add_subplot(2, 1, 1, projection=cart_proj)
# ax = fig.add_subplot(2, 1, 1, projection=crs.PlateCarree())
# ds1 = ds.isel(Time=11)
# slp = ds1.slp
# smooth_slp = smooth2d(slp, 400, cenweight=4)
# smooth_slp = smooth_slp.values[ty1:ty2, tx1:tx2]


# t2 = ds1.T2.values - 273.15
# t2 = t2[ty1:ty2, tx1:tx2]
# u10, v10 = (
#     ds1.uvmet10.values[0, ty1:ty2, tx1:tx2] * (60 * 60 / 1000),
#     ds1.uvmet10.values[1, ty1:ty2, tx1:tx2] * (60 * 60 / 1000),
# )

# valid = np.array(ds1.Time.dt.strftime("%Y-%m-%dT%H"))
# save_time = valid
# valid = datetime.strptime(str(valid), "%Y-%m-%dT%H").strftime("%H UTC %A %d %B, %Y")
# ax.set_title(f"Valid: {valid}")

# divider = make_axes_locatable(ax)
# ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)

# ax.add_feature(states, linewidth=0.5, edgecolor="black", zorder=9)
# ax.coastlines("50m", linewidth=0.8)
# ax.text(
#     0.022,
#     0.96,
#     "A",
#     fontsize=18,
#     ha="center",
#     va="center",
#     transform=ax.transAxes,
#     zorder=10,
#     color="k",
#     bbox=dict(facecolor="white", edgecolor="k", lw=0.5),
# )
# ax.text(
#     -124.1,  ## 28th
#     43.94,
#     "L",
#     transform=crs.Geodetic(),
#     zorder=10,
#     color="red",
#     fontweight="bold",
#     fontsize=20,
#     bbox=dict(facecolor="white", edgecolor="k", lw=0.5, boxstyle="round", pad=0.2),
# )

# ax.text(
#     -115.2,
#     50.6,
#     "H",
#     transform=crs.Geodetic(),
#     zorder=10,
#     color="blue",
#     fontweight="bold",
#     fontsize=20,
#     bbox=dict(facecolor="white", edgecolor="k", lw=0.5, boxstyle="round", pad=0.2),
# )

# CS = ax.contour(
#     lons,
#     lats,
#     smooth_slp,
#     levels=np.arange(960, 1032, 2),
#     linewidths=0.6,
#     colors="black",
#     zorder=9,
#     transform=crs.PlateCarree(),
# )
# cb = ax.clabel(CS, inline=1, fontsize=9, fmt="%1.0f", inline_spacing=-2.4, zorder=9)


# contourf = ax.contourf(
#     lons,
#     lats,
#     t2,
#     levels=levels,
#     cmap=cmap,
#     extend="both",
#     zorder=4,
#     alpha=0.9,
#     transform=crs.PlateCarree(),
# )

# widths = np.linspace(0, 1.4, u10[::skip, ::skip].size)
# ax.quiver(
#     lons[::skip, ::skip],
#     lats[::skip, ::skip],
#     u10[::skip, ::skip],
#     v10[::skip, ::skip],
#     zorder=8,
#     lw=widths,
#     transform=crs.PlateCarree(),
# )
# ax.set_xlim([-130, -114])
# ax.set_ylim([43.2, 55.5])
# gl = ax.gridlines(
#     # crs = crs.PlateCarree(),
#     draw_labels=True,
#     linewidth=0.5,
#     color="gray",
#     alpha=0.5,
#     linestyle="dotted",
#     zorder=5,
# )
# gl.xlocator = mticker.FixedLocator(list(np.arange(-180, 180, 3)))
# gl.top_labels = False
# gl.right_labels = False
# gl.left_labels = True
# gl.bottom_labels = False

# ## make study area domain


# color = "black"
# ax.plot(lons_study[0], lats_study[0], color=color, linewidth=2, zorder=10, alpha=1)
# ax.plot(lons_study[-1].T, lats_study[-1].T, color=color, linewidth=2, zorder=10, alpha=1)
# ax.plot(lons_study[:, 0], lats_study[:, 0], color=color, linewidth=2, zorder=10, alpha=1)
# ax.plot(
#     lons_study[:, -1].T,
#     lats_study[:, -1].T,
#     color=color,
#     linewidth=2,
#     zorder=10,
#     alpha=1,
#     label="Study Area",

# )


# ax = fig.add_subplot(2, 1, 2, projection=crs.PlateCarree())
# # ax = fig.add_subplot(2, 1, 2, projection=cart_proj)
# ds1 = ds.isel(Time=15)
# slp = ds1.slp
# smooth_slp = smooth2d(slp, 400, cenweight=4)
# smooth_slp = smooth_slp.values[ty1:ty2, tx1:tx2]


# t2 = ds1.T2.values - 273.15
# t2 = t2[ty1:ty2, tx1:tx2]
# u10, v10 = (
#     ds1.uvmet10.values[0, ty1:ty2, tx1:tx2] * (60 * 60 / 1000),
#     ds1.uvmet10.values[1, ty1:ty2, tx1:tx2] * (60 * 60 / 1000),
# )

# valid = np.array(ds1.Time.dt.strftime("%Y-%m-%dT%H"))
# save_time = valid
# valid = datetime.strptime(str(valid), "%Y-%m-%dT%H").strftime("%H UTC %A %d %B, %Y")
# ax.set_title(f"Valid: {valid}")

# divider = make_axes_locatable(ax)
# ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)

# ax.add_feature(states, linewidth=0.5, edgecolor="black", zorder=9)
# ax.coastlines("50m", linewidth=0.8)
# ax.text(
#     0.022,
#     0.96,
#     "B",
#     fontsize=18,
#     ha="center",
#     va="center",
#     transform=ax.transAxes,
#     zorder=10,
#     color="k",
#     bbox=dict(facecolor="white", edgecolor="k", lw=0.5),
# )
# ax.text(
#     -128.0,  ## 29th
#     43.9,
#     "L",
#     transform=crs.Geodetic(),
#     zorder=10,
#     color="red",
#     fontweight="bold",
#     fontsize=20,
#     bbox=dict(facecolor="white", edgecolor="k", lw=0.5, boxstyle="round", pad=0.2),
# )

# ax.text(
#     -117.3,  ## 29th
#     51.4,
#     "H",
#     transform=crs.Geodetic(),
#     zorder=10,
#     color="blue",
#     fontweight="bold",
#     fontsize=20,
#     bbox=dict(facecolor="white", edgecolor="k", lw=0.5, boxstyle="round", pad=0.2),
# )

# CS = ax.contour(
#     lons,
#     lats,
#     smooth_slp,
#     levels=np.arange(960, 1032, 2),
#     linewidths=0.6,
#     colors="black",
#     zorder=9,
#     transform=crs.PlateCarree(),
# )
# cb = ax.clabel(CS, inline=1, fontsize=9, fmt="%1.0f", inline_spacing=-2.4, zorder=9)

# contourf = ax.contourf(
#     lons,
#     lats,
#     t2,
#     levels=levels,
#     cmap=cmap,
#     extend="both",
#     zorder=4,
#     alpha=0.9,
#     transform=crs.PlateCarree(),
# )

# widths = np.linspace(0, 1.4, u10[::skip, ::skip].size)
# ax.quiver(
#     lons[::skip, ::skip],
#     lats[::skip, ::skip],
#     u10[::skip, ::skip],
#     v10[::skip, ::skip],
#     zorder=8,
#     lw=widths,
#     transform=crs.PlateCarree(),
# )
# ax.set_xlim([-130, -114])
# ax.set_ylim([43.2, 55.5])
# gl = ax.gridlines(
#     crs=crs.PlateCarree(),
#     draw_labels=True,
#     linewidth=0.5,
#     color="gray",
#     alpha=0.5,
#     linestyle="dotted",
#     zorder=5,
# )
# gl.top_labels = False
# gl.right_labels = False
# gl.left_labels = True
# gl.bottom_labels = True

# fig.tight_layout(rect=[0, 0.03, 1, 0.9])
# fig.subplots_adjust(right=0.99, wspace=-0.45)
# cbaxes = fig.add_axes([0.88, 0.1, 0.04, 0.8])
# clb = fig.colorbar(contourf, cax=cbaxes, pad=0.02)
# # clb.ax.set_title('Temp at 2m ($^\circ$C)', ha='right', fontsize=8, rotation=-90)
# clb.ax.get_yaxis().labelpad = 14
# clb.ax.set_ylabel("Temperature at 2m ($^\circ$C)", rotation=270, fontsize=11)


# ## make study area domain


# color = "black"
# ax.plot(lons_study[0], lats_study[0], color=color, linewidth=2, zorder=10, alpha=1)
# ax.plot(lons_study[-1].T, lats_study[-1].T, color=color, linewidth=2, zorder=10, alpha=1)
# ax.plot(lons_study[:, 0], lats_study[:, 0], color=color, linewidth=2, zorder=10, alpha=1)
# ax.plot(
#     lons_study[:, -1].T,
#     lats_study[:, -1].T,
#     color=color,
#     linewidth=2,
#     zorder=10,
#     alpha=1,
#     label="Study Area",

# )


# # clb = plt.colorbar(contourf,fraction=0.04, pad=0.004)
# # clb.ax.get_yaxis().labelpad = 12
# # clb.ax.set_ylabel("Temperature ($^\circ$C)", rotation=270, fontsize=11)
# # fig.tight_layout(rect=[0, 0.03, 1, 0.9])
# plt.figtext(0.1, 0.975, f"UBC WRF-NAM 4km Domain", fontsize=14)
# plt.figtext(
#     0.1,
#     0.91,
#     "Temperature at 2m ($^\circ$C)  \nSea Level Pressure (hPa) \n10m Wind Vectors",
#     fontsize=11,
# )
# # plt.show()
# plt.savefig(str(root_dir) + f"/img/slp-final-vertical.png", dpi=250, bbox_inches="tight")
# plt.close()


# %%

fig = plt.figure(figsize=(12, 6))
# ax = fig.add_subplot(2, 1, 1, projection=cart_proj)
ax = fig.add_subplot(1, 2, 1, projection=crs.PlateCarree())
ds1 = ds.isel(Time=11)
slp = ds1.slp
smooth_slp = smooth2d(slp, 400, cenweight=4)
smooth_slp = smooth_slp.values[ty1:ty2, tx1:tx2]


t2 = ds1.T2.values - 273.15
t2 = t2[ty1:ty2, tx1:tx2]
u10, v10 = (
    ds1.uvmet10.values[0, ty1:ty2, tx1:tx2] * (60 * 60 / 1000),
    ds1.uvmet10.values[1, ty1:ty2, tx1:tx2] * (60 * 60 / 1000),
)

valid = np.array(ds1.Time.dt.strftime("%Y-%m-%dT%H"))
save_time = valid
valid = datetime.strptime(str(valid), "%Y-%m-%dT%H").strftime("%H UTC %A %d %B, %Y")
ax.set_title(f"Valid: {valid}")

divider = make_axes_locatable(ax)
ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)

ax.add_feature(states, linewidth=0.5, edgecolor="black", zorder=9)
ax.coastlines("50m", linewidth=0.8)
ax.text(
    0.022,
    0.96,
    "A",
    fontsize=18,
    ha="center",
    va="center",
    transform=ax.transAxes,
    zorder=10,
    color="k",
    bbox=dict(facecolor="white", edgecolor="k", lw=0.5),
)
ax.text(
    -124.1,  ## 28th
    43.94,
    "L",
    transform=crs.Geodetic(),
    zorder=10,
    color="red",
    fontweight="bold",
    fontsize=20,
    bbox=dict(facecolor="white", edgecolor="k", lw=0.5, boxstyle="round", pad=0.2),
)

ax.text(
    -115.2,
    50.6,
    "H",
    transform=crs.Geodetic(),
    zorder=10,
    color="blue",
    fontweight="bold",
    fontsize=20,
    bbox=dict(facecolor="white", edgecolor="k", lw=0.5, boxstyle="round", pad=0.2),
)

CS = ax.contour(
    lons,
    lats,
    smooth_slp,
    levels=np.arange(960, 1032, 2),
    linewidths=0.6,
    colors="black",
    zorder=9,
    transform=crs.PlateCarree(),
)
cb = ax.clabel(CS, inline=1, fontsize=9, fmt="%1.0f", inline_spacing=-2.4, zorder=9)

contourf = ax.contourf(
    lons,
    lats,
    t2,
    levels=levels,
    cmap=cmap,
    extend="both",
    zorder=4,
    alpha=0.9,
    transform=crs.PlateCarree(),
)

widths = np.linspace(0, 1.4, u10[::skip, ::skip].size)
ax.quiver(
    lons[::skip, ::skip],
    lats[::skip, ::skip],
    u10[::skip, ::skip],
    v10[::skip, ::skip],
    zorder=8,
    lw=widths,
    transform=crs.PlateCarree(),
)
ax.set_xlim([-130, -114])
ax.set_ylim([43.2, 55.5])
gl = ax.gridlines(
    # crs = crs.PlateCarree(),
    draw_labels=True,
    linewidth=0.5,
    color="gray",
    alpha=0.5,
    linestyle="dotted",
    zorder=5,
)
gl.xlocator = mticker.FixedLocator(list(np.arange(-180, 180, 4)))
gl.top_labels = False
gl.right_labels = False
gl.bottom_labels = True
gl.left_labels = True


## make study area domain


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


ax = fig.add_subplot(1, 2, 2, projection=crs.PlateCarree())
# ax = fig.add_subplot(2, 1, 2, projection=cart_proj)
ds1 = ds.isel(Time=15)
slp = ds1.slp
smooth_slp = smooth2d(slp, 400, cenweight=4)
smooth_slp = smooth_slp.values[ty1:ty2, tx1:tx2]


t2 = ds1.T2.values - 273.15
t2 = t2[ty1:ty2, tx1:tx2]
u10, v10 = (
    ds1.uvmet10.values[0, ty1:ty2, tx1:tx2] * (60 * 60 / 1000),
    ds1.uvmet10.values[1, ty1:ty2, tx1:tx2] * (60 * 60 / 1000),
)

valid = np.array(ds1.Time.dt.strftime("%Y-%m-%dT%H"))
save_time = valid
valid = datetime.strptime(str(valid), "%Y-%m-%dT%H").strftime("%H UTC %A %d %B, %Y")
ax.set_title(f"Valid: {valid}")

divider = make_axes_locatable(ax)
ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)

ax.add_feature(states, linewidth=0.5, edgecolor="black", zorder=9)
ax.coastlines("50m", linewidth=0.8)
ax.text(
    0.022,
    0.96,
    "B",
    fontsize=18,
    ha="center",
    va="center",
    transform=ax.transAxes,
    zorder=10,
    color="k",
    bbox=dict(facecolor="white", edgecolor="k", lw=0.5),
)
ax.text(
    -128.0,  ## 29th
    43.9,
    "L",
    transform=crs.Geodetic(),
    zorder=10,
    color="red",
    fontweight="bold",
    fontsize=20,
    bbox=dict(facecolor="white", edgecolor="k", lw=0.5, boxstyle="round", pad=0.2),
)

ax.text(
    -117.3,  ## 29th
    51.4,
    "H",
    transform=crs.Geodetic(),
    zorder=10,
    color="blue",
    fontweight="bold",
    fontsize=20,
    bbox=dict(facecolor="white", edgecolor="k", lw=0.5, boxstyle="round", pad=0.2),
)

CS = ax.contour(
    lons,
    lats,
    smooth_slp,
    levels=np.arange(960, 1032, 2),
    linewidths=0.6,
    colors="black",
    zorder=9,
    transform=crs.PlateCarree(),
)
cb = ax.clabel(CS, inline=1, fontsize=9, fmt="%1.0f", inline_spacing=-2.4, zorder=9)

contourf = ax.contourf(
    lons,
    lats,
    t2,
    levels=levels,
    cmap=cmap,
    extend="both",
    zorder=4,
    alpha=0.9,
    transform=crs.PlateCarree(),
)

widths = np.linspace(0, 1.4, u10[::skip, ::skip].size)
ax.quiver(
    lons[::skip, ::skip],
    lats[::skip, ::skip],
    u10[::skip, ::skip],
    v10[::skip, ::skip],
    zorder=8,
    lw=widths,
    transform=crs.PlateCarree(),
)
ax.set_xlim([-130, -114])
ax.set_ylim([43.2, 55.5])
gl = ax.gridlines(
    crs=crs.PlateCarree(),
    draw_labels=True,
    linewidth=0.5,
    color="gray",
    alpha=0.5,
    linestyle="dotted",
    zorder=5,
)
gl.xlocator = mticker.FixedLocator(list(np.arange(-180, 180, 4)))
gl.top_labels = False
gl.right_labels = False
gl.left_labels = False
gl.bottom_labels = True

# fig.tight_layout(rect=[0, 0.03, 0.9, 0.9])
fig.tight_layout()
fig.subplots_adjust(right=0.9, wspace=0)
cbaxes = fig.add_axes([0.9, 0.15, 0.02, 0.7])
clb = fig.colorbar(contourf, cax=cbaxes, pad=0.01)
# clb.ax.set_title('Temp at 2m ($^\circ$C)', ha='right', fontsize=8, rotation=-90)
clb.ax.get_yaxis().labelpad = 14
clb.ax.set_ylabel("Temperature at 2m ($^\circ$C)", rotation=270, fontsize=11)


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


plt.figtext(0.1, 0.97, f"UBC WRF-NAM 4km Domain", fontsize=14)
plt.figtext(
    0.1,
    0.88,
    "Temperature at 2m ($^\circ$C)  \nSea Level Pressure (hPa) \n10m Wind Vectors",
    fontsize=11,
)
# plt.show()
plt.savefig(
    str(root_dir) + f"/img/slp-final-horizontal.png", dpi=250, bbox_inches="tight"
)
# plt.close()
# %%
