import context
import json
import xarray as xr
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cartopy.io.img_tiles as cimgt
import cartopy.crs as ccrs
from datetime import datetime, timedelta

from context import data_dir
import requests
import fiona
import pandas as pd
import geopandas as gp
from utils.utils import openkml
from utils.cfcompliant import cfcompliant
import cartopy.feature as cfeature

fwfilein = '/Volumes/Scratch/FWF-WAN00CG/d03'
smfilein = str(data_dir)

ds_wrf = xr.open_dataset('/Users/crodell/fwf/data/wrf/wrfout_d03_2021-01-14_00:00:00')

ds_fw0 = xr.open_dataset(fwfilein + '/202106/fwf-daily-d03-2021062006.nc').isel(time = 0)
ds_fw1 = xr.open_dataset(fwfilein + '/202107/fwf-daily-d03-2021070306.nc').isel(time = 0)

ds_sm0 = cfcompliant(smfilein + '/dispersion-2021062008.nc')
ds_sm0 = ds_sm0.isel(time = 12, z = 0)
ds_sm1 = cfcompliant(smfilein + '/dispersion-2021070308.nc')
ds_sm1 = ds_sm1.isel(time = 12, z = 0)


kml0 = openkml(str(data_dir) + '/fire_locations-2021062008.kml')
kml1 = openkml(str(data_dir) + '/fire_locations-2021070308.kml')



stamen_terrain = cimgt.Stamen('terrain-background')

# projections that involved
st_proj = stamen_terrain.crs  #projection used by Stamen images
ll_proj = ccrs.PlateCarree()  #CRS for raw long/lat

## bring in state/prov boundaries
states_provinces = cfeature.NaturalEarthFeature(
    category="cultural",
    name="admin_1_states_provinces_lines",
    scale="50m",
    facecolor="none",
)


# create fig and axes using intended projection
fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(2, 1, 2, projection=st_proj)

ty1, ty2 = 140, -80
tx1, tx2 = 200, -500
lons_sm = ds_sm0.LONG.values[ty1:ty2, tx1:tx2]
lats_sm = ds_sm0.LAT.values[ty1:ty2, tx1:tx2]
pm25 = ds_sm1.pm25.values[ty1:ty2, tx1:tx2]
pm25 = np.ma.masked_array(pm25, pm25 < 1)

CS = ax.pcolormesh(
    lons_sm,
    lats_sm,
    pm25,
    vmin = 10,
    vmax = 100,
    zorder=9,
    transform=ll_proj,
)
ax.add_image(stamen_terrain, 8) # this requests image, and plot
ax.set_extent([np.min(lons_sm), np.max(lons_sm), np.min(lats_sm), np.max(lats_sm)], crs=ll_proj)


ax = fig.add_subplot(2, 1, 1, projection=ccrs.Mercator())

ty1, ty2 = 0, -1
tx1, tx2 = 0, -1
lons_fw = ds_fw0.XLONG.values[ty1:ty2, tx1:tx2]
lats_fw = ds_fw0.XLAT.values[ty1:ty2, tx1:tx2]
U = ds_fw0.U.values[ty1:ty2, tx1:tx2]
# U  = np.ma.masked_array(U, ds_wrf.LANDMASK.values[0,ty1:ty2, tx1:tx2] < 1)

CS = ax.pcolormesh(
    lons_fw,
    lats_fw,
    U,
    vmin = 0,
    vmax = 100,
    zorder=1,
    transform=ll_proj,
)
# ax.add_image(stamen_terrain, 8) # this requests image, and plot
# ax.set_xlim(np.min(lons_sm), np.max(lons_sm))
# ax.set_xlim([-140, -102])
ax.set_ylim([40, 80])

ax.set_extent([np.min(lons_sm), np.max(lons_sm), np.min(lats_sm), np.max(lats_sm)], crs=ll_proj)
## add map features
# ax.gridlines()
# ax.add_feature(cfeature.LAND, zorder=1)
ax.add_feature(cfeature.LAKES, zorder=10)
ax.add_feature(cfeature.OCEAN, zorder=10)
ax.add_feature(cfeature.BORDERS, zorder=10)
ax.add_feature(cfeature.COASTLINE, zorder=10)
# ax.set_xlabel("Longitude", fontsize=18)
# ax.set_ylabel("Latitude", fontsize=18)
# cb = ax.clabel(CS, inline=1, fontsize=9, fmt="%1.0f", inline_spacing=-2.4, zorder=9)

