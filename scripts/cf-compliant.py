import context
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from netCDF4 import Dataset

from context import wrf_dir, root_dir, data_dir


# data_dir = '/Volumes/cer/heatwave/data'
domain = "d02"
ds = xr.open_dataset(str(data_dir) + f"/wrfout_{domain}_2021062506_2021070100.nc")
# ds = ds.isel(Time = slice(0,3))
# get 50 kPa geo heights
# pressure = ds.pressure.values
# h_925 = ds.height_interp_925.values
# h_850 = ds.height_interp_850.values
# h_700 = ds.height_interp_700.values
h_500 = ds.height_interp_500.values
# h_250 = ds.height_interp_250.values

# geo_height = np.stack([h_250,h_500,h_700,h_850,h_925])
# geo_height = np.transpose(geo_height, axes=(1,0,2,3))

## get x,y x, time coordinate dimensions as an array
x = ds.west_east.values
y = ds.south_north.values
z = ds.bottom_top.values
# z = np.array([0,1,2,3,4])

date_range = np.array(ds.Time, dtype="datetime64[h]")
# date_range = np.array(date_range, dtype= 'S13')


## define LAT and LONG 2D arrays
XX, YY = ds.XLONG.values, ds.XLAT.values
Z = ds.height.isel(Time=0, south_north=0, west_east=0).values
# Z = ds.height.isel(Time = 0).values
# Z = np.array([925,850,700,500,250][::-1])

p_500 = np.full_like(h_500, h_500 - 530)


## create new dataset a set up to be CF Compliant
ds_cf = xr.Dataset(
    data_vars=dict(
        # pressure=(["time", "z", "y", "x"], pressure.astype("float32")),
        # geo_height=(["time", "z", "y", "x"], geo_height.astype("float32")),
        h_500=(["time", "y", "x"], h_500.astype("float32")),
        p_500=(["time", "y", "x"], p_500.astype("float32")),
        x=(["x"], x.astype("int32")),
        y=(["y"], y.astype("int32")),
        z=(["z"], z.astype("int32")),
        Times=(["time"], date_range.astype("S13")),
        # HEIGHT=(["z", "y", "x"], Z.astype("float32")),
    ),
    coords=dict(
        LONG=(["y", "x"], XX.astype("float32")),
        LAT=(["y", "x"], YY.astype("float32")),
        # HEIGHT=(["z", "y", "x"], Z.astype("float32")),
        HEIGHT=(["z"], Z.astype("float32")),
        time=date_range,
    ),
    attrs=dict(description="BC 2021 Heatwave"),
)

## add axis attributes from cf compliance
ds_cf["time"].attrs["axis"] = "Time"
ds_cf["x"].attrs["axis"] = "X"
ds_cf["y"].attrs["axis"] = "Y"
ds_cf["z"].attrs["axis"] = "Z"


## add units attributes from cf compliance
# ds_cf["pressure"].attrs["units"] = "hPa"
# ds_cf["geo_height"].attrs["units"] = "geo"
ds_cf["h_500"].attrs["units"] = "dm"
ds_cf["p_500"].attrs["units"] = "kg"
ds_cf["LONG"].attrs["units"] = "degree_east"
ds_cf["LAT"].attrs["units"] = "degree_north"
ds_cf["HEIGHT"].attrs["units"] = "dm"


def compressor(ds):
    """
    Compresses datasets
    """
    ## load ds to memory
    ds = ds.load()
    ## use zlib to compress to level 9
    comp = dict(zlib=True, complevel=9)
    ## create endcoding for each variable in dataset
    encoding = {var: comp for var in ds.data_vars}

    return ds, encoding


ds_cf, encoding = compressor(ds_cf)

## write the new dataset
ds_cf.to_netcdf(
    str(data_dir) + "/pressure.nc",
    encoding=encoding,
    mode="w",
)


# from mpl_toolkits import mplot3d
# import numpy as np
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = plt.axes(projection='3d')

# ax.plot_surface(XX, YY, h_500[0,:,:], rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
# ax.set_title('surface')
# plt.show()
