import context
import json
import xarray as xr
from pathlib import Path
import numpy as np

from datetime import datetime, timedelta


from context import data_dir
import geopandas as gp
import pandas as pd
import fiona


def openkml(kml):
    try: 
        df = pd.read_csv(kml[:-3] + 'csv')
    except:
        gp.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
        # empty GeoDataFrame
        df = gp.GeoDataFrame()
        # iterate over layers
        for layer in fiona.listlayers(kml):
            if layer == 'Fire Information':
                pass
            else:
                s = gp.read_file(kml, driver='KML', layer=layer)
                df = df.append(s, ignore_index=True)


        df.head()
        df['lon'] = df.geometry.x
        df['lat'] = df.geometry.y
        df = df.drop(columns=['Name', 'Description', 'geometry'])
        df.to_csv(kml[:-3] + 'csv', index = False)

    return df


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


def cfcompliant(filein):
    """
    Converts Hysplit dataset to be cf compliant.
    Also, reformats julian datatime to standard datetime and creates LAT LONG arrays of the model domain.
    """
    ## open dataset
    ds = xr.open_dataset(filein)

    ## get PM25 as numpy array
    PM25 = ds.PM25.values
    ## get time flags as numpy array
    tflag = ds.TFLAG.values

    ## get first time index...this will need to be rethought but works for 00Z initialization times
    hysplit_start = str(tflag[0, 0, 0]) + "0" + str(tflag[0, 0, 1])

    ## convert from julian datetime to standard datetime
    start = datetime.strptime(hysplit_start, "%Y%j%H%M%S").strftime("%Y%m%d%H%M%S")
    print(f"start: {start}")

    ## get last time index...this will need to be rethought but works (most of the time) with the if else statement below
    hysplit_stop = str(tflag[-1, 0, 0]) + str(tflag[-1, 0, 1])

    ## convert from julian datetime to standard datetime
    if len(hysplit_stop) < 9:
        stop = datetime.strptime(hysplit_stop, "%Y%j%H").strftime("%Y%m%d%H%M%S")
    else:
        stop = datetime.strptime(hysplit_stop, "%Y%j%H%M%S").strftime("%Y%m%d%H%M%S")
    print(f"stop: {stop}")

    ## create a new datetime numpy array with one hour frequency
    date_range = pd.date_range(start, stop, freq="1H")

    ## get x coordinate dimensions and create an array
    xnum = ds.dims["COL"]
    dx = ds.attrs["XCELL"]
    xorig = ds.attrs["XORIG"]
    x = np.arange(0, xnum)

    ## get y coordinate dimensions and create an array
    ynum = ds.dims["ROW"]
    dy = ds.attrs["YCELL"]
    yorig = ds.attrs["YORIG"]
    y = np.arange(0, ynum)

    ## create LAT and LONG 2D arrays based on the x and y coordinates
    X = np.arange(0, xnum) * dx + xorig
    Y = np.arange(0, ynum) * dy + yorig
    XX, YY = np.meshgrid(X, Y)

    ## get z coordinate dimensions and create an array
    Z = np.array(ds.attrs["VGLVLS"][:-1])
    z = np.arange(0, len(Z))

    ## create new dataset a set up to be CF Compliant
    ds_cf = xr.Dataset(
        data_vars=dict(
            pm25=(["time", "z", "y", "x"], PM25.astype("float32")),
            x=(["x"], x.astype("int32")),
            y=(["y"], y.astype("int32")),
            z=(["z"], z.astype("int32")),
            Times=(["time"], date_range.astype("S19")),
        ),
        coords=dict(
            LONG=(["y", "x"], XX.astype("float32")),
            LAT=(["y", "x"], YY.astype("float32")),
            HEIGHT=(["z"], Z.astype("float32")),
            time=date_range,
        ),
        attrs=dict(description="BlueSky Canada PM25 Forecast"),
    )

    ## add axis attributes from cf compliance
    ds_cf["time"].attrs["axis"] = "Time"
    ds_cf["x"].attrs["axis"] = "X"
    ds_cf["y"].attrs["axis"] = "Y"
    ds_cf["z"].attrs["axis"] = "Z"

    ## add units attributes from cf compliance
    ds_cf["pm25"].attrs["units"] = "um m^-3"
    ds_cf["LONG"].attrs["units"] = "degree_east"
    ds_cf["LAT"].attrs["units"] = "degree_north"

    return ds_cf

