import context
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.plots import add_metpy_logo, SkewT
from metpy.units import units
from siphon.simplewebservice.wyoming import WyomingUpperAir

from context import root_dir


###########################################

# Change default to be better for skew-T
plt.rcParams["figure.figsize"] = (9, 9)
plt.rcParams.update({"font.size": 16})

###########################################

# Upper air data can be obtained using the siphon package, but for this example we will use
# some of MetPy's sample data.

date = datetime(2021, 6, 30, 0)
station = "CWVK"

col_names = ["pressure", "height", "temperature", "dewpoint", "direction", "speed"]

# df = pd.read_fwf(get_test_data('jan20_sounding.txt', as_file_obj=False),
#                  skiprows=5, usecols=[0, 1, 2, 3, 6, 7], names=col_names)

df = WyomingUpperAir.request_data(date, station)


# Drop any rows with all NaN values for T, Td, winds
df = df.dropna(
    subset=("temperature", "dewpoint", "direction", "speed"), how="all"
).reset_index(drop=True)

###########################################
# We will pull the data out of the example dataset into individual variables and
# assign units.


p = df["pressure"].values * units.hPa
T = df["temperature"].values * units.degC
Td = df["dewpoint"].values * units.degC
wind_speed = df["speed"].values * units.knots
wind_dir = df["direction"].values * units.degrees
u, v = mpcalc.wind_components(wind_speed, wind_dir)


###########################################
skew = SkewT()

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(p, T, "r", lw=3)
skew.plot(p, Td, "g", lw=3)

# Set spacing interval--Every 50 mb from 1000 to 100 mb
my_interval = np.arange(100, 1000, 50) * units("mbar")

# Get indexes of values closest to defined interval
ix = mpcalc.resample_nn_1d(p, my_interval)

# Plot only values nearest to defined interval values
skew.plot_barbs(p[ix], u[ix], v[ix])

# Add the relevant special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()
skew.ax.set_ylim(1000, 100)

plt.ylabel("Hectopascal", fontsize=18)
plt.xlabel("Degree Celsius", fontsize=16)

datestr = date.strftime("%HZ %d %B %Y")

plt.title(f"{station} Vernon, BC", loc="left")
plt.title(f"{datestr}", loc="right", fontsize=15)

# Show the plot
plt.savefig(str(root_dir) + "/img/sounding.png", dpi=250, bbox_inches="tight")
