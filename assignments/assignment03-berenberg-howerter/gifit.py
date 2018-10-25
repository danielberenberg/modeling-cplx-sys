import sys, os
import imageio
from glob import glob

place = "figs/traffic_gif/"

# Get all gif image files
imgfiles = sorted(glob(place+'*'))

# Make GIF!!!!   Naming convention:
#                                t: number of timesteps
#                               br: bikerate (bike speed delay)
#                              b2c: proportion of bikes in the system
#                            flows: single flow rate if uniform from every direction
#                                   or flow for each direction
#                   avg cflow stat: average proportion of cars moving per time step over entire time interval
#                                   average proportion of cars moving per time step over last 1/3 of time interval
#                   avg bflow stat: average proportion of bikes moving per time step

with imageio.get_writer('figs/gifs/last_traffic_gif.gif', mode='I') as writer:
    for filename in imgfiles:
        image = imageio.imread(filename)
        writer.append_data(image)
