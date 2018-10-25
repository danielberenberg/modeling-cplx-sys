import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import sys, os
import imageio
from glob import glob

from traffic_model_2_functions import *
from traffic_model_2 import *

bike2car  = 0.0  # proportion of vehicles that enter the system that are bikes

time = 700
b_rate = 3
# flow_rates: (dict) - probability of vehicles entering the system
#                     from N,S,E, or W directions @ time t
allflowseq = False
if allflowseq:
flowrate = 0.8
flows = {d:flowrate for d in set("NSEW")}
else:
flows = {"N":0.95,
"S":0.7,
"E":0.7,
"W":0.95}

dim = 100        # dimensions of the system/grid

traffic_model(bike2car,time,b_rate,flows,dim)
