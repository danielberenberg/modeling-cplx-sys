import sys, os
import imageio
from glob import glob

from traffic_model_2_functions import *
from traffic_model_2 import *

# Our main arguments to change:
#                             : time = total time to run sys for
#                             : b_rate = time delay for bicycles
#                             : flows = flow rates from each direction
#                             : bike2car = prob. that a veh entering
#                                          the system is a bicycle
#                             : dim = dimensions of the system grid

bike2car  = 0.05  # proportion of vehicles that enter the system that are bikes
                  # Copenhagen is @ ~0.46
                  # Amsterdam ~0.57
                  # Portland, Or ~0.063
                  # San Fran, CA ~0.039
                  # Davis, CA ~0.166
                  # VT statewide ~0.006
time = 200
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


# incoming traffic zones:|
N_traffic = [dim//20, dim//8, dim//4, dim//3, dim//3+dim//10, dim//2,  dim - dim//4]
W_traffic = [dim//10, dim//4, dim//3, 2*dim//3 - dim//4, dim//2, dim//2 + dim//6, dim//6, dim - dim//4, dim - dim//8]

traffic_model(bike2car,time,b_rate,flows,allflowseq,N_traffic,W_traffic,dim)
