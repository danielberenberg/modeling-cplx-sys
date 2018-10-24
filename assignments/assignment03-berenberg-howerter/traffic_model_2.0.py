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

# Our main arguments to change:
#                             : b_rate = time delay for bicycles
#                             : bike2car = prob. that a veh entering
#                                          the system is a bicycle
#                             : time = total time to run sys for
#                             : flow rates

b_rate = 3

type2name = {"I":-1, "E":0, "R":1, "C":2}
for i in range(b_rate):
    type2name['B'+str(i)] = 3+i

coin_flip = lambda p: random.random() < p
place = "figs/traffic_gif/"

os.system("rm {}".format(os.path.join(place,"*")))
plt.close()
bike2car  = 0.2  # proportion of vehicles that enter the system that are bikes
dim = 100        # dimensions of the system/grid
time = 100      # total time system runs for
system = np.zeros((dim,dim))

# flow_rates: (dict) - probability of vehicles entering the system
#                     from N,S,E, or W directions @ time t
allflowseq = True
if allflowseq:
    flowrate = 0.4
    flows = {d:flowrate for d in set("NSEW")}
else:
    flows = {"N":0.2,
             "S":0.1,
             "E":0.05,
             "W":0.15}

# incoming traffic zones:|
N_traffic = [dim//4,dim//2,dim - dim//4]
W_traffic = [dim//4,dim//2,dim - dim//4]

# laying roads
traffic_in    = lay_roads(N_traffic,W_traffic,system)
intersections, system = place_intersections(system)

flowstats = [] # number of vehicles that move per time step / total vehicles in system
exitflows = []
timesteps = []

for t in range(time):
    shadow = system.copy()
    carmove = 0
    bikemove = 0
    # vehicles entering the system
    veh_in = 0
    veh_out = 0
    for direction in flows:
        for t_in in traffic_in[direction]:
            if shadow[t_in] == type2name["R"] and coin_flip(flows[direction]):
                if coin_flip(bike2car):
                    for i in range(b_rate):
                        if (t+i) % b_rate == 0:
                            system[t_in] = type2name["B"+str(i)]   # a bike appears
                else:
                    system[t_in] = type2name["C"]   # a car appears
                veh_in +=1

    # do general, nonintersection related movement
    for i in range(b_rate):
        if (t+i) % b_rate == 0:
            # find and move bikes
            for bi, bj in zip(*np.where(shadow == type2name["B"+str(i)])):
                mv = move_vehicle(type2name,"B"+str(i),bi, bj, shadow, system)
                if mv:
                    if mv['veh_move']:
                        bikemove +=1
                        if mv['veh_exit']:
                            veh_out +=1

    for ci, cj in zip(*np.where(shadow == type2name["C"])):
        mv = move_vehicle(type2name,"C",ci,cj,shadow,system)
        if mv:
            if mv['veh_move']:
                carmove +=1
                if mv['veh_exit']:
                    veh_out +=1

    # look at intersections, move appropriately
    # who's at an intersection?
    for I in intersections:
        for direction in I.nsew.keys():
            if shadow[I.nsew[direction]] > 1 and direction not in I.queue:
                I.queue.append(direction)

        if I.queue:
            dmover = I.queue.pop(0)
            i,j = I.nsew[dmover]
            next_coord = random.choice(I.potential_locs[dmover])
            if shadow[next_coord] == type2name["R"]:
                system[next_coord] = shadow[i,j]
                system[i,j] = type2name["R"]
                if shadow[i,j] == 'C':
                    carmove +=1
                elif shadow[i,j] =='B':
                    bikemove +=1

    carmove += cars_pass_bikes(type2name,b_rate,shadow,system)

    # total number of C and B at start of timestep:
    uni, cts = np.unique(shadow, return_counts = True)
    cnts = dict(zip(uni,cts))
    totcar = cnts.get(type2name['C'],0)
    totbike = 0
    totbikecanmove = 0
    Bs = [c for c in cnts if c > type2name['C']]
    for B in Bs:
        totbike += cnts[B]
    for i in range(b_rate):
        if (t+i) % b_rate == 0 and type2name['B{}'.format(str(i))] in cnts:
            totbikecanmove = cnts[type2name['B'+str(i)]]

    if totcar == 0:
        carstat = -1
    else:
        carstat = carmove/totcar #+ 0.01

    if veh_in == 0:
        existat = -1
    else:
        existat = veh_out/veh_in

    if totbike == 0:
        bikestat1 = -1
    else:
        bikestat1 = bikemove/totbike #- 0.01

    if totbikecanmove == 0:
        bikestat2 = -1
    else:
        bikestat2 = bikemove/totbikecanmove

    flowstats.append((carstat,bikestat1,bikestat2))
    exitflows.append(existat)
    timesteps.append(t)

    avg_cfs, avg_2half_cfs , avg_bfs= plot_grid(system,timesteps,flowstats,exitflows,t)

    if t == time - 1:
        plot_final_stats(timesteps,flowstats,exitflows)


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

if allflowseq:
    path = '_t-{}_br-{}_b2c-{}_flows-{}_avgcf-{}-{}_avgbf-{}'.format(time,b_rate,round(bike2car,2),round(flows['N'],2),round(avg_cfs,3),round(avg_2half_cfs,3),round(avg_bfs,3))
else:
    path = '_t-{}_br-{}_b2c-{}_flowN-{}-S-{}-E-{}-W-{}_avgcf-{}-{}_avgbf-{}'.format(time,b_rate,round(bike2car,2),round(flows['N'],2),round(flows['S'],2),round(flows['E'],2),round(flows['W'],2),round(avg_cfs,3),round(avg_2half_cfs,3),round(avg_bfs,3))
with imageio.get_writer('figs/traffic_gif'+path+'.gif', mode='I') as writer:
    for filename in imgfiles:
        image = imageio.imread(filename)
        writer.append_data(image)

os.system('scp figs/final_stats.pdf figs/final_stats'+path+'.pdf')
