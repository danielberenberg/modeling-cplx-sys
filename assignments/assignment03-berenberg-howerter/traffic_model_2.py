import numpy as np
import random
import pickle
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors
from matplotlib import rc
import multiprocessing as mp
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import sys, os
import imageio
from glob import glob
from traffic_model_2_functions import lay_roads, place_intersections, cars_pass_bikes, move_vehicle, plot_grid, plot_final_stats

MAX_PLOTPROCS = 5
POOL = mp.Pool(MAX_PLOTPROCS)

# Our main arguments to change:
#                             : time = total time to run sys for
#                             : b_rate = time delay for bicycles
#                             : flows = flow rates from each direction
#                             : bike2car = prob. that a veh entering
#                                          the system is a bicycle
#                             : dim = dimensions of the system grid

def traffic_model(bike2car,
                  time,
                  b_rate,
                  flows,
                  N_traffic,
                  W_traffic,
                  dim,
                  odir):

    """
    Model traffic on a 2-d grid with two types of vehicles,
    bicycles and cars.

    args:
        :bik2car (float) - ratio of bicycles to cars
        :time (int) - maximum number of timesteps
        :flows (dict str --> float) - incoming traffic flow rates for N,S,E,W
        :N_traffic 
    """
    dimx, dimy = dim
    system = np.zeros((dimx,dimy))
    type2name = {"I":-1, "E":0, "R":1, "C":2}
    for i in range(b_rate):
        type2name['B'+str(i)] = 3+i

    coin_flip = lambda p: random.random() < p
    place = odir
    os.makedirs(os.path.join(place,"png"), exist_ok=True)
    print("png directory: {}".format(os.path.join(place,"png")))
    os.system("rm {}".format(os.path.join(place,"png","*")))
    plt.close()


    # laying roads
    traffic_in    = lay_roads(N_traffic,W_traffic,system)
    intersections, system = place_intersections(system)

    flowstats = [] # number of vehicles that move per time step / total vehicles in system
    exitflows = []
    timesteps = []
    in_flow = np.zeros(time)
    out_flow = np.zeros(time)

    for t in range(time):
        shadow = system.copy()
        carmove = 0
        bikemove = 0
        intbikemove = 0
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
                dmover = I.queue[0]
                i,j = I.nsew[dmover]
                next_coord = random.choice(I.potential_locs[dmover])
                if shadow[next_coord] == type2name["R"]:
                    system[next_coord] = shadow[i,j]
                    system[i,j] = type2name["R"]
                    if shadow[i,j] == type2name['C']:
                        carmove +=1
                    elif shadow[i,j] in [type2name['B'+str(i)] for i in range(b_rate)]:
                        bikemove +=1
                        intbikemove += 1
                    I.queue.pop(0)

        carmove += cars_pass_bikes(type2name,b_rate,shadow,system)

        # total number of C and B at start of timestep:
        uni, cts = np.unique(shadow, return_counts=True)
        cnts = dict(zip(uni,cts))
        totcar = cnts.get(type2name['C'],0)
        totbike = 0
        totbikecanmove = 0
        Bs = [c for c in cnts if c > type2name['C']]
        for B in Bs:
            totbike += cnts[B]
        for i in range(b_rate):
            if (t+i) % b_rate == 0 and type2name['B{}'.format(str(i))] in cnts:
                totbikecanmove = cnts[type2name['B'+str(i)]] + intbikemove

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

        in_flow[t] += veh_in
        out_flow[t] += veh_out

        #plot_grid(system,timesteps,flowstats,exitflows,t)
        POOL.apply_async(plot_grid, (deepcopy(system),
                                     deepcopy(timesteps),
                                     deepcopy(flowstats),
                                     deepcopy(exitflows),
                                     deepcopy(t),
                                     deepcopy(in_flow[:t+1]),
                                     deepcopy(out_flow[:t+1]),
                                     os.path.join(odir,"png")))

        if t == time - 1:
            plot_final_stats(timesteps,flowstats,exitflows,odir=place)

    # Get average stats at end:
    carstats = [f[0] for f in flowstats]
    totbikstats = [f[1] for f in flowstats]
    bikstats = [f[2] for f in flowstats]
    avg_cfs = np.mean([cs for cs in carstats if cs >= 0])
    avg_2half_cfs = np.mean([cs for cs in carstats[len(carstats)//3:] if cs >= 0])
    avg_bfs = np.mean([bs for bs in bikstats if bs >= 0])
    avg_tbfs = np.mean([bs for bs in totbikstats if bs >= 0])
    avg_exs = np.mean([e for e in exitflows if e >= 0])


    # Get all gif image files
    imgfiles = sorted(glob(os.path.join(place, '*.png')))
    # Make GIF!!!!   Naming convention:
    #                                t: number of timesteps
    #                               br: bikerate (bike speed delay)
    #                              b2c: proportion of bikes in the system
    #                            flows: single flow rate if uniform from every direction
    #                                   or flow for each direction
    #                   avg cflow stat: average proportion of cars moving per time step over entire time interval
    #                                   average proportion of cars moving per time step over last 1/3 of time interval
    #                   avg bflow stat: average proportion of bikes moving per time step

    #if allflowseq:
    #    path = '_t-{}_br-{}_flows-{}_b2c-{}_avgcf-{}-{}_avgbf-{}'.format(time, b_rate, round(flows['N'],2), round(bike2car,2), round(avg_cfs,3), round(avg_2half_cfs,3), round(avg_bfs,3))
    #else:
    path = '_t-{}_br-{}_flowN-{}-S-{}-E-{}-W-{}_b2c-{}_avgcf-{}-{}_avgbf-{}'.format(time,
                                                                                    b_rate, 
                                                                                    round(flows['N'],2), 
                                                                                    round(flows['S'],2), 
                                                                                    round(flows['E'],2), 
                                                                                    round(flows['W'],2), 
                                                                                    round(bike2car,2), 
                                                                                    round(avg_cfs,3), 
                                                                                    round(avg_2half_cfs,3), 
                                                                                    round(avg_bfs,3))
    with imageio.get_writer(os.path.join(place,"traffic.gif"), mode='I') as writer:
        trk.totbike[t] = sum([cnts[k] for k in type2id if k in cnts if k > type2id['C']])
        for filename in imgfiles:
            image = imageio.imread(filename)
            writer.append_data(image)

    #os.system('scp figs/final_stats.pdf figs/final_stats'+path+'.pdf')

    # Save stats:
    allstats = {'carflows':[fs[0] for fs in flowstats],
                'totbikeflows':[fs[1] for fs in flowstats],
                'bikeflows':[fs[2] for fs in flowstats],
                'exitflows':exitflows,
                'timesteps':timesteps}

    pklfilename = os.path.join(place,"FinalStats" + path + ".pkl")
    #pklfilename = 'pickles/final_stats'+path+'.pkl'

    with open(pklfilename, 'wb') as handle:
        pickle.dump(allstats, handle, protocol=pickle.HIGHEST_PROTOCOL)
