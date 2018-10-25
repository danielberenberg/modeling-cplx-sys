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

coin_flip = lambda p: random.random() < p
place = "figs/traffic_gif/"

# colors:
I_color = 'rosybrown'
R_color = 'lightgray'
E_color = 'honeydew'
C_color = 'slateblue'
C_colord = 'darkslateblue'
C_color2 = 'navy'
B_color = 'orange'
B_colord = 'darkorange'
B_color2 = 'gold'
a = 0.8 #alpha val

def plot_grid(G,ts,fs,es,t):
    plt.close()
    fig = plt.figure(figsize=(10,15))
    gridspec.GridSpec(3,2)

    plt.subplot2grid((3,2), (0,0), colspan=2, rowspan=2)
    plt.axis('off')
    bounds = [-1,0,1,2,3,4]
    cmap = colors.ListedColormap([I_color, E_color, R_color, C_color, B_color])
    norm = colors.BoundaryNorm(bounds,5)
    plt.imshow(G, interpolation='nearest', cmap=cmap, norm=norm)

    plt.subplot2grid((3,2), (2,0), colspan=2, rowspan=1)
    carstats = [f[0] for f in fs]
    totbikstats = [f[1] for f in fs]
    bikstats = [f[2] for f in fs]

    avg_cfs = np.mean([cs for cs in carstats if cs >= 0])
    avg_2half_cfs = np.mean([cs for cs in carstats[len(carstats)//3:] if cs >= 0])
    avg_bfs = np.mean([bs for bs in bikstats if bs >= 0])
    avg_tbfs = np.mean([bs for bs in totbikstats if bs >= 0])
    avg_exs = np.mean([e for e in es if e >= 0])

    plt.plot([ts[0],ts[-1]],[avg_cfs,avg_cfs],color = C_color, alpha=a-.1,label='avg. prop. for cars')
    plt.plot([ts[0],ts[-1]],[avg_2half_cfs,avg_2half_cfs],color = C_color2, alpha=a-.1,label='avg. prop. for cars over last third of t interval')
    plt.scatter(ts[:-1], carstats[:-1], s=50,alpha=a, facecolors='none', edgecolors= C_color)#,label='prop. of cars moving in system at timestep t')
    plt.scatter(ts[-1], carstats[-1], s=50, facecolors='none', edgecolors= C_colord)

    plt.plot([ts[0],ts[-1]],[avg_bfs,avg_bfs],color = B_color, alpha=a-.1,label='avg. prop. for bikes that can move')
    plt.plot([ts[0],ts[-1]],[avg_tbfs,avg_tbfs],color = B_color2, alpha=a-.1,label='avg. prop. for all bikes')
    plt.scatter(ts[:-1], bikstats[:-1], s=50,alpha=a, facecolors='none', edgecolors= B_color)#, label='prop. of bikes moving, that can, at timestep t')
    plt.scatter(ts[-1], bikstats[-1], s=50, facecolors='none', edgecolors= B_colord)
    plt.scatter(ts, totbikstats, s=50,alpha=a, facecolors='none', edgecolors= B_color2) #,label='prop. of bikes moving, out of all bikes, at timestep t')

    plt.plot([ts[0],ts[-1]],[avg_exs,avg_exs],color = 'crimson',alpha=a-.1,label='avg. out/in ratio for veh in system')
    #plt.scatter(ts, es, s=50,alpha=0.4, facecolors='none', edgecolors='crimson',label='out/in ratio for veh in system')

    plt.ylim(-0.1,1.1)
    plt.xlabel('timestep, t',fontsize = 10)
    plt.ylabel('proportion of vehicles that moved at timestep, t',fontsize = 10)
    plt.legend(loc=3,fontsize = 8)
    #fig.tight_layout()
    #plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds)
    plt.savefig(os.path.join(place,"CAgif_timestep_{:04d}".format(t)),bbox_inches='tight')
    #plt.show()



def plot_final_stats(ts,fs,es):
    plt.close()
    fig = plt.figure(figsize=(15,5))

    carstats = [f[0] for f in fs]
    totbikstats = [f[1] for f in fs]
    bikstats = [f[2] for f in fs]

    avg_cfs = np.mean([cs for cs in carstats if cs >= 0])
    avg_2half_cfs = np.mean([cs for cs in carstats[len(carstats)//3:] if cs >= 0])
    avg_bfs = np.mean([bs for bs in bikstats if bs >= 0])
    avg_tbfs = np.mean([bs for bs in totbikstats if bs >= 0])
    avg_exs = np.mean([e for e in es if e >= 0])

    plt.plot([ts[0],ts[-1]],[avg_cfs,avg_cfs],color = C_color, alpha=a-.1,label='avg. prop. for cars')
    plt.plot([ts[0],ts[-1]],[avg_2half_cfs,avg_2half_cfs],color = C_color2, alpha=a-.1,label='avg. prop. for cars over last third of t interval')
    plt.scatter(ts[:-1], carstats[:-1], s=50,alpha=a, facecolors='none', edgecolors= C_color)#,label='prop. of cars moving in system at timestep t')
    plt.scatter(ts[-1], carstats[-1], s=50, facecolors='none', edgecolors= C_colord)

    plt.plot([ts[0],ts[-1]],[avg_bfs,avg_bfs],color = B_color, alpha=a-.1, label='avg. prop. for bikes that can move')
    plt.plot([ts[0],ts[-1]],[avg_tbfs,avg_tbfs],color = B_color2, alpha=a-.1, label='avg. prop. for all bikes')
    plt.scatter(ts[:-1], bikstats[:-1], s=50,alpha=a, facecolors='none', edgecolors= B_color)#, label='prop. of bikes moving, that can, at timestep t')
    plt.scatter(ts[-1], bikstats[-1], s=50, facecolors='none', edgecolors= B_colord)
    plt.scatter(ts, totbikstats, s=50,alpha=a, facecolors='none', edgecolors= B_color2)#,label='prop. of bikes moving, out of all bikes, at timestep t')

    plt.plot([ts[0],ts[-1]],[avg_exs,avg_exs],color = 'crimson',alpha=a-.1,label='avg. out/in ratio for veh in system')
    #plt.scatter(ts, es, s=50,alpha=0.4, facecolors='none', edgecolors='crimson',label='out/in ratio for veh in system')

    plt.ylim(-0.1,1.1)
    plt.xlabel('timestep, t',fontsize = 10)
    plt.ylabel('proportion of vehicles that moved at timestep, t',fontsize = 10)
    plt.legend(loc=3,fontsize = 7)
    plt.savefig("figs/final_stats.pdf",bbox_inches='tight')


def place_intersections(G):
    """
    - Place intersections on the grid G.
    - Intersections are placed at any place 4x4 subsection of G of the following forms:

        [0][1][1][0]  (1)
        [1][1][1][1]
        [1][1][1][1]
        [0][1][1][0],

    The function for now only support intersections of type (1).

    args:
        :NxN grid G
    returns:
        :the intersection placed grid
    """
    intersection_regime = np.array([[0,1,1,0],
                               [1,1,1,1],
                               [1,1,1,1],
                               [0,1,1,0]])
    intersection_i, intersection_j = np.where(find_regime(intersection_regime, G) == 1)
    intersections = []
    for i,j in zip(intersection_i, intersection_j):
        G[i+1,j+1] = -1
        G[i+2,j+2] = -1
        G[i+1,j+2] = -1
        G[i+2,j+1] = -1
        intersections.append(Intersection((i,j)))
        #Is = [i+1, i+2, i+1, i+2]
        #Js = [j+1, j+2, j+1, j+2]
        #G[(Is,Js)] = -1

    return intersections, G


def find_regime(submatrix, grid):
    """
    convolve throughout grid and return masks of indices that describe matches in
    the grid to the specified submatrix

    args:
        :submatrix -   2d array of dimensions n x m
        :grid      -   2d array symbolizing the world of dimensions N x M
    returns:
        :masks that describe matches of the regime in the grid
    """
    n, m = submatrix.shape
    N, M = grid.shape
    convolve_mat = np.zeros((N-n+1, M-m+1))

    for row in range(N - n + 1):
        for col in range(M - m + 1):
            view = grid[row:row + n, col:col + m]
            if np.array_equal(view, submatrix):
                convolve_mat[row, col] = 1
            else:
                convolve_mat[row, col] = 0

    return convolve_mat


class Intersection:
    def __init__(self, coordinates):
        i, j = self.i, self.j = coordinates

        # compute the neighborhood of coordinates
        # for this intersection
        self.nsew = {"N":(i,   j+1),
                     "S":(i+3, j+2),
                     "E":(i+1, j+3),
                     "W":(i+2, j)}

        self.potential_locs = {"N":[(i+1,j),
                                    (i+3,j+1),
                                    (i+2,j+3)],
                               "S":[(i+1,j),
                                    (i+2,j+3),
                                    (i,  j+2)],
                               "E":[(i+1,j),
                                    (i,  j+2),
                                    (i+3,j+1)],
                               "W":[(i,  j+2),
                                    (i+2,j+3),
                                    (i+3,j+1)]}
        self.queue = []

    def __str__(self):
        return "Intersection @ ({},{})".format(self.i,self.j)

    def __repr__(self):
        return str(self)



class Passing_regime:
    def __init__(self, direction, coordinates):
        i, j = self.i, self.j = coordinates

        if direction == 'Ebound':
            self.regime = np.array([[1,1,1,1],
                                    [2,3,1,1],
                                    [0,0,0,0]])
            self.old_loc = (i+1, j)

            self.new_loc = (i+1, j+3)

        elif direction == 'Wbound':
            self.regime = np.array([[0,0,0,0],
                                    [1,1,3,2],
                                    [1,1,1,1]])
            self.old_loc = (i+1,   j+3)

            self.new_loc = (i+1,   j)

        elif direction == 'Nbound':
            self.regime = np.array([[1,1,0],
                                    [1,1,0],
                                    [1,3,0],
                                    [1,2,0]])
            self.old_loc = (i+3,  j+1)

            self.new_loc = (i,   j+1)

        elif direction == 'Sbound':
            self.regime = np.array([[0,2,1],
                                    [0,3,1],
                                    [0,1,1],
                                    [0,1,1]])
            self.old_loc = (i,   j+1)

            self.new_loc = (i+3, j+1)

    def __str__(self):
        return "Car can pass bike @ ({},{})".format(self.old_loc[0],self.old_loc[1])

    def __repr__(self):
        return str(self)



def lay_roads(N_traffic, W_traffic, G):
    """
    args:
        N_traffic: (X length list) of x coords to roads
        W_traffic: (Y length list) of y coords to roads
    returns:
        :(dict) maps from direction in {N,S,E,W} to
         traffic incoming coordinates from N,S,E,W respectively
    """
    x,y = G.shape
    S_traffic = [i+1 for i in N_traffic]
    E_traffic = [j-1 for j in W_traffic]

    # laying roads
    for i1,i2 in zip(N_traffic,S_traffic):
        G[:,i1] = 1
        G[:,i2] = 1

    for j1,j2 in zip(W_traffic,E_traffic):
        G[j1,:] = 1
        G[j2,:] = 1

    return {"N":[(0, j) for j in N_traffic],"W":[(i, 0) for i in W_traffic],
            "S":[(x-1,j) for j in S_traffic],"E":[(i,y-1) for i in E_traffic]}

def move_vehicle(type2name, vtype, vi, vj, shadow, G):
    M, N = G.shape
    northmost = (vi == 0 and shadow[vi, vj+1] == type2name["E"])
    southmost = (vi == N - 1 and shadow[vi, vj-1] == type2name["E"])
    eastmost = (vj == M - 1 and shadow[vi+1, vj] == type2name["E"])
    westmost = (vj == 0 and shadow[vi - 1, vj] == type2name["E"])
    if northmost or southmost or eastmost or westmost:
        G[vi,vj] = type2name["R"]
        return {'veh_move':True, 'veh_exit':True}

    # E V
    #   R
    try:
        if shadow[vi+1, vj] == type2name["R"] and shadow[vi, vj -1] == type2name["E"]:
            G[vi,vj] = type2name["R"]
            G[vi+1, vj] = type2name[vtype]
            return {'veh_move':True, 'veh_exit':False}
    except IndexError:
        pass

    # R
    # V E
    try:
        if shadow[vi-1, vj] == type2name["R"] and shadow[vi, vj+1] == type2name["E"]:
            G[vi,vj] = type2name["R"]
            G[vi-1, vj] = type2name[vtype]
            return {'veh_move':True, 'veh_exit':False}
    except IndexError:
        pass
    # V R
    # E
    try:
        if shadow[vi,vj+1] == type2name["R"] and shadow[vi+1,vj] == type2name["E"]:
            G[vi,vj] = type2name["R"]
            G[vi, vj+1] = type2name[vtype]
            return {'veh_move':True, 'veh_exit':False}
    except IndexError:
        pass
    # E
    # R V
    try:
        if shadow[vi,vj-1] == type2name["R"] and shadow[vi-1, vj] == type2name["E"]:
            G[vi,vj] = type2name["R"]
            G[vi, vj-1] = type2name[vtype]
            return {'veh_move':True, 'veh_exit':False}
    except IndexError:
        pass


def cars_pass_bikes(type2name,b_rate,shadow,G):
    '''
    perform a regime search for cars that could pass bicycles
    args: shadow: current state of the system
          G     : next state of the system

    Bike Passing Regimes:

     R R R R  eastbound  -> move C from (i+1, j  ) to (i+1, j+3)
     C B R R
     E E E E

     E E E E  westbound  -> move C from (i  , j+3) to (i  , j  )
     R R B C
     R R R R

     R R E    northbound -> move C from (i+3, j+1) to (i  , j+1)
     R R E
     R B E
     R C E

     E C R    southbound -> move C from (i  , j  ) to (i+3, j  )
     E B R
     E R R
     E R R
    '''

    bikepass_regimes = {'Ebound':[np.array([[1,1,1,1],
                                           [2,3+i,1,1],
                                           [0,0,0,0]]) for i in range(b_rate)],
                        'Wbound':[np.array([[0,0,0,0],
                                            [1,1,3+i,2],
                                            [1,1,1,1]]) for i in range(b_rate)],
                        'Nbound':[np.array([[1,1,0],
                                            [1,1,0],
                                            [1,3+i,0],
                                            [1,2,0]]) for i in range(b_rate)],
                        'Sbound':[np.array([[0,2,1],
                                           [0,3+i,1],
                                           [0,1,1],
                                           [0,1,1]]) for i in range(b_rate)]}

    carmoves = 0
    for direction in bikepass_regimes:
        for regime in bikepass_regimes[direction]:
            regimei,regimej = np.where(find_regime(regime,shadow) == 1)
            for i,j, in zip(regimei,regimej):
                R = Passing_regime(direction,(i,j))
                G[R.old_loc] = type2name['R']
                G[R.new_loc] = type2name['C']
                carmoves += 1

    return carmoves
