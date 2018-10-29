import os
import pickle
import random
import imageio
import argparse
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from glob import glob
from copy import deepcopy
from matplotlib import colors
from matplotlib import rc
from multiprocessing import Pool
from matplotlib.colors import BoundaryNorm, ListedColormap 

rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
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

cmap = ListedColormap([I_color, E_color, R_color, C_color, B_color])
norm = BoundaryNorm([-1,0,1,2,3,8], 5)

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
        """
        An intersection is a junction point on a grid matrix.
        """
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
    
    @staticmethod
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
    
        return intersections

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

class PassingRegime:
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
    
    @staticmethod
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
                    R = PassingRegime(direction,(i,j))
                    G[R.old_loc] = type2name['R']
                    G[R.new_loc] = type2name['C']
                    carmoves += 1
    
        return carmoves


class StatTracker:
    def __init__(self,nplotters=5,**kwargs):
        """
        StatTracker is a ball of data that is collected throughout the course of 
        the simulation. It allows us to generate plots/visualizations and make
        computations slightly more modularly.
        
        StatTracker has a plotting method that is meant to be called from
        the main loop. We specify a maximum number of processes to help the
        plotting process

        StatTrackers are instantiated with no parameters to keep track of.
        Use setattr to set stats to track

        args:
            :nplotters (int) -- the number of processes to delegate to a plotting
        """
        for k,v in kwargs.items():
            if not isinstance(v, (int,float)) and v >= 0:
                raise ValueError("{} should be a constant >= 0".format(k))

            setattr(self,k,v)

    def save(self, filename):
        with open(filename,"wb") as p:
            pickle.dump(vars(self),p)

    def new_attribute(self,attr_names,watch_clauses):
        for attr_name, watch_clause in zip(attr_names, watch_clauses):
            if not hasattr(self, watch_clause):
                raise RuntimeError("{} is not a predefined max attribute".format(watch_clause))
            
            setattr(self, attr_name, np.zeros(getattr(self, watch_clause)))

def plot_movement(ax,t, cars_moved,total_cars,bikes_moved,total_bikes):
    kw = {"linestyle":"-","marker":"o"}
    ax.set_ylim(-0.1,1.1)
    ax.plot(t,cars_moved/total_cars, **kw, label="Prop. movable cars")
    ax.plot(t,bikes_moved/total_bikes, **kw, label="Prop. movable bikes")
    ax.set_title("Movement")
    ax.legend(loc=3,fontsize=8)

def plot_throughput(ax,t,inflow,outflow):
    kw = {"linestyle":"-","marker":"o"}
    ax.set_title("Thoughput")
    ax.plot(t,inflow,**kw, label="Input")
    ax.plot(t,outflow, **kw, label="Output")
    ax.legend(loc=3,fontsize=8)
    ax.set_xlabel('$t$',fontsize=10)

def plot_grid(ax, grid):
    ax.axis('off') 
    ax.imshow(grid, interpolation='nearest', cmap=cmap, norm=norm)

def gif_it(pngdir, savedir):
    with imageio.get_writer(os.path.join(savedir,'traffic.gif'), mode='I') as writer:
        for png in sorted(glob(os.path.join(pngdir,"*.png"))):
            image = imageio.imread(png)
            writer.append_data(image)
    
def plot(G,
         t,
         cars_moved,
         bikes_moved,
         total_cars,
         total_bikes,
         inflow,
         outflow,
         dest): #ts,fs,es,t,inflow,outflow,place):
    """
    plot the grid and some other visualizations along with it.
    args:
        :G (np.ndarray) - 2d grid
        :ts (list) - timestep values
        :fs (list) - bike stats
        :es (list) - exit stats
        :t - timestep
        :place (str) - path to plot to
    """
    # num movable/num total for both cars and bikes, num in/num out for both vehicles (throughput)
    plt.close()   #---------------------- clear previous figure
    fig = plt.figure(figsize=(10,15)) #-- new figure
    
    # the subaxes
    ax1 = plt.subplot2grid((4,2), (0,0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((4,2), (2,0), colspan=2, rowspan=1)

    ax3 = plt.subplot2grid((4,2),(3,0), colspan=2, rowspan=1, sharex=ax2)
    
    # ------------------------- axis 1 (traffic grid) -------------------------- #
    plot_grid(ax1, G)
    # ------------------------- axis 2 (movements) -------------------------- #
    plot_movement(ax2, t, cars_moved, total_cars, bikes_moved, total_bikes)
    # ------------------------- axis 3 (throughputs) -------------------------- #
    plot_throughput(ax3, t, inflow, outflow)
    
    plt.tight_layout()
    plt.savefig(os.path.join(dest,"CAgif_timestep_{:04d}".format(len(total_bikes)-1)),bbox_inches='tight')

def setup():
    parser = argparse.ArgumentParser(description="traffic modeling cellular automata, investigating vehicle interactions")
    parser.add_argument("-d",
                        type=intgt0, 
                        nargs=2,
                        help="x dimensions of the system",
                        default=(100,100))

    parser.add_argument("-T",help="total time to run system for",
                        default=250, type=intgt0)
    
    parser.add_argument("-b",help="probability of a bike spawning",
                        default=0., type=float01)

    parser.add_argument("-r","--b-rate",dest="b_rate",help="time delay for bicycles",
                        default=3, type=intgt0)

    parser.add_argument("--flow","-f",dest="flow",nargs=4,
                        help="probability of a vehicle flowing in from the N,S,E,W respectively",
                        default=(1.,1.,1.,1.), type=float01)

    parser.add_argument("-o",dest="output_directory",
                        help="output directory",
                        default="trafsim",
                        type=exists)

    parser.add_argument("-p","--num-plotters",dest="nb_plotters",
                        type=intgt0,default=5,help="number of processes to plot with")

    return parser

def getargs():
    """
    tease input arguments into their proper format by parsing them in with parse_args()
    and further validating/organizing them, summarize the parameter inputs with 
    an adhoc file format in the toplevel of the outdirectory
    """

    raw = vars(setup().parse_args()) # gets a dictionary mapping string keys to their values
    arguments = dict()               # fresh, teased out args

    arguments['T'] = raw['T']
    arguments['beta'] = raw['b']
    arguments['mu']   = raw['b_rate']

    dimx, dimy = arguments['dimensions'] = raw['d']

    # do prepare output directory:
    arguments['out'] = outdir = raw['output_directory']
    arguments['png'] = exists(os.path.join(outdir,"png"))         # intermediary png directory
    arguments['pkl'] = exists(os.path.join(outdir,"pickles"))     # pickle directory
    arguments['fig'] = exists(os.path.join(outdir,"figs"))        # figures directory (final stats)

    os.system("rm {}".format(os.path.join(arguments['png'], "*")))
    # format flowrates
    arguments['flows'] = dict(zip(list(set("NSEW")),raw['flow']))
    arguments['plotters'] = Pool(raw['nb_plotters'])

    with open(os.path.join(outdir,"arguments.txt"),"w") as argfile:
        kw = {"file":argfile}
        title = "Argument Summary"
        datecreated = "Written at: {}".format(dt.datetime.now().strftime("%A, %b %-d, %Y [%H:%M:%S]")) 
        datecreated = (datecreated + "\n{0}").format("+"*len(datecreated))
        print(title,**kw)
        print(datecreated,**kw)
        # ------------------------------
        print("Output locations:",**kw)
        cdots = len(arguments['out'])*"." + "/" 
        png = arguments['png'].split("/")[-1]
        pkl = arguments['pkl'].split("/")[-1]
        fig = arguments['fig'].split("/")[-1]
        print("\tTop level---->{}".format(arguments['out']),**kw)
        print("\tPNG loc------>{0}{1}".format(cdots,png),**kw)
        print("\tpickles loc-->{0}{1}".format(cdots,pkl),**kw)
        print("\tfigures loc-->{0}{1}".format(cdots,fig),**kw)
        # -----------------------------
        print("\nIncoming flow probs",*"north -- {N:0.2f},south -- {S:0.2f},east --- {E:0.2f},west --- {W:0.2f}".format(**arguments['flows']).split(","),sep="\n\t",**kw)
        # ----------------------------
        print("\np(bike_spawns) = {}".format(arguments['beta']),**kw)
        print("bikes move every {} steps".format(arguments['mu']),**kw)
        print("dimensions of grid : {}".format(arguments['dimensions']),**kw)

    return arguments

def exists(directory):
    """
    function is called to verify the output directory exists, if it doesn't make it.
    """
    directory = str(directory)
    os.makedirs(directory, exist_ok=True)
    return directory

def intgt0(val):
    val = int(val)
    if not val > 0:
        raise ValueError("Expected integer > 0")

    return val

def float01(val):
    val = float(val)
    if not 0 <= val <= 1:
        raise ValueError("Expected float in [0,1]")

    return val

def coin_flip(p):
    return random.random() < p


def main(**kwargs):
    """
    run the main traffic simulation.
    expected args:
        :beta (float in [0,1])     - probability of a bike spawning
        :mu   (int > 0)            _ the move rate 
        :T (int > 0)               - maximum timestep 
        :flows (dictionary)        - dictionary denoting the incoming flow rates from each direction
        :dimensions (tuple size 2) - the (x,y) dimensions of the gridworld 
        :out,png,pkl,fig           - output directory throughout the simulation

    the main behavior of the simulation is to step through time until T, pumping in new vehicles, B w/ prob beta
    flowing in from each direction according to their flow rates and observe their interactions
    """
    global norm
    beta      = kwargs['beta']
    mu        = kwargs['mu']
    T         = kwargs['T']
    flow      = kwargs['flows']
    dimx,dimy = kwargs['dimensions']
    figs      = kwargs['fig']
    png       = kwargs['png']
    pkl       = kwargs['pkl']
    
    
    # lay roads
    system    = np.zeros((dimx,dimy))
    ntraf = [10,20,40,50]
    wtraf = [10,20,50,60]
    traffic_in = lay_roads(ntraf,wtraf,system)
    #traffic_in = lay_roads([(dimx-24)//2, (dimx-8)//2, dimx//2, dimx//2+4, dimx//2+8,dimx//2+12], 
    #                       [(dimy-24)//2,(dimy-8)//2,dimy//2,dimy//2+4,dimy//2+8, dimy//2+12], 
    #                       system)
    # place intersections
    intersections = Intersection.place_intersections(system)

    # maps cell types to ids 
    type2id = {"I":-1, "E":0, "R":1, "C":2}
    type2id.update({'B{}'.format(i):3+i for i in range(mu)})
    # will keep track of stats we care about
    trk = StatTracker(**{"T":T})
    trk.new_attribute(["t","inflow","outflow","bikes_moved","cars_moved",
                       "totbike","totcar"],["T"]*7)
    setattr(trk,"png_dir",png)
    setattr(trk,"gif_dir",figs)

    for t in range(T):
        trk.t[t] = t
        # creating a copied version of the grid allows for synchronous updates
        shadow = deepcopy(system)
        # class of bikes that will move this turn
        bike_class = "B{}".format(mu - (mu - (t % mu))) 
        #               for each direction, 
        for direction in flow:  # for each of the entry points from that direction
            for t_in in traffic_in[direction]:
                # roll dice whether or not a vehicle will enter here
                if shadow[t_in] == type2id["R"] and coin_flip(flow[direction]):
                    # roll for bike
                    if coin_flip(beta):
                        system[t_in] = type2id[bike_class]
                    else:
                        system[t_in] = type2id["C"]   # a car appears
                    trk.inflow[t] +=1 
                    #veh_in +=1

        # do general, nonintersection related movement
        for bi, bj in zip(*np.where(shadow == type2id[bike_class])):
            mv = move_vehicle(type2id, bike_class, bi, bj, shadow, system)
            if mv:
                trk.bikes_moved[t] += int(mv['veh_move'])
                trk.outflow[t] += int(mv['veh_exit'])

        for ci, cj in zip(*np.where(shadow == type2id["C"])):
            mv = move_vehicle(type2id, "C", ci, cj, shadow, system)
            if mv:
                trk.cars_moved[t] += int(mv['veh_move'])
                trk.outflow[t] += int(mv['veh_exit'])


        # look at intersections, move appropriately
        for I in intersections:                             # iterating through the intersections,
            for direction in I.nsew.keys():             # this routine identifies stopped traffic and
                if shadow[I.nsew[direction]] > 1 and direction not in I.queue: # places new positions in the queue
                    I.queue.append(direction)                                       
            
            # if there are stopped vehicles
            if I.queue:
                mover = I.queue[0] 
                i,j = I.nsew[mover]  # the location of the potential_mover
                next_coord = random.choice(I.potential_locs[mover]) # the next location for the mover 
                if shadow[next_coord] == type2id["R"]: # can only move if the location is not occupied
                    system[next_coord] = shadow[i,j]   # move the vehicle 
                    system[i,j] = type2id["R"]         # previous cell forgets it
                    if shadow[i,j] == type2id['C']:
                        trk.cars_moved[t] += 1
                    elif shadow[i,j] in [type2id['B{}'.format(i)] for i in range(mu)]:
                        trk.bikes_moved[t] +=1

                    # successfully moved so get rid of it in queue
                    I.queue.pop(0)
        
        # let cars pass bikes
        trk.cars_moved[t] += PassingRegime.cars_pass_bikes(type2id,mu,shadow,system)

        # total number of C and B at start of timestep:
        uni, cts = np.unique(shadow, return_counts=True)
        cnts = dict(zip(uni,cts))
        trk.totcar[t] = cnts.get(type2id['C'],0)
        trk.totbike[t] = sum([cnts[k] for k in cnts if k > type2id['C']])
        
        args = list(map(deepcopy,
                        [system,
                         trk.t[:t+1],
                         trk.cars_moved[:t+1],
                         trk.bikes_moved[:t+1],
                         trk.totcar[:t+1],
                         trk.totbike[:t+1],
                         trk.inflow[:t+1],
                         trk.outflow[:t+1]])) + [png]
        
        #kwargs['plotters'].apply_async(plot,args)
        plot(*args)
    #kwargs['plotters'].apply_async(gif_it, (png,figs))
    gif_it(png, figs)
    pklfilename = os.path.join(pkl,"FinalStats.pkl")
    trk.save(pklfilename) 
    print('done')


if __name__ == "__main__":
    main(**getargs())
