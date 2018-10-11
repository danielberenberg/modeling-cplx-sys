"""
fast Diffusion Limited Aggregation
"""

import numpy as np
import random
from skimage import measure
from functools import reduce
import argparse
import matplotlib.pyplot as plt
import imageio
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion Limited Aggregation")
    parser.add_argument("-d",
                        dest="dimensions",
                        type=int,
                        help="dimensions of the matrix to conduct simulation",
                        default=100)

    parser.add_argument("-n",
                        dest="generations",
                        type=int,
                        help="number of generations",
                        default=100)

    parser.add_argument("-k",
                        dest="n_walkers",
                        type=int,
                        help="number of walkers in spawned at first in the system",
                        default=100)

    parser.add_argument("-s",
                        dest="outfile",
                        type=str,
                        default="DLA.gif",
                        help="name of the final gif")

    parser.add_argument("-f",
                        dest="frame_directory",
                        type=str,
                        default="out",
                        help="intermediary frame directory")

    return parser

def spawn_walkers(mat, N=10):
    """
    spawn N random walkers on the given matrix

    args:
        :mat - matrix to place walkers
        :N (int) - number of walkers to spawn_walkers
    """
    # viable locations to spawn are unoccupied
    viable_loc_mat = np.ones(mat.shape) - mat
    viable_locs = list(np.array(np.where(viable_loc_mat > 0)).T)
    assert len(viable_locs) >= N, "There are not {} spawn locations".format(N)

    spawn_locs = random.sample(viable_locs, N)
    args = zip(*spawn_locs)
    mat[list(args)] = 1

def move_walkers(mat):
    """
    parallelize the movement of random walkers; for now if a collision
    occurs then that is where they particles stop moving
    
    0) define a new matrix
    1) sweep through matrix and find walkers
        1a) clustered walkers are locked for this (and hence subsequent) timesteps
    2) vectorizedly decide walkers' next positions
    3) return the results

    args:
        :mat - matrix containing walkers
    """
    # label clusters with skimage magic
    walkers_lbld = measure.label(mat, connectivity=1)
    print("walkers labeled\n",walkers_lbld)
    # extract indexes by returning all indices that had a one
    idx = [tuple(i for i in np.squeeze(np.where(walkers_lbld == label)).T.tolist()) for label in np.unique(walkers_lbld) if label]
    print(idx)
    #print([np.where(walkers_lbld == label) for label in np.unique(walkers_lbld) if label] 
    #indices = list(reduce(lambda x1,x2: x1+x2
    M = np.zeros(mat.shape)  # new matrix
    maskX, maskY = [], [] 
    try:  # test if there are clusters > length 2, concatenate each such cluster, split into 2 sep. arrays
        maskX, maskY = zip(*reduce(lambda x1,x2: x1 + x2, filter(lambda x: len(x) > 1, idx)))
        M[maskX, maskY] = 1
    except TypeError:  # implies no clusters exist
        pass
    
    # map from integers to directions up, down, left, right
    NSEW = {0:np.array([0,1]),
            1:np.array([0,-1]),
            2:np.array([-1,0]),
            3:np.array([1,0])}
    
    # get movers that aren't found in clusters
    nonmovers = set(zip(maskX, maskY))
    #movers = list(filter(lambda x: x not in nonmovers, idx))
    movers = list(filter(lambda x: x not in nonmovers, idx))
    # select moves for each mover
    moves = np.array(list(map(lambda x: NSEW[x], np.random.randint(0,high=4,size=len(movers)))))
    # generate the next positions for the walkers to move with periodic boundaries
    next_positions = list(zip(*np.mod(movers + moves,mat.shape)))
    M[next_positions] = 1
    
    return M

def plot_matrix(A, name):
    X,Y = zip(*np.array(np.where(A>0)).T.tolist())
    plt.scatter(X,Y)
    plt.xticks([]); plt.yticks([])
    plt.savefig(name)
    plt.clf()

    return name

if __name__ == "__main__":
    args = parse_args().parse_args()
    
    L = len(str(args.generations))

    fmt = "\r[{:%d" % L + "d}]"
    outfile_str = os.path.join(args.frame_directory, "GifPhoto__{:05d}.png") 
    os.makedirs(args.frame_directory, exist_ok=True)
    M = np.zeros((args.dimensions,args.dimensions))
    spawn_walkers(M, N=args.n_walkers)
    filenames = []

    for g in range(args.generations):
        print(fmt.format(g),end="",flush=True)
        M = move_walkers(M)
        filenames.append(plot_matrix(M,outfile_str.format(g)))
    
    
    with imageio.get_writer(args.outfile, mode="I") as W:
        for fname in filenames:
            img = imageio.imread(fname)
            W.append_data(img)
