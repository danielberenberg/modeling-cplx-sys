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
    parser.add_argument("-d","-dims",
                        dest="dimensions",
                        type=int,
                        help="dimension of the square matrix to conduct simulation",
                        default=100)

    parser.add_argument("-g","--gens",
                        dest="generations",
                        type=int,
                        help="number of generations",
                        default=100)

    parser.add_argument("-n","--nb-walkers",
                        dest="n_walkers",
                        type=int,
                        help="number of walkers in spawned at first in the system",
                        default=100)

    parser.add_argument("-o","--gif-name",
                        dest="outfile",
                        type=str,
                        default="DLA.gif",
                        help="name of the final gif")

    parser.add_argument("-f","--frame-dir",
                        dest="frame_directory",
                        type=str,
                        default="out",
                        help="intermediary frame directory")

    parser.add_argument("-p",
                        dest="spawn_prob",
                        type=float,
                        default=1e-3,
                        help="probability of new walkers spawning; if it is activated"
                            " a number between 1 and nb-walkers is sampled uniformly;"
                            " that many walkers are spawned")
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
    dimx, dimy = mat.shape
    origin = [[dimx/2, dimy/2],
              [dimx/2 + 1, dimy/2],
              [dimx/2 - 1, dimy/2],
              [dimx/2, dimy/2 + 1],
              [dimx/2, dimy/2 - 1],
              [dimx/2+1,dimy/2+1],
              [dimx/2-1,dimy/2+1],
              [dimx/2-1,dimy/2-1],
              [dimx/2+1,dimy/2-1]]
    factor = 2
    # label clusters with skimage magic
    walkers_lbld = measure.label(mat, connectivity=factor)
    # extract indexes by returning all indices that had a one
    idx = [np.array(np.where(walkers_lbld == label)).T.tolist() for label in np.unique(walkers_lbld) if label]
    indices = list(map(lambda x: tuple(i for i in x), reduce(lambda x1,x2: x1+x2, idx)))
    M = np.zeros(mat.shape)  # new matrix
    maskX, maskY = [], [] 
    try:  # test if there are clusters > length 2, concatenate each such cluster, split into 2 sep. arrays
        collisions = filter(lambda x: len(x) > 1 and any(o in x for o in origin), idx)
        maskX, maskY = zip(*reduce(lambda x1,x2: x1 + x2, collisions))
        M[maskX, maskY] = 1
    except TypeError:  # implies no clusters exist
        pass
    
    # map from integers to directions up, down, left, right
    NSEW = {0:factor*np.array([0,1]),
            1:factor*np.array([0,-1]),
            2:factor*np.array([-1,0]),
            3:factor*np.array([1,0])}
    
    # get movers that aren't found in clusters
    nonmovers = set(zip(maskX, maskY))
    movers = list(filter(lambda x: x not in nonmovers, indices))
    if not movers:
        return M
    movers = np.array(movers)
    # select moves for each mover
    moves = np.array(list(map(lambda x: NSEW[x], np.random.randint(0,high=4,size=len(movers)))))
    # generate the next positions for the walkers to move with periodic boundaries
    next_positions = list(zip(*np.mod(movers + moves,mat.shape)))
    M[next_positions] = 1
    return M, len(indices) - len(nonmovers)

def plot_matrix(A, name):
    kwargs = {"color":"black","marker":"s","facecolors":"white","edgecolors":"black","s":2}
    X1,Y1 = zip(*np.array(np.where(A == 1)).T.tolist())
    dimx, dimy = A.shape
    origin = [dimx/2, dimy/2]
    dim = A.shape[0]
    plt.xlim(0,dim)
    plt.ylim(0,dim)
    plt.scatter(X1,Y1, **kwargs)
    plt.scatter(*origin,**kwargs)
    num = int(name.split("__")[-1].split(".")[0])
    plt.text(1,1,num)
    plt.xticks([]); plt.yticks([])
    plt.savefig(name)
    plt.clf()

    return name

def flip(p):
    """
    flip a coin and return whether it came up heads or tails
    args:
        :p (float) - prob of heads
    returns:
        (bool) - H or T
    """
    return random.random() < p

if __name__ == "__main__":
    args = parse_args().parse_args()
    L = len(str(args.generations))

    fmt = "\r[{:%d" % L + "d}]"
    outfile_str = os.path.join(args.frame_directory, "GifPhoto__{:05d}.png") 
    os.makedirs(args.frame_directory, exist_ok=True)
    M = np.zeros((args.dimensions,args.dimensions))
    filenames = []
    spawn_walkers(M, args.n_walkers)
    for g in range(args.generations):

        if flip(args.spawn_prob):
            nspawn = np.random.randint(1,args.n_walkers)
            spawn_walkers(M, nspawn)

        print(fmt.format(g),end="",flush=True)
        M, nviable = move_walkers(M)
        filenames.append(plot_matrix(M,outfile_str.format(g+1)))
     
    print("\r" + 80*" " + "\rwriting {}....".format(args.outfile),end="")
    with imageio.get_writer(args.outfile, mode="I") as W:
        for fname in filenames:
            img = imageio.imread(fname)
            W.append_data(img)
    print("DONE")
