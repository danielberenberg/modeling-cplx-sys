import sys, os
import argparse
import numpy as np
from traffic_model_2 import traffic_model
from glob import glob

def parse_args():
    """
    Our main arguments to change:

        :time = total time to run sys for
        :b_rate = time delay for bicycles
        :flows = flow rates from each direction
        :bike2car = prob. that a veh entering
                   the system is a bicycle
        :dim = dimensions of the system grid

    """
    parser = argparse.ArgumentParser(description="investigating vehicle interactions")
    parser.add_argument("-x",type=int, help="x dimensions of the system",
                        default=100)

    parser.add_argument("-y",type=int,help="y dimension of the system",
                        default=100)

    parser.add_argument("-T",help="total time to run system for",
                        default=700, type=int)
    
    parser.add_argument("-pb",help="probability of a bike spawning",
                        default=0., type=float)

    parser.add_argument("--b-rate",dest="b_rate",help="time delay for bicycles",
                        default=3, type=int)

    parser.add_argument("--northflow","-n",dest="n",
                        help="probability of a vehicle flowing in from the north",
                        default=1., type=float)

    parser.add_argument("--southflow","-s",dest="s",
                        help="probability of a vehicle flowing in from the south",
                        default=1., type=float)

    parser.add_argument("--eastflow","-e",dest="e",
                        help="probability of a vehicle flowing in from the east",
                        default=1., type=float)

    parser.add_argument("--westflow","-w",dest="w",
                        help="probability of a vehicle flowing in from the west",
                        default=1., type=float)

    parser.add_argument("-o",dest="output_directory",
                        help="output directory, specifying `compute` will force the script to generate a gif with a summary filename",
                        default="compute",
                        type=exists)

    return parser

def exists(directory):
    """
    function is called to verify the output directory exists, if it doesn't make it.
    """
    directory = str(directory)
    if directory == "compute":
        return directory

    os.makedirs(directory, exist_ok=True)
    return directory

if __name__ == "__main__":
    args = parse_args().parse_args()
    output_directory = args.output_directory
    if args.output_directory == "compute":
        args.output_directory = f"figs/TrafficSimulation__pb_{args.pb}__nf_{args.n}__sf_{args.s}__ef_{args.e}__wf_{args.w}__T_{args.T}__br_{args.b_rate}"
        output_directory = exists(args.output_directory)

    flow_rates = {"N":args.n,
                  "S":args.s,
                  "E":args.e,
                  "W":args.w}

    # flow_rates: (dict) - probability of vehicles entering the system
    #                     from N,S,E, or W directions @ time t
    #allflowseq = True
    #if allflowseq:
    #    flowrate = 1.
    #    flows = {d:flowrate for d in set("NSEW")}
    #else:
    #    flows = {"N":0.95,
    #             "S":0.7,
    #             "E":0.7,
    #             "W":0.95}
    
    dim = 100        # dimensions of the system/grid
    
    # incoming traffic zones:|
    N_traffic = [dim//20, dim//8, dim//4, dim//3, dim//3+dim//10, dim//2,  dim - dim//4]
    W_traffic = [dim//10, dim//4, dim//3, 2*dim//3 - dim//4, dim//2, dim//2 + dim//6, dim//6, dim - dim//4, dim - dim//8]
    #W_traffic = [args.x//5]
    #N_traffic = [args.y//3]
    
    traffic_model(args.pb,args.T,args.b_rate,flow_rates,N_traffic,W_traffic,(dim,dim), output_directory)
