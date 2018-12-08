"""
Complex Opinion in the Political Arena 
    - Measuring the interactions between campaign strategy, voting methodology, and
      election result happiness.

Nodes are considered to be people in a voting society with a d-length vector of opinions O.

Each singular opinion oᵢ∈ O is a value in [-1, 1] that represents how strongly 
the individual feels about issue i.
    i close to -1 implies the individual strongly opposes the issue,
    i close to  1 implies the individual strongly supports the issue, 
    and i close to 0 implies a neutral opinion.

Each oᵢ is sampled from a uniform distribution in [-1, 1]. 
"""
import argparse
import numpy as np
import pandas as pd
import sys, os

TOTAL_DISAGREE = -1
TOTAL_AGREE = 1

def intgt0(x):
    """verify input is an integer > 0"""
    x = int(x)
    if x > 0:
        return x
    raise ValueError

def parse_args():
    """sets up the arguments that can be passed into the script"""
    parser = argparse.ArgumentParser(description="Quantifying voter happiness")
    parser.add_argument("-N","--population-size",
                        dest="pop_size",
                        type=intgt0,
                        help="Number of individuals who will cast a vote",
                        default=1000)

    parser.add_argument("-d","--num-opinions",
                        dest="vector_size",
                        type=intgt0,
                        help="Number of opinions per individual",
                        default=10)


    parser.add_argument("-v","--voting-scheme",
                        dest="voting_scheme",
                        help="the type of voting scheme to test",
                        choices=["general","ranked","approval"],
                        default="ranked")

    parser.add_argument("-t","--opinion-transparency",
                        dest="mask_size",
                        help="the number of exposed opinions per candidate, a value in (0, d], "+
                             "where d is the opinion vector size "+
                             "spe$cifying `all` or a value greater than the number of opinions "+
                             "will result in the mask being length d",
                        default='all')
    return parser

class Individual:
    ID = 0

    def __init__(self, n_opinions=10, opinion_vector=None, voter_id=None):
        """
        Initialize a voting individual in the population.
        Creates an opinion vector and assigns an voter ID number
        args:
            :n_opinions (int) - size of the indiv.'s opinion vector
            :opinion_vector (array-like) - the vector of opinions (generated if None)
            :voter_id (int) - the id for this voter
        """
        if opinion_vector is None:
            self.opinions = np.random.uniform(TOTAL_DISAGREE, TOTAL_AGREE, n_opinions)
        else:
            self.opinions = opinion_vector

        if voter_id is None:
            self.voter_id = Individual.ID
            Individual.ID += 1
        else:
            self.voter_id = voter_id

    @property
    def num_opinions(self):
        return len(self.opinions)

    def to_list(self):
        return [self.voter_id] + [op for op in self.opinions]

def get_voters(output_dir, n_voters, n_opinions):
    """
    generates and caches voting individuals with some number of opinions
    - the file will be cached to:
        output_dir/VotingPopulation__N_{n_voters}__D_{n_opinions}.csv
    - if the output file specified already exists, this file will be opened
      and its contents returned as a set of individuals

    args:
        :output_file (str) - path to the cache file
        :n_voters (int)    - number of voters to create
        :n_opinions (str)  - number of opinions per voter
    returns:
        :(list) - the list of voters
    """
    cache_file = os.path.join(output_dir, f"VotingPopulation__N_{n_voters}__D_{n_opinions}.csv")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(cache_file):
        pop_df = pd.read_csv(cache_file)
        voter_rows = pop_df.values.tolist()
        voters = []
        for voter_data in voter_rows:
            v_id        = voter_data[0]
            opinion_vec = voter_data[1:]
            voters.append(Individual(voter_id=v_id, opinion_vector=opinion_vec))
        return voters
    else:
        pop_df = pd.DataFrame(columns=["id"] + [f"opinion_{i}" for i in range(n_opinions)])
        voters = []
        for i in range(n_voters):
            voter = Individual(n_opinions=n_opinions)
            voters.append(voter)
            pop_df.loc[i] = voter.to_list()
        
        pop_df.to_csv(cache_file)
        return voters

    
if __name__ == "__main__":
    args = parse_args().parse_args()
    try:
        args.mask_size = int(args.mask_size)
    except ValueError:
        if isinstance(args.mask_size, str) and args.mask_size == 'all':
            args.mask_size = args.vector_size
        else:
            sys.exit("[!!]")
    #elif args.mask_size < 1:
    #    sys.exit("[!!] Opinion transparency should be a value > 0")
    #if args.mask_size > args.vector_size or args.mask_size > :
    #    sys.exit("[!!] Mask size ")
