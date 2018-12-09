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

    parser.add_argument("-c","--num-candidates",
                        dest="num_candidates",
                        type=intgt0,
                        help="the number of candidates per election",
                        default=10)

    parser.add_argument("-t","--opinion-transparency",
                        dest="mask_size",
                        help="the number of exposed opinions per candidate, a value in (0, d], "+
                             "where d is the opinion vector size "+
                             "spe$cifying `all` or a value greater than the number of opinions "+
                             "will result in the mask being length d",
                        default='all')
    return parser

class Voter:
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
            self._opinions = np.random.uniform(TOTAL_DISAGREE, TOTAL_AGREE, n_opinions)
        else:
            self._opinions = np.array(opinion_vector)

        if voter_id is None:
            self._id = Voter.ID
            Voter.ID += 1
        else:
            self._id = voter_id

    @property
    def num_opinions(self):
        return len(self.opinions)
    
    @property
    def opinions(self):
        return self._opinions
    
    @property
    def id(self):
        return self._id

    def to_list(self):
        return [self.voter_id] + [op for op in self.opinions]
    
class Candidate(Voter):
    
    def __init__(self, voter, transparency):
        """
        Initialize a Candidate, one such individual that has all the qualities
        of a Voter with the added attribute of a set of exposed opinions
        
        args:
            :transparency (int) - the number of opinions exposed by the mask
        """
        super(Candidate, self).__init__(n_opinions=voter.num_opinions,
                                        opinion_vector=voter.opinions,
                                        voter_id=voter.id)
        # generate a mask of size `transparency`
        idxs = list(range(self.num_opinions))   # the indices to expose/not expose
        exposed_idx = np.random.choice(idxs, transparency)  # choose the exposed ops

        # generate the opinion mask
        self._opinion_mask = [x in exposed_idx for x in idxs]
    
    @property
    def exposure(self):
        return self._opinion_mask

    @property
    def views(self):
        return self.opinions[self._opinion_mask]
    
def _get_voters(output_dir, n_voters, n_opinions, population_num):
    """
    generates and caches voting individuals with some number of opinions
    - the file will be cached to:
        output_dir/VotingPopulation__N_{n_voters}__D_{n_opinions}.csv
    - if the output file specified already exists, this file will be opened
      and its contents returned as a set of individuals

    args:
        :output_dir (str)  - directory path to the cache the populatin file
        :n_voters (int)    - number of voters to create
        :n_opinions (str)  - number of opinions per voter
        :population_num    - id of the population
    returns:
        :(list) - the list of voters, realized as Individual objects
    """
    cache_file = os.path.join(output_dir, f"VotingPopulation{population_num:02d}__N_{n_voters}__D_{n_opinions}.csv")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(cache_file):
        pop_df = pd.read_csv(cache_file)
        voter_rows = pop_df.values.tolist()
        voters = []
        for voter_data in voter_rows:
            v_id        = voter_data[0]
            opinion_vec = voter_data[1:]
            voters.append(Voter(voter_id=v_id, opinion_vector=opinion_vec))
        return voters
    else:
        pop_df = pd.DataFrame(columns=["id"] + [f"opinion_{i}" for i in range(n_opinions)])
        voters = []
        for i in range(n_voters):
            voter = Voter(n_opinions=n_opinions)
            voters.append(voter)
            pop_df.loc[i] = voter.to_list()
        
        pop_df.to_csv(cache_file)
        return voters

def choose_candidates(population, transparency_level, n_candidates):
    """
    choose candidates from the voting pool
    args:
        :population (list of Voter) - the voting population
        :transparency_level (int)   - the number of exposed opinions
    returns:
        :a list of candidates with the given transparency lvl 
    """
    nominated = np.random.choice(population, n_candidates)
    return [Candidate(nom, transparency_level) for nom in nominated]

def population_stream(output_dir, n_voters, n_opinions, n_populations=100):
    """
    generate different populations of voters (size n_voters) with some number of opinions 
    args:
        :output_dir (str) - path to read/write populations from/to 
        :n_voters   (int) - number of voting individuals in the system
        :n_opinions (int) - number of opinions per indiv.
        :n_populations (int) - the number of populations to stream out
    yields:
        :a generator of these voter populations
    """
    for pop in range(n_populations):
        yield _get_voters(output_dir, n_voters, n_opinions, pop)
    
if __name__ == "__main__":
    args = parse_args().parse_args()
    try:
        args.mask_size = int(args.mask_size)
        if args.mask_size > args.vector_size:
            print(f"[!!] Detected opinion transparency value > number of opinions, setting them equal")
            args.mask_size = args.vector_size
    except ValueError:
        if isinstance(args.mask_size, str) and args.mask_size == 'all':
            args.mask_size = args.vector_size
        else:
            sys.exit(f"[!!] unrecognized args: -t/--opinion-transparency:{args.mask_size}")
