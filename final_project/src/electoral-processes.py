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
import sys, os
import argparse
import numpy as np
import pandas as pd
from collections import Counter

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
                        help="the type of voting scheme to test; "+
                             "voting scheme will test for tolerance values between 1 and num_opinons",
                        choices=["ranked","approval"],
                        nargs=2)

    parser.add_argument("-c","--num-candidates",
                        dest="num_candidates",
                        type=intgt0,
                        help="the number of candidates per election",
                        default=10)

    parser.add_argument("-t","--opinion-transparency",
                        dest="mask_size",
                        help="the number of exposed opinions per candidate, a value in (0, d], "+
                             "where d is the opinion vector size "+
                             "specifying `all` or a value greater than the number of opinions "+
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

    def agreeability(self, candidate):
        """
        return the amount of agreeability a candidate's views have with this voter

        agreeability values close to 0 imply that this voter's opinions align with the candidate's
        larger agreeability values imply that this voter's opinions and the opinions of the candidate
        deviate

        args:
            :candidate (Candidate)
        returns:
            :the mean absolute error between the candidate's (exposed) opinion vector 
             and the voter's own opinion
        """
        relevant_opinions = self.opinions[candidate.exposure]
        return sum(abs(relevant_opinions - candidate.views))/candidate.transparency

    def happiness(self, candidate):
        """
        return how happy a voter is with a candidate by computing the 
        mean absolute error between their opinion vectors
        """
        return sum(abs(self.opinions - candidate.opinions))/len(self.opinions)

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
        exposed_idx = np.random.choice(idxs, transparency, replace=False)  # choose the exposed ops

        # generate the opinion mask
        self._opinion_mask = [x in exposed_idx for x in idxs]
    
    @property
    def exposure(self):
        return self._opinion_mask

    @property
    def transparency(self):
        return len(self.views)

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

def ranked_choice_voting(population, nominees, rank_size=1):
    """
    Ranked choice voting scheme;

    Each voter in the population lists their ideal candidates from 1 to rank_size 
    (the `rank_size` least dissimilar candidates from their own position)

    After ranking has taken place, the ballots are tallied in the following way:

        1. measure the number of voters for each candidate that ran
        2. If a candidate's vote count ≥ 50%, that candidate wins and the algorithm is completed.
        3. Else, kick the bottom candidate out of the race and redistribute their
           votes to the remaining candidates. Go to 1.

    In this context, 'redistribute' means look at the rank sheets for each voter that
    has a booted candidate and allocate that voter's vote to the next candidate on the rank sheet.

    Note that a rank_size = 1 ⇒  voting scheme analogous to classic, US-style voting (1 vote per voter) 

    args:
        :population (list of Voter) - the voting population
        :nominees (list of Candidate) - the number of candidates running in the election
        :rank_size (int) - the number of 
    returns:
        :the candidate that won
    """

    # casting votes
    all_ballots = []
    for voter in population:
        all_ballots.append(rank_sheet(voter, nominees, rank_size=rank_size))
    
    return ranked_choice_tally_votes(all_ballots)

def ranked_choice_tally_votes(ballot_box):
    votes = [ballot[0].id for ballot in ballot_box]
    candidate2ballot = Counter(votes)
    candidate2prop = calculate_shares(candidate2ballot)
    max_prop = max(candidate2prop.values())

    if max_prop >= 0.51:
        max_candidate = list(filter(lambda c: candidate2prop[c] == max_prop, candidate2prop))[0]
        return max_candidate
    else:
        min_prop = min(candidate2prop.values())
        min_candidate = list(filter(lambda c: candidate2prop[c] == min_prop, candidate2prop))[0]
        for ballot in ballot_box:
            if ballot[0].id == min_candidate.id:
                ballot[0].pop(0)
        return ranked_choice_tally_votes(ballot_box)
        
def calculate_shares(candidate2votes):
    total_votes     = sum(candidate2votes.values()) 
    candidate2share = {c:candidate2votes[c]/total_votes for c in candidate2votes}

    return candidate2share

def rank_sheet(voter, candidate_roster, rank_size=1):
    """
    return the ballot sheet for a voter in ranked choice method
    args:
        :voter (Voter)
        :candidate_roster (list of Candidates)
        :rank_size (int) - number of candidates to write on the ballot sheet
    """
    return sorted(candidate_roster, 
                  key=lambda candidate:voter.agreeability(candidate))[:rank_size]

def approval_voting(population, num_approved=1):
    pass
    
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
