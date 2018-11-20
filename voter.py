"""
implements the voter model on an input network
"""
import csv
import argparse
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='voter model')
    parser.add_argument('network',
                        help='input undirected network, expected as csv where the first '+
                        'two element of each row symbolize the endpoints of an edge')
    return parser

def read_edgelist(filename,**kwargs):
    """
    reads an edgelist, returns an undirected network
    """
    G = nx.Graph()
    with open(filename,'r') as csvfile:
        reader = csv.reader(csvfile, **kwargs)
        for row in reader:
            src, targ = row[:2]
            if src.lower() == 'source' and targ.lower() == 'target':
                continue
            G.add_edge(src,targ)
    return G

def voter_model(network, T=1000):
    network = sorted(nx.connected_component_subgraphs(network), key=len, reverse=True)[0]
    T = range(T)
    blue_share = []
    for t in T:
        shadow = network.copy()
        for u in shadow:
            neighbors = shadow.neighbors(u)
            colors = [network.node[n]['color'] for n in neighbors]
            adopted = random.choice(colors)
            network.node[u]['color'] = adopted
        colors = list(nx.get_node_attributes(network, 'color').values())
        blue = list(filter(lambda c:c == 'blue', colors))
        blue_share.append(len(blue)/len(colors))
    return blue_share

def initial_coloring(net, pblue=0.5):
    nodes        = net.nodes()
    number_blue  = int(round(pblue * len(nodes)))
    blue_nodes   = set(random.sample(nodes, k=number_blue))
    node_attrs = dict()
    for u in nodes:
        if u in blue_nodes:
            node_attrs[u] = 'blue'
        else:
            node_attrs[u] = 'red'
    nx.set_node_attributes(net,name='color',values=node_attrs)
    return net

if __name__ == '__main__':
    args = parse_args().parse_args()
    net  = read_edgelist(args.network)
    
    RUNS = 10
    BLUE_PROPS = np.arange(0.05,0.99,0.05)
    mx_time = 200

    blueprop2proptimeline = dict() # trend of consensus prop
    blueprop2consensus = dict()    # number of times reached consensus
    blueprop2consensus_t = dict()  # average time consensus reached for each proportion

    wipe = '\r' + " "*80 + '\r'
    for prop in BLUE_PROPS:
        total = np.zeros(mx_time)
        for run in range(RUNS):
            print(wipe + f'run={run}, blue prop={prop:0.2f}',end='',flush=True)
            net = initial_coloring(net, pblue=prop)
            timeline = voter_model(net, T=mx_time) 
            consensus = timeline[-1] == 1.
            try:
                blueprop2consensus[prop] += consensus 
            except KeyError:
                blueprop2consensus[prop] = int(consensus)
            if consensus:
                try:
                    blueprop2consensus_t[prop] += timeline.index(1.) 
                except KeyError:
                    blueprop2consensus_t[prop] = timeline.index(1.)
            timelines = np.array(timeline)
            total += timeline
        try:
            blueprop2consensus_t[prop]  = blueprop2consensus_t[prop]/RUNS
        except KeyError:
            blueprop2consensus_t[prop]  = np.inf
        blueprop2consensus[prop]    = blueprop2consensus[prop]/RUNS
        blueprop2proptimeline[prop] = total/RUNS
        print('...DONE')
    
    print(f'Number of nodes: {len(net)}')
    print(f'Number of edges: {len(net.edges())}')
    avg_degree = np.mean(list(dict(net.degree()).values()))
    print(f'<k>: {avg_degree}')
    # plotting
    fig = plt.figure(figsize=(10, 10))
    ax3 = plt.subplot2grid((2,4), (0,0),rowspan=1,colspan=4)
    ax1 = plt.subplot2grid((2,4), (1,0),rowspan=1,colspan=2)
    ax2 = plt.subplot2grid((2,4), (1,2),rowspan=1,colspan=2)
    axarr = [ax1, ax2, ax3]
    # average time to reach consensus per proportion
    ts = [blueprop2consensus_t[prop] for prop in BLUE_PROPS]
    axarr[0].scatter(BLUE_PROPS, ts)
    axarr[0].set_xlabel('Proportion blue')
    axarr[0].set_ylabel('Time to reach consensus')
    axarr[0].set_xticklabels([round(prop,2) for prop in BLUE_PROPS[::2]])
    axarr[0].set_xticks(BLUE_PROPS[::2])
    # probability of consensus per proportion
    prob_cons = [blueprop2consensus[prop] for prop in BLUE_PROPS]
    axarr[1].set_xticklabels([round(prop,2) for prop in BLUE_PROPS[::2]])
    axarr[1].set_xticks(BLUE_PROPS[::2])
    axarr[1].bar(BLUE_PROPS, prob_cons, width=0.025)
    axarr[1].set_xlabel('Proportion blue')
    axarr[1].set_ylabel('Probability of consensus')
    # trendlines for consensus to reach
    for prop in BLUE_PROPS:
        axarr[2].plot(blueprop2proptimeline[prop], label=prop)
    axarr[2].set_xlabel('Time')
    axarr[2].set_ylabel('Proportion blue')
    plt.tight_layout()
    plt.savefig('voter_model_plot.pdf')




