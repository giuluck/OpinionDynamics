import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import coo_matrix


# datasets utility functions
def load_reddit():
    a, _, _, z, _, _ = loadmat('res/Reddit.mat')['Reddit'][0, 0]
    return a.toarray(), pd.DataFrame(z).groupby(0)[2].mean().values


def load_twitter():
    # we use -1 when loading the adjacency matrix since nodes are stored with id {1, ..., N}
    z = pd.read_csv('res/opinion_twitter.txt', sep='\t', header=None)
    a = pd.read_csv('res/edges_twitter.txt', sep='\t', header=None) - 1
    a = coo_matrix((np.ones(len(a)), zip(*a.values)))
    return a.toarray(), pd.DataFrame(z).groupby(0)[2].mean().values


DATASETS = {'reddit': load_reddit, 'twitter': load_twitter}

# user data entry
if len(sys.argv) == 1:
    datasets = list(DATASETS.keys())
else:
    datasets = []
    for i, d in enumerate(sys.argv[1:]):
        assert d in DATASETS, f'Unknown dataset "{d}", possible options are {list(DATASETS.keys())}'
        datasets.append(d)

for d in datasets:
    # retrieve data and remove isolated nodes
    adj, expressed = DATASETS[d]()
    isolated = [i for i, r in enumerate(adj) if r.sum() == 0]
    expressed = np.delete(expressed, isolated, 0)
    adj = np.delete(adj, isolated, 0)
    adj = np.delete(adj, isolated, 1)
    # compute innate opinions and normalize into [-1, 1] (convert expressed opinions into that range as well)
    matrix = np.diag(adj.sum(axis=0)) - adj + np.eye(len(adj))
    innate = matrix @ expressed
    innate = 2 * (innate - innate.min()) / (innate.max() - innate.min()) - 1
    expressed = 2 * expressed - 1
    # compute and print homophily statistics
    neighborhood_expressed = np.array([np.mean(expressed[neighbors != 0]) for neighbors in adj])
    neighborhood_innate = np.array([np.mean(innate[neighbors != 0]) for neighbors in adj])
    homophily_expressed = 1 - np.square((expressed - neighborhood_expressed) / 2)
    homophily_innate = 1 - np.square((innate - neighborhood_innate) / 2)
    print(d.upper())
    print(pd.DataFrame.from_dict({'expressed': homophily_expressed, 'innate': homophily_innate}).describe())
    print()
    # create and plot graph (invert the weight of edges between nodes with opinions of opposite signs to separate them)
    for i, row in enumerate(adj):
        for j, value in enumerate(row):
            adj[i][j] = value * np.sign(innate[i] * innate[j])
    graph = nx.Graph(adj)
    plt.figure(figsize=(16, 16))
    pos = nx.spring_layout(graph)
    edges = nx.draw_networkx_edges(graph, pos=pos, edge_color='#444', alpha=0.1)
    nodes = nx.draw_networkx_nodes(graph, pos=pos, node_color=innate, cmap=plt.get_cmap('seismic'))
    nodes.set_edgecolor('#444')
    plt.show()
