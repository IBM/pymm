#!/usr/bin/python3
# coding: utf-8
'''
Author: Moshik Hershcovitch <moshikh@il.ibm.com> 2022
License: 3-clause BSD
'''

import networkx as nx
import pickle
import time

nodes = 200*1000
edges_prob = 0.0001
seed = 1

t0 = time.time() 
G = nx.fast_gnp_random_graph(nodes, edges_prob, seed=1, directed=False)
print ("Undirected Graph (nodes, edges) = ({:d}, {:d}),  the time to generate this graph is: {:0.2f}".format(G.number_of_nodes(), G.number_of_edges(), time.time() - t0))

filename = 'data/10_1_graph.pickled'
nx.write_gpickle(G, filename)
