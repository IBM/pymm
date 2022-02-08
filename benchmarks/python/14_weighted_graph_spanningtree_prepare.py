#!/usr/bin/python3
# coding: utf-8

'''
Author: Moshik Hershcovitch <moshikh@il.ibm.com> 2022
License: 3-clause BSD
'''

import networkx as nx
import pickle
# import graph
import random
import time

nodes = 300*1000
edges_prob = 0.0001
seed = 1

t0 = time.time() 
G = nx.fast_gnp_random_graph(nodes, edges_prob, seed=1, directed=False)
for (u, v) in G.edges():
    G.edges[u,v]['weight'] = random.randint(0,10000)
print ("Digraph (nodes, edges) = ({:d}, {:d}) with weights the time to generate this digraph is: {:0.2f}".format(G.number_of_nodes(), G.number_of_edges(), time.time() - t0))


filename = 'data/14_digraph.pickled'
print ("write to " + filename)
nx.write_gpickle(G, filename)
