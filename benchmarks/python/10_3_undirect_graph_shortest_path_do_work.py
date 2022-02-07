#!/usr/bin/env python
# coding: utf-8

'''
Author: Moshik Hershcovitch <moshikh@il.ibm.com> 2022
License: Apache, Version 2.0
'''
import networkx as nx
import time

# The time depands on the graph that was loading in the preparation
filename = 'data/10_3_graph.pickled'
t0 = time.time()
G =  nx.read_gpickle(filename)
t = time.time() - t0
print ("Loading graph from file "+ filename +  " took %0.2fsec" % t)
print ("graph node(%d), edges(%d)" % (G.number_of_nodes(), G.number_of_edges()))

# Shortest path
import time
def do_shortest_path():
    n = nx.nodes(G)
    i = 0 
    t0 = time.time()
    for j in n:
        i = i + 1
        nx.shortest_path_length(G, j)
    t = time.time() - t0            
    print ("Found the shortest path to all two consecutive nodes (according to their index in the graph): rounds %u, time %.02f" % (i, t))

do_shortest_path()   


