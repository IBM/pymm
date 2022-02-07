#!/usr/bin/env python
# coding: utf-8

import networkx as nx
import time



# The time depands on the graph that was loading in the preparation
filename = 'data/10_2_graph.pickled'
t0 = time.time()
G =  nx.read_gpickle(filename)
t = time.time() - t0
print ("Loading graph from file "+ filename +  " tool %0.2fsec" % t)
print ("graph node(%d), edges(%d)" % (G.number_of_nodes(), G.number_of_edges()))



# Compute number of triangels
import time
def do_triangels(G):
    t0 = time.time()    
    T_dict = nx.triangles(G) 
    T = len(T_dict.keys())
    t = time.time() - t0   
    print ("Number of triangels in the graph are %d , time %.02f" % (T, t))

do_triangels(G)  


