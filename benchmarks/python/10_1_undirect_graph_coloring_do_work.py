#!/usr/bin/env python
# coding: utf-8

import networkx as nx
import time


'''
Author: Moshik Hershcovitch <moshikh@il.ibm.com> 2022
License: Apache, Version 2.0
'''

# The time depands on the graph that was loading in the preparation
filename = 'data/10_1_graph.pickled'
t0 = time.time()
G =  nx.read_gpickle(filename)
t = time.time() - t0
print ("Loading graph from file "+ filename +  " in %0.2fsec" % t)
print ("graph node(%d), edges(%d)" % (G.number_of_nodes(), G.number_of_edges()))

#    color differ by at most 1.

# Basic startegies 
### largest_first
### random_sequential
### smallest_last
### independent_set --- too long
### connected_sequential_bfs
### connected_sequential_dfs


def do_coloring (strategy_type = "largest_first", equitable = False):
    t0 = time.time()
    nx.coloring.greedy_color(G, strategy_type )
    t = time.time() - t0
    print ("time for strategy type=%s cloloring is %0.2f " % (strategy_type,t))

do_coloring(strategy_type = "connected_sequential_bfs") 
do_coloring(strategy_type = "connected_sequential_dfs") 
do_coloring(strategy_type = "largest_first") 
do_coloring(strategy_type = "random_sequential") 
#do_coloring(strategy_type = "independent_set") 
do_coloring(strategy_type = "smallest_last") 


