#!/usr/bin/python3
# coding: utf-8

'''
Author: Moshik Hershcovitch <moshikh@il.ibm.com> 2022
License: 3-clause BSD
'''


import networkx as nx
import pandas as pd
from networkx.algorithms import find_cycle
import time
# import graph

# The time depands on the graph that was loading in the preparation
filename = 'data/9_digraph.pickled'
t0 = time.time()
G =  nx.read_gpickle(filename)
t = time.time() - t0
print ("Loading graph from file "+ filename +  " in %0.2fsec" % t)
print ("graph node(%d), edges(%d)" % (G.number_of_nodes(), G.number_of_edges()))

# how many nodes has cycle
def do_count_nodes_that_have_cycles(count):
    n = nx.nodes(G)
    i = 0
    w = 0
    t0 = time.time()
    for j in n:
        if (w == count):
            break
        w = w + 1   
        try:
            nx.find_cycle(G, j)
            i = i + 1
        except: 
            continue
    t = time.time() - t0    
    print ("Found %d nodes with cycles out of %d tested nodes , time %.02f" % (i, count, t))

do_count_nodes_that_have_cycles(len(nx.nodes(G))) 
