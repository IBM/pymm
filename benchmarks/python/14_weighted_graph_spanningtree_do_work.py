#!/bin/python3
import time
import os

'''
Author: Moshik Hershcovitch <moshikh@il.ibm.com> 2022
License: 3-clause BSD
'''

import networkx as nx
import pandas as pd
from networkx.algorithms import find_cycle
import time


# The time depands on the graph that was loading in the preparation
filename = 'data/14_digraph.pickled'
t0 = time.time()
G =  nx.read_gpickle(filename)
t = time.time() - t0
print ("Loading graph from file "+ filename +  " in %0.2fsec" % t)
print ("graph node(%d), edges(%d)" % (G.number_of_nodes(), G.number_of_edges()))


def do_spanning_tree():
    n = list(nx.nodes(G))
    i = 0
    w = 0
    t0 = time.time()
    nx.maximum_spanning_tree(G)
    t = time.time() - t0    
    print ("Found maximum flows: time %.02f" % (t))


do_spanning_tree() 

