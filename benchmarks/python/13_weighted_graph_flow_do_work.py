#!/bin/python3
# coding: utf-8

'''
Author: Moshik Hershcovitch <moshikh@il.ibm.com> 2022
License: Apache, Version 2.0
'''

import networkx as nx
import pandas as pd
from networkx.algorithms import find_cycle
import time

rounds = 30 

# The time depands on the graph that was loading in the preparation
filename = 'data/13_digraph.pickled'
t0 = time.time()
G =  nx.read_gpickle(filename)
t = time.time() - t0
print ("Loading graph from file "+ filename +  " in %0.2fsec" % t)
print ("graph node(%d), edges(%d)" % (G.number_of_nodes(), G.number_of_edges()))



# Flow algorithm 

def do_flow(rounds):
    n = list(nx.nodes(G))
    i = 0
    w = 0
    t0 = time.time()
    for j in range(len(n)-2):
        if (w == rounds):
            break
        t = time.time() - t0    
        print ("Start calc maximum flows [%d/%d] , time %.02f" % (w, rounds, t))
        w = w + 1   
        try:
            nx.maximum_flow(G, n[j] ,n[j+1])
            i = i + 1
        except: 
            continue
    t = time.time() - t0    
    print ("Found %d maximum flows out of %d nodes , time %.02f" % (i, rounds, t))

do_flow(rounds) 

