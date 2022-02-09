#!/usr/bin/python3
# coding: utf-8

'''
Author:  Daniel Waddington <daniel.waddington@ibm.com> 2021
Changed: Moshik Hershcovitch <moshikh@il.ibm.com> 2022
License: Apache, Version 2.0
'''

# this is a very simple word vectorization implementation
import numpy as np
import urllib.request, urllib.parse, urllib.error
import bs4 as bs
import time
import pickle

count = 10000 


# one-time
import nltk
nltk.download('punkt')
nltk.download('stopwords')

web_host = 'bio3'

def load_index():
    '''
    Load filelist.txt index from remote server
    '''
    url = 'http://' + web_host + '/text/enron/filelist.txt'
    file = urllib.request.urlopen(url)
    index = []
    for line in file:
        index.append(line.decode("utf-8").rstrip('\n'))
    return index
print ("loading index")
index = load_index()


def load_file(index, pos):
    '''
    Load a random image
    '''
    url = 'http://' + web_host + '/text/enron/' + index[pos]
    return urllib.request.urlopen(url).read()




def load_data(count):
    file_data=[]
    tic = time.perf_counter()
    for i in range(count):
        print ("load file number %d out of %d " % (i+1, count))
        file_data.append(load_file(index, i))
    toc = time.perf_counter()      
    print(f"loading took {toc - tic:0.4f} seconds")    
    return file_data    


print ("loading %d files " % count)
file_data  = load_data(count)
filename = "data/2_files.pickled"
print ("save files to " + filename)
pickle.dump(file_data, open(filename, "wb"))



