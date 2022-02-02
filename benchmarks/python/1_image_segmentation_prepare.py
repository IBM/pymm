#!/usr/bin/python3
# coding: utf-8

import numpy as np
import skimage as img
from skimage import data, io, filters
import urllib.request, urllib.parse, urllib.error
import random
import simplejpeg
import pickle


'''
Author:  Daniel Waddington <daniel.waddington@ibm.com> 2021
Changed: Moshik Hershcovitch <moshikh@il.ibm.com> 2022
License: Apache, Version 2.0
'''

count = 150 
web_host = '192.168.0.11'

print ("loading index")
def load_index():
    '''
    Load filelist.txt index from remote server
    '''
    url = 'http://' + web_host + '/images/imagenet/filelist.txt'
    file = urllib.request.urlopen(url)
    index = []
    for line in file:
        index.append(line.decode("utf-8").rstrip('\n'))
    return index

index = load_index()


def random_load_image(index):
    '''
    Load a random image
    '''
    i = random.randint(0,len(index))
    url = 'http://' + web_host + '/images/imagenet/' + index[i+1]
    data = urllib.request.urlopen(url).read()
    image = simplejpeg.decode_jpeg(data)
    return image

def prepare_images(count):
    import time
    images = []
    tic = time.perf_counter()
    for i in range(count):
        print ("load image number %d out of %d " % (i+1, count))
        images.append(random_load_image(index))
    toc = time.perf_counter()    
    print(f"loading in {toc - tic:0.4f} seconds")
    return images   

print ("loading %d images " % count)
images = prepare_images(count)

filename = "data/1_images.pickled"
print ("save images to " + filename)
pickle.dump(images, open(filename, "wb"))
