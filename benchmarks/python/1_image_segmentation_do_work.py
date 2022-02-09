#!/usr/bin/python3
# coding: utf-8

'''
Author:  Daniel Waddington <daniel.waddington@ibm.com> 2021
Changed: Moshik Hershcovitch <moshikh@il.ibm.com> 2022
License: Apache, Version 2.0
'''

import numpy as np
import skimage as img
from skimage import data, io, filters
import urllib.request, urllib.parse, urllib.error
import random
import simplejpeg
import pickle
import numpy as np
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries

is_slic = 1 # 1 for to run  slic
is_quickshift = 0 # 1 for to run  quickshift

filename = "data/1_images.pickled"
images  = pickle.load(open(filename, "rb"))
print ("The number of images are: " + str(len(images)))

def perform_segmentation_work_quickshift(image):
    segments = quickshift(image, kernel_size=16, max_dist=12, ratio=0.5)    
    return mark_boundaries(image, segments)

def perform_segmentation_work_slic(image):
    segments = slic(image, n_segments=20, compactness=10, sigma=1, start_label=1)
    return mark_boundaries(image, segments)





def do_work(count):
    import time
    tic = time.perf_counter()
    for i in np.arange(count):
        if (is_slic):
            perform_segmentation_work_slic(images[i])
        if (is_quickshift):    
            perform_segmentation_work_quickshift(images[i])
    toc = time.perf_counter()
    print(f"Execution in {toc - tic:0.4f} seconds")

count = len(images)
print ("perform_segmentation on " + str(count) + " images")
do_work(count)




