#!/usr/bin/python3
import pymm
import numpy as np
import skimage as img
from skimage import data, io, filters
import urllib.request, urllib.parse, urllib.error
import random

def display(image):
    io.imshow(image)
    io.show()
    
def perform_segmentation(image):
    import numpy as np
    from skimage.filters import sobel
    from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
    from skimage.util import img_as_float
    from skimage.segmentation import mark_boundaries

    segments = slic(image, n_segments=20, compactness=10, sigma=1, start_label=1)
    return mark_boundaries(image, segments)


#-- main --
shelf = pymm.shelf('imageExample',size_mb=32,backend='hstore-cc',pmem_path='/mnt/pmem0')

# load image to persistent shelf
#
if 'testImage' not in shelf.items:
    print('Loading test image from file...')
    shelf.testImage = data.coins()

# create segmented image on shelf
#
if 'segmentedTestImage' not in shelf.items:    
    shelf.segmentedTestImage = perform_segmentation(shelf.testImage)

# display segmented image
#
display(shelf.segmentedTestImage)

