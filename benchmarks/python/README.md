# Python Micro Benchmarks

Python micro-benchmarks are various algorithms in the fields of Machine Learning, Graphs, Signal Processing, and more.  
They run for 30-60 sec, and most of them can be easily adjusted to run longer.

In most of the benchmarks, there are two python scripts, the first is a preparation phase that saves the data to a local directory, and the second is the do_work that runs the algorithm.


## Benchmarks

### 1: Segmentation [Image Processing]
This benchmark performs image segmentation to create a new image. 
In the preparation phase, it downloads 150 (default) images and processes them in the running phase.
To change the running time:
- Change the number of images that are processed. In the preparation file 
The default number of images is: 150
```
count = 150
```
- Change the algorithm that processes the image: in the do_work file the default is processing the image using slic algorithm, you can add quickshift or run quickshift by itself. 
```
is_slic = 1 # 1 for to run  slic
is_quickshift = 0 # 1 for to run  quickshift
```


### 2: Word vectorization [NLP]
This benchmark loads text corpus (set of documents).  Build bag-of-words and then build per-document feature vector. The benchmark uses sentiment analysis, text recognition, information retrieval.

To change the running time:
- Change the number of files that are processed. In the preparation file
The default number of images is: 10000
```
count = 10000
```







