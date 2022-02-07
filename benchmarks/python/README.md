# Python Micro Benchmarks

Python micro-benchmarks are various algorithms in the fields of Machine Learning, Graphs, Signal Processing, and more.  
They run for 30-60 sec, and most of them can be easily adjusted to run longer.

In most of the benchmarks, there are two python scripts, the first is a preparation phase that saves the data to a local directory, and the second is the do_work that runs the algorithm.


## Benchmarks

### 1. Segmentation [Image Processing]
This benchmark performs image segmentation to create a new image. 
In the preparation phase, it downloads 150 (default) images and processes them in the running phase.
To change the running time:
- The first option is changing the number of images that are processedi in the preparation file. 
The default number of images is: 150
```
count = 150
```
- The second option is changing the algorithm that processes the image: in the do_work file the default is processing the image using slic algorithm, you can add quickshift or run quickshift by itself. 
```
is_slic = 1 # '1' for runing slic algorithm [default]
is_quickshift = 0 # '1' for running quickshift algorithm
```


### 2. Word vectorization [NLP]
This benchmark loads text corpus (set of documents).  Build bag-of-words and then build per-document feature vector. The benchmark uses sentiment analysis, text recognition, information retrieval.

To change the running time:
- Change the number of files that are processed. In the preparation file
The default number of images is: 10000
```
count = 10000
```


### 3. FFT [Signal Processing]
This benchmark remove noise from audio file.
In the preparation phase, it downloads audio file and add noise, in the running phase it removes the noise.


### 4. Feature selection [Machine Learning] 
This benchmark select the best features using OMP (Orthogonal Matching Pursuit) from a dataset. 
For selecting the best featuers it use in: 
- 4_1 - linear regreation algorithm.
- 4_2 - logistic regreation algorithm.

To change the running time:
- The first option is changing the number of features selected in the do_work file.
The default number of selected features are: 225 
```
count = 225
```
- The second option is changing the dataset size. The preparetion file is downloading different size of datasets, in the do_work you can load the size that you wish. 
The default for 4_1  is 200MB.
The default for 4_2  is 200MB.



### 5. People recognition [Machine Learning]

 
### 6. Image decomposition [Linear Algebra]


### 7 Sorting [Basic Algorithm]
This benchmark sorts two ndarrays, one integer ndarray and one random ndarray. It uses two algorithms, quicksort and mergesort, to sort each ndarray.
In the preparation phase it is random the two ndarray and in the do_work it sort each ndarray two times (quicksort and mergesort).

To change the running time:
- You can change the size of the ndarray. The default size is 1GB 
```
size = 1*GB
```

### 8. Matrix Rotation

### 9. Simple Directed graph (Cycle) [Graph]
In this benchmark we count how many cycles are in a graph.
In the preparation phase we generate a graph with the following parameters:
nodes: 25K, edges: 62K
In the do_work phase the benchmark counts the number of cycles that are in this graph.
To change the running time:
- You can change the size of the graph, there are two parameters, number of nodes and edge probability.
This is the default values
```
nodes = 25*1000
edges_pob = 0.0001
```



### 10. Simple Undirected graph (Cloloring, Triangels, Sourtest path) [Graph]

### 11. Optimization [Signal Processing]

### 12. Biclustering [Data Mining]

### 13. Weighted Directed Graph (flow) [Graph]

### 14. Weighted Undirected Graph (spanning tree) [Graph]

### 15. K-Nearest-Neighbors [Machine Learning]

### 16. Clustering algorithm (kmeans) [Machine Learning] 	


