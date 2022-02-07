# Python Micro Benchmarks

Python micro-benchmarks are various algorithms in Machine Learning, Graphs, Signal Processing, and more.  
They run for 30-60 sec, and most of them can be easily adjusted to run longer.

In most of the benchmarks, there are two python scripts, the first is a preparation phase that saves the data to a local directory, and the second is the do_work that runs the algorithm.


## Benchmarks

### 1. Segmentation [Image Processing]
/This benchmark performs image segmentation to create a new image.
The preparation phase downloads 150 (default) images and processes them in the running phase.
To change the running time:
- The first option is changing the number of images processed in the preparation file.
The default number of images is: 150
```
count = 150
```
- The second option is changing the algorithm that processes the image: in the do_work file, the default is processing the image using slic algorithm. You can add quickshift or run quickshift by itself.
```
is_slic = 1 # '1' for runing slic algorithm [default]
is_quickshift = 0 # '1' for running quickshift algorithm


### 2. Word vectorization [NLP]
This benchmark loads text corpus (set of documents).  Build bag-of-words and then build per-document feature vector. The benchmark uses sentiment analysis, text recognition, information retrieval.

To change the running time:
- Change the number of files that are processed. In the preparation file
The default number of images is: 10000
```
count = 10000
```

### 3. FFT [Signal Processing]
This benchmark removes noise from the audio file.
In the preparation phase, it downloads an audio file and adds noise. In the running phase, it removes the noise.


### 4. Feature selection [Machine Learning]
This benchmark selects the best features using OMP (Orthogonal Matching Pursuit) from a dataset.
For selecting the best features, it uses in:
- 4_1 - linear regression algorithm.
- 4_2 - logistic regression algorithm.

To change the running time:
- The first option is changing the number of features selected in the do_work file.
The default number of selected features are: 225
```
count = 225
```
- The second option is changing the dataset size. The preparation file is downloading different sizes of datasets. In the do_work you can load the size that you wish.
The default for 4_1  is 200MB.
The default for 4_2  is 200MB.


### 5. People recognition [Machine Learning]

 
### 6. Image decomposition [Linear Algebra]


### 7. Sorting [Basic Algorithm]
This benchmark sorts two ndarrays, one integer ndarray, and one random ndarray. It uses two algorithms, quicksort, and mergesort, to sort each ndarray.
In the preparation phase, it's random the two ndarray, and in the do_work it sorts each ndarray two times (quicksort and mergesort).

To change the running time:
- You can change the size of the ndarray. The default size is 1GB
```
size = 1*GB
```

### 8. Matrix Rotation

### 9. Simple Directed graph (Cycle) [Graph]

In this benchmark, we count how many cycles are in a graph.
In the preparation phase, we generate a graph with the following parameters:
nodes: 25K, edges: 62K
In the do_work phase, the benchmark counts the number of cycles in this graph.

We use networkx repo to create the graph and use networkx implementation of the simple algorithms.

To change the running time:
- You can change the size of the graph. There are two parameters, the number of nodes and edge probability.
Here are the default values
```
nodes = 25*1000
edges_prob = 0.0001
```

### 10. Simple Undirected Graph (Coloring, Triangles, Shortest path) [Graph]

In this benchmark, we will run simple algorithms on undirected graphs.
In the preparation phase, we generate a graph.
In the do_work phase, the benchmark counts the algorithm.

We use networkx repo to create the Graph and use networkx implementation of the simple algorithms.

#### 10.1
Run greedy coloring algorithms with different basic strategies on the Graph:

The basic strategies are:
##### Basic strategies
- largest_first
- random_sequential
- smallest_last
- connected_sequential_bfs
- connected_sequential_dfs

To change the running time:
- You can change the size of the Graph. There are two parameters, the number of nodes and edge probability.
Here are the default values in 10_1
```
nodes = 200*1000
edges_prob = 0.0001
```

#### 10.2
Count how triangles in the Graph:

To change the running time:
- You can change the size of the Graph. There are two parameters, the number of nodes and edge probability.
Here are the default values in 10_2
```
nodes = 250*1000
edges_prob = 0.0001
```

#### 10.3
Calculate the shortest path to all two consecutive nodes (according to their index in the Graph):

To change the running time:
- You can change the size of the Graph. There are two parameters, the number of nodes and edge probability.
Here are the default values in 10_3
```
nodes = 5*1000
edges_prob = 0.001
```

### 11. Optimization [Signal Processing]

### 12. Biclustering [Data Mining]

### 13. Weighted Directed Graph (flow) [Graph]

To change the running time, you have two options:
Change the number of rounds of the flow algorithms running from different nodes.
In the do_work file, we have a variable for the number of rounds. The default value is:
```
rounds = 50
```

- You can change the size of the graph. There are two parameters, the number of nodes and edge probability.
Here are the default values in 10_3
```
nodes = 50*1000
edges_prob = 0.0001
```

### 14. Weighted Undirected Graph (spanning tree) [Graph]

### 15. K-Nearest-Neighbors [Machine Learning]

### 16. Clustering algorithm (kmeans) [Machine Learning] 	


