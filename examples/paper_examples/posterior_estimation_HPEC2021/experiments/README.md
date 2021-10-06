This is our experiments directory. There are two main scripts at the moment (am still working on memory mapping on DRAM script). These scripts are:

1) ./mnist_dram.py
2) ./mnist_pymm.py
3) ./mnist_mmap.py

These scripts will run a SWAG (SWA-Gaussian) experiment using either pymm or dram. Both scripts take a few optional arguments and only one required argument. The required argument is the model string (aka class name of the model in lower-case).

The posterior in these experiments is a function of the number of samples. The number of bytes occupied by the posterior follows this rule:

            Bytes(Posterior) = 12*n*(k+3) + 12

where n is the number of parameters in the model, and k is the number of samples that the posterior expects (following the original SWAG paper https://proceedings.neurips.cc/paper/2019/file/118921efba23fc329e6560b27861f0c2-Paper.pdf).

The following models are available (and are the required argument for each script to run):
1) model_2conv2fc. This model has 2 conv layers and 2 fc layers totaling 21,840 params
2) model_2fc. This model has 2 fc layers totaling 795,010 params
3) model_3fc. This model has 3 fc layers totaling 1,290,510 params
4) model_4fc. This model has 4 fc layers totaling 2,797,010 params

The default behavior for the scripts is to use all posterior samples (recorded after every minibatch as default behavior). For MNIST, there are 600 batches using the standard minibatch size of 100). Given the batch_size and the number of epochs, and the recording frequency (i.e. 1/num_minibatches_to_record_a_sample), we can compute the parameter k to be:

            k = 60000 * epochs * frequency / batch_size

Using default values, if we wanted to target more posterior memory sizes, you should need to train with more epochs than the defualt (-e=1).
These are the models we have:
model            
-----------------
model_2conv2fc   
model_2fc        
model_3fc        
model_4fc        


By defualt, results will be saved to ./results/mnist/{dram,pymm}.csv unless a path is specified otherwise. Please use the -h flag to see the arguments for the scripts.
By default, the training excute on gpu if you want to tain on cpu, please use -c=-1
