# Non-Volatile Memory Accelerated Posterior Estimation
## Abstract

Bayesian inference allows machine learning models to express uncertainty. Current machine learning models use only a single learnable parameter combination when making predictions, and as a result are highly overconfident when their predictions are wrong. To use more learnable parameter combinations efficiently, these samples must be drawn from the posterior distribution. Unfortunately computing the posterior directly is infeasible, so often researchers approximate it with a well known distribution such as a Gaussian. In this paper, we show that through the use of high-capacity persistent storage, models whose posterior distribution was too big to approximate are now feasible, leading to improved predictions in downstream tasks.


## Link to the paper:

## Bibtex

# How to run: 
You can find all the details in the experiments folder

# Examples:
There are 4 settings of experiments:
## DRAM + Memory_mode
Regular implementation, when the data is beyond DRAM then Optane-PM 
accessible through DRAM cache misses --- The processing is on the DRAM data.
Example:
``` python
expirenants>  python3 mnist_dram.py  model_4fc -p /mnt/nvme0/models/  -c 1 -e 75  --results_filepath mnist_dram_mm_model_4fc_e75.gpu.csv
```

## Optane-PM - persistent mode
We use PyMM library in FS-DAX or Dev-DAX mode which offers cache-line granularity
with direct load/store access
``` python
expirenants> python3 mnist_pymm.py  model_4fc -p /mnt/nvme0/models/  -f /mnt/pmem0 -c 1 -s 520000 -e 75 --results_filepath mnist_pymm_dax_model_4fc_e75.gpu.csv
expirenants> python3 mnist_pymm.py  model_4fc -p /mnt/nvme0/models/  -f /dev/dax0.0 -c 1 -s 520000 -e 75 --results_filepath mnist_pymm_dax_model_4fc_e75.gpu.csv
```



## MMAP
We use the python class numpy.memmap for this implementation
Example:
``` python
expirenants>  python3 mnist_mmap.py  model_4fc -p /mnt/nvme0/models/  -c 1  --posterior_path /mnt/nvme0/mmap/  -e 75 --results_filepath mnist_mmap_model_4fc_e75.gpu.csv
```




# Helpful commands for creating different configurations:

## Optane-PM - App-Direct - fsdax
``` bash
sudo ndctl list
sudo ndctl destroy-namespace namespace0.0 -f
sudo mkdir /mnt/pmem0
sudo ndctl create-namespace -m fsdax -e namespace0.0 --align 2M --size 811748818944 --force
sudo mkfs -t ext4 /dev/pmem0
sudo mount -o dax /dev/pmem0 /mnt/pmem0
sudo chmod go+rw /mnt/pmem0
mkdir /mnt/pmem0/mystuff
```


## Optane-PM - App-Direct mode - dev-dax
``` bash
sudo ndctl list
sudo ndctl destroy-namespace namespace0.0 -f
sudo ndctl create-namespace -m devdax -e namespace1.0 --size 811748818944  --align 2M --force
sudo chmod go+rw /dev/dax0.0
```



## Optane-PM - Memory mode
``` bash
sudo ipmctl create -goal MemoryMode=100
sudo ipmctl show -memoryresources
sudo reboot
```



## Mount NVMe0
``` bash
sudo mount /dev/nvme0n1 /mnt/nvme0/
```




