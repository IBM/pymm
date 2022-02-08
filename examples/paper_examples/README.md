# Code examples for published papers
1. [Non-Volatile Memory Accelerated Posterior Estimation - HPEC2021](https://github.com/IBM/mcas/wiki/List-of-MCAS-publications)


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




