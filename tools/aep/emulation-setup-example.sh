#!/bin/bash
#
# This example will illustrate how to set up two regions of emulated persistent memory from DRAM
#
# add kernel params: e.g. memmap=2G!28G memmap=2G!30G
#
sudo insmod ~/mcas/build/dist/bin/mcasmod.ko
sudo insmod ~/mcas/build/dist/lib/modules/4.18.0-193.13.2.el8_2.x86_64/xpmem.ko

sudo ndctl create-namespace -e namespace0.0 -m devdax --align 2M --force
sudo ndctl create-namespace -e namespace1.0 -m devdax --align 2M --force

sudo chmod a+rwx /dev/dax*
