#!/usr/bin/bash

DATAROOT=./data

mkdir $DATAROOT
cd $DATAROOT

# ImageNet-R
#wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
tar xvf imagenet-r.tar

# ImageNet-A
#wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
tar xvf imagenet-a.tar

