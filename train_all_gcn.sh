#!/bin/sh 
set -e 
set -x

python train.py --conv_operator GCN \
    --num_layers 2 

python train.py --conv_operator GCN \
    --num_layers 4 

python train.py --conv_operator GCN \
    --num_layers 8 

python train.py --conv_operator GCN \
    --num_layers 16 
