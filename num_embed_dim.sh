#!/bin/sh 
set -e 
set -x

python train.py --conv_operator GCN \
    --num_layers 4 --hidden_channels 64

python train.py --conv_operator GCN \
    --num_layers 4 --hidden_channels 128 

python train.py --conv_operator GCN \
    --num_layers 4 --hidden_channels 512

python train.py --conv_operator GCN \
    --num_layers 4 --hidden_channels 1024
