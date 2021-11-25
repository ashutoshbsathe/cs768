#!/bin/sh 
set -e 
set -x

python train.py --conv_operator GAT \
    --num_layers 4 --num_attention_heads 2 

python train.py --conv_operator GAT \
    --num_layers 4 --num_attention_heads 4

python train.py --conv_operator GAT \
    --num_layers 4 --num_attention_heads 8

python train.py --conv_operator GAT \
    --num_layers 4 --num_attention_heads 16 
