#!/bin/bash


for depth in 30 15
do
    python3 train.py --depth=$depth --lbe_beta=0.0 --lbe_alpha=0.0
    python3 train.py --depth=$depth --lbe_beta=0.2 --lbe_alpha=0.5
done