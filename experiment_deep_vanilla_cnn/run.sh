#!/bin/bash

python3 train.py --dataset="mnist" --name="MNIST LBE" --init_method="random" --lbe_alpha=2.0 --lbe_alpha_min=2.0 --lbe_beta=1e-1 --epochs=1000
