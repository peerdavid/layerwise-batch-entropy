#!/bin/bash


for depth in 30 15
do
    python3 create_loss_surface.py --no-cuda --resolution 50 --depth $depth --steps 2000 --lbe_beta=0.5 --lbe_alpha=0.5
    python3 create_loss_surface.py --no-cuda --resolution 50 --depth $depth --steps 2000 --lbe_beta=0.0 --lbe_alpha=0.0
done
