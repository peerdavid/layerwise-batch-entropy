#!/bin/bash

python3 train.py --norm="BatchNorm"
python3 train.py --norm="LayerNorm"
python3 train.py --norm="WeightNorm"
python3 train.py --norm="SELU"
