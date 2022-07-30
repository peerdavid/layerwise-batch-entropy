#!/bin/bash

python3 -m venv env

source env/bin/activate
pip3 install --upgrade pip

# If running on 3090 gpus
# pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r requirements.txt
