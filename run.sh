#!/bin/sh

PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)
export PYTHONPATH=$PYTHONPATH:$PARENT_DIR

# env: safetygym-car safetygym-point
python main.py -c config/safetygym-car.json \
    -s seed 7346588

python main.py -c config/safetygym-car.json \
    -s seed 1

python main.py -c config/safetygym-car.json \
    -s seed 49283

python main.py -c config/safetygym-car.json \
    -s seed 789