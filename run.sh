#!/bin/sh

PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)
export PYTHONPATH=$PYTHONPATH:$PARENT_DIR

# env: safetygym-car safetygym-point
for i in 1 43567 49283 789 7346588
do
    python main.py -c config/safetygym-car.json \
        -s seed $i
done