#!/bin/sh

PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)
export PYTHONPATH=$PYTHONPATH:$PARENT_DIR

# quadrotor cartpole-move
python main.py -c config/quadrotor-mbpol.json \
    -s seed 64578
python main.py -c config/quadrotor-csc.json \
    -s seed 64578

python main.py -c config/quadrotor-mbpol.json \
    -s seed 219803
python main.py -c config/quadrotor-csc.json \
    -s seed 219803

python main.py -c config/quadrotor-mbpol.json \
    -s seed 456
python main.py -c config/quadrotor-csc.json \
    -s seed 456