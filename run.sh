#!/bin/sh

PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)
export PYTHONPATH=$PYTHONPATH:$PARENT_DIR

# quadrotor cartpole-move
python main.py -c config/quadrotor-mbpol.json
python main.py -c config/quadrotor-csc.json

python main.py -c config/quadrotor-mbpol.json \
    -s seed 7465
python main.py -c config/quadrotor-csc.json \
    -s seed 7465