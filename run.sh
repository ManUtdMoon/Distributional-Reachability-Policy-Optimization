#!/bin/sh

PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)
export PYTHONPATH=$PYTHONPATH:$PARENT_DIR

python main.py -c config/quadrotor-mbpol.json  # quadrotor cartpole-move