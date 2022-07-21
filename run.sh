#!/bin/sh

PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)
export PYTHONPATH=$PYTHONPATH:$PARENT_DIR

python main.py -c config/safetygym-point.json  # quadrotor cartpole-move