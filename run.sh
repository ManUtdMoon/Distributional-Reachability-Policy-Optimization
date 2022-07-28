#!/bin/sh

PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)
export PYTHONPATH=$PYTHONPATH:$PARENT_DIR

# env: quadrotor cartpole-move safetygym-car safetygym-point
python main.py -c config/safetygym-car.json