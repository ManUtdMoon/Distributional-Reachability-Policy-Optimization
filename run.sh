#!/bin/sh

PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)
export PYTHONPATH=$PYTHONPATH:$PARENT_DIR

# quadrotor cartpole-move
for i in 1 49283 64578 219803 4354
do
    CUDA_VISIBLE_DEVICES=1 python main.py -c config/cartpole-move-csc.json \
        -s seed $i
done