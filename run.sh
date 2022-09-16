#!/bin/sh

PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)
export PYTHONPATH=$PYTHONPATH:$PARENT_DIR

# quadrotor cartpole-move
for i in 1 43567 7346588 789 49283
do
    python main.py -c config/safetygym-car.json \
        -s seed $i \
        -s alg_cfg.sac_cfg.enable_csc True
done
