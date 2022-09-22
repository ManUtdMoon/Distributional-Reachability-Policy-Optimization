#!/bin/sh

PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)
export PYTHONPATH=$PYTHONPATH:$PARENT_DIR

# env: safetygym-car safetygym-point
for i in 86 46 51 90 72
do
    python main.py -c config/safetygym-car.json \
        -s seed $i \
        -s alg_cfg.eval_shield_type linear
        -s alg_cfg.sac_cfg.enable_switch False
done