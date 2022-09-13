#!/bin/sh

PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)
export PYTHONPATH=$PYTHONPATH:$PARENT_DIR

for i in 1 43567 789 49283 7346588
do
    python main.py -c config/safetygym-point.json \
        -s seed $i

    python main.py -c config/safetygym-point.json \
        -s alg_cfg.sac_cfg.update_violation_cost True \
        -s seed $i
done