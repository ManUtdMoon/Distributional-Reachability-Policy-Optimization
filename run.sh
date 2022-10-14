#!/bin/sh

PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)
export PYTHONPATH=$PYTHONPATH:$PARENT_DIR

# quadrotor
# DRPO
for i in 64578 219803 4354 43567 49283
do
    python main.py -c config/quadrotor.json \
        -s seed $i \
        -s alg_cfg.safe_shield True \
        -s alg_cfg.sac_cfg.qc_under_uncertainty True \
        -s alg_cfg.sac_cfg.distributional_qc True \
        -s alg DRPO
done

# cartpole-move
for i in 1 43567 49283 789 8768
do
    python main.py -c config/cartpole-move.json \
        -s seed $i \
        -s alg_cfg.safe_shield True \
        -s alg_cfg.sac_cfg.qc_under_uncertainty True \
        -s alg_cfg.sac_cfg.distributional_qc True \
        -s alg DRPO
done