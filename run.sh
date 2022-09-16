#!/bin/sh

PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)
export PYTHONPATH=$PYTHONPATH:$PARENT_DIR

# quadrotor cartpole-move

for i in 5435
do
    python main.py -c config/cartpole-move.json \
        -s seed $i \
        -s alg_cfg.sac_cfg.qc_under_uncertainty True \
        -s alg_cfg.sac_cfg.distributional_qc True \
        -s alg_cfg.safe_shield True
done