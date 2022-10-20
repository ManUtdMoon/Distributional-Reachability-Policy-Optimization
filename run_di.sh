#!/bin/sh

PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)
export PYTHONPATH=$PYTHONPATH:$PARENT_DIR

# DRPO
for i in 219803 # 4354 43567 64578 49283
do
    python main.py -c config/double_integrator.json \
        -s seed $i \
        -s alg_cfg.safe_shield True \
        -s alg_cfg.sac_cfg.qc_under_uncertainty False \
        -s alg_cfg.sac_cfg.distributional_qc False \
        -s alg DRPO
done