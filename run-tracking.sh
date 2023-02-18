#!/bin/sh

PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)
export PYTHONPATH=$PYTHONPATH:$PARENT_DIR

# tracking-sine
# DRPO
for i in 22
do
    python main.py -c config/tracking-sine.json \
        -s seed $i \
        -s alg_cfg.safe_shield True \
        -s alg_cfg.sac_cfg.qc_under_uncertainty True \
        -s alg_cfg.sac_cfg.distributional_qc True \
        -s alg_cfg.eval_shield_type none \
        -s alg DRPO
done