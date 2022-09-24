#!/bin/sh

PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)
export PYTHONPATH=$PYTHONPATH:$PARENT_DIR

# env: safetygym-car safetygym-point
for i in 46 51 90 72 # 86 46 51 90 72
do
    # Fusion
    python main.py -c config/safetygym-car.json \
        -s seed $i \
        -s alg_cfg.eval_shield_type linear \
        -s alg_cfg.sac_cfg.enable_switch False \
        -s alg_cfg.sac_cfg.enable_pi_qc False

    # Switch
    python main.py -c config/safetygym-car.json \
        -s seed $i \
        -s alg_cfg.eval_shield_type safe \
        -s alg_cfg.sac_cfg.enable_switch True \
        -s alg_cfg.sac_cfg.enable_pi_qc False

    # Unified
    python main.py -c config/safetygym-car.json \
        -s seed $i \
        -s alg_cfg.eval_shield_type no \
        -s alg_cfg.sac_cfg.enable_switch False \
        -s alg_cfg.sac_cfg.enable_pi_qc True
done