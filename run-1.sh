#!/bin/sh

PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)
export PYTHONPATH=$PYTHONPATH:$PARENT_DIR

# env: safetygym-car safetygym-point
for i in 789 7346588
do
    # Fusion
    CUDA_VISIBLE_DEVICES=1 python main.py -c config/safetygym-car.json \
        -s seed $i \
        -s alg_cfg.eval_shield_type linear \
        -s alg_cfg.sac_cfg.enable_switch False \
        -s alg_cfg.sac_cfg.enable_pi_qc False

    # # Switch
    CUDA_VISIBLE_DEVICES=1 python main.py -c config/safetygym-car.json \
        -s seed $i \
        -s alg_cfg.eval_shield_type safe \
        -s alg_cfg.sac_cfg.enable_switch True \
        -s alg_cfg.sac_cfg.enable_pi_qc False \
        -s alg Switch

    # Unified
    CUDA_VISIBLE_DEVICES=1 python main.py -c config/safetygym-car.json \
        -s seed $i \
        -s alg_cfg.eval_shield_type no \
        -s alg_cfg.sac_cfg.enable_switch False \
        -s alg_cfg.sac_cfg.enable_pi_qc True \
        -s alg Unify
done