#!/bin/sh

PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)
export PYTHONPATH=$PYTHONPATH:$PARENT_DIR

# env: safetygym-car safetygym-point
for i in 789 7346588
do
    # Q_h^*
    CUDA_VISIBLE_DEVICES=1 python main.py -c config/safetygym-car.json \
        -s seed $i \
        -s alg_cfg.eval_shield_type linear \
        -s alg_cfg.sac_cfg.enable_switch False \
        -s alg_cfg.sac_cfg.enable_pi_qc False \
        -s alg qh_star \
        -s alg_cfg.safe_shield True

    # Q_h^\pi
    CUDA_VISIBLE_DEVICES=1 python main.py -c config/safetygym-car.json \
        -s seed $i \
        -s alg_cfg.eval_shield_type linear \
        -s alg_cfg.sac_cfg.enable_switch False \
        -s alg_cfg.sac_cfg.enable_pi_qc True \
        -s alg qh_pi \
        -s alg_cfg.safe_shield True
done