#!/bin/sh

PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)
export PYTHONPATH=$PYTHONPATH:$PARENT_DIR

# # quadrotor cartpole-move
# # Vanilla
# for i in 43567 748365 219803 4354 64578
# do
#     python main.py -c config/quadrotor.json \
#         -s seed $i \
#         -s alg_cfg.eval_shield_type no
# done

# # Uncertainty only
# for i in 748365 6790 90 43567 4354
# do
#     python main.py -c config/quadrotor.json \
#         -s seed $i \
#         -s alg_cfg.sac_cfg.qc_under_uncertainty True \
#         -s alg_cfg.sac_cfg.distributional_qc True \
#         -s alg_cfg.eval_shield_type no
# done

# # Shield only
# for i in 64578 219803 748365 57634 467
# do
#     python main.py -c config/quadrotor.json \
#         -s seed $i \
#         -s alg_cfg.safe_shield True
# done

# DRPO
for i in 64578 219803 4354 1 49283
do
    python main.py -c config/quadrotor.json \
        -s seed $i \
        -s alg_cfg.safe_shield False \
        -s alg_cfg.sac_cfg.qc_under_uncertainty True \
        -s alg_cfg.sac_cfg.distributional_qc True \
        -s alg_cfg.eval_shield_type no \
        -s alg qh_pi \
        -s alg_cfg.sac_cfg.enable_pi_qc True
done