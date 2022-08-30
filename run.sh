#!/bin/sh

PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)
export PYTHONPATH=$PYTHONPATH:$PARENT_DIR

# quadrotor cartpole-move
python main.py -c config/quadrotor.json
python main.py -c config/quadrotor.json \
    -s alg_cfg.sac_cfg.qc_under_uncertainty True \
    -s alg_cfg.sac_cfg.distributional_qc True \
    -s alg_cfg.safe_shield True

python main.py -c config/quadrotor.json -s seed 219803

python main.py -c config/quadrotor.json \
    -s seed 219803 \
    -s alg_cfg.sac_cfg.qc_under_uncertainty True \
    -s alg_cfg.sac_cfg.distributional_qc True \
    -s alg_cfg.safe_shield True

python main.py -c config/quadrotor.json -s seed 456

python main.py -c config/quadrotor.json \
    -s seed 456 \
    -s alg_cfg.sac_cfg.qc_under_uncertainty True \
    -s alg_cfg.sac_cfg.distributional_qc True \
    -s alg_cfg.safe_shield True