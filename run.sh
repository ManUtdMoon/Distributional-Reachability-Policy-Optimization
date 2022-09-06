#!/bin/sh

PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)
export PYTHONPATH=$PYTHONPATH:$PARENT_DIR

python main.py -c config/safetygym-point.json
python main.py -c config/safetygym-point.json \
    -s alg_cfg.sac_cfg.update_violation_cost True

# python main.py -c config/cartpole-move.json \
#     -s seed 43567
# python main.py -c config/cartpole-move.json \
#     -s alg_cfg.sac_cfg.update_violation_cost True \
#     -s seed 43567

# python main.py -c config/cartpole-move.json \
#     -s seed 49283
# python main.py -c config/cartpole-move.json \
#     -s alg_cfg.sac_cfg.update_violation_cost True \
#     -s seed 49283