# Distributional Reachability Policy Optimization (DRPO)
Code for the paper [**"Safe Model-Based Reinforcement Learning with an Uncertainty-Aware Reachability Certificate"**](https://arxiv.org/abs/2210.07553), co-authored by [Dongjie Yu*](https://manutdmoon.github.io/), [Wenjun Zou*](https://github.com/THUzouwenjun), Yujie Yang*, [Haitong Ma](https://mahaitongdae.github.io), Shengbo Eben Li, [Jingliang Duan](https://github.com/Jingliang-Duan) and [Jianyu Chen](http://people.iiis.tsinghua.edu.cn/~jychen/).

The paper is submitted to IEEE Transactions on Automation Science and Engineering.

## Acknowledgements
The code is based on [SMBPO](https://github.com/gwthomas/Safe-MBPO) by Garrett Thomas. Thank him for his wonderful and clear implementation.

## Branches Overview
| Branch name 	| Usage 	|
|:---:	|:---:	|
| [drpo-other_env-viz](https://github.com/ManUtdMoon/Safe_MBRL) 	| DRPO implementation for ``quadrotor`` and ``cartpole-move``; also for ablation_1 on different modules; training curves visualization. 	|
| [drpo-safetygym-viz](https://github.com/ManUtdMoon/Safe_MBRL/tree/drpo-safetygym-viz) 	| DRPO implementation for ``safetygym-car`` and ``safetygym-point``; also for ablation_1 on different modules; ablation_2 on different $\Phi^{-1}(\beta)$. 	|
| [csc-other_env](https://github.com/ManUtdMoon/Safe_MBRL/tree/csc-other_env), [csc-safetygym](https://github.com/ManUtdMoon/Safe_MBRL/tree/csc-safetygym) 	| ``Conservative Safety Critics`` and ``MBPO-Lagrangian`` implementation for different envs. 	|
| [smbpo](https://github.com/ManUtdMoon/Safe_MBRL/tree/smbpo), [smbpo-safetygym](https://github.com/ManUtdMoon/Safe_MBRL/tree/smbpo-safetygym) 	| ``SMBPO`` and ``MBPO`` implementation for different envs. 	|
| [drpo-safetygym-ablation_3-constraints](https://github.com/ManUtdMoon/Safe_MBRL/tree/drpo-safetygym-ablation_3-constraints) 	| Ablation_3 on different constraint formulations (intermediate policy or shield policy). 	|
| Other branches are all deprecated. 	|  	|


## Prerequisites
1. Install MuJoCo and [mujoco-py](https://github.com/openai/mujoco-py).
2. Clone [safe-control-gym](https://github.com/ManUtdMoon/safe-control-gym) and [safety-gym](https://github.com/ManUtdMoon/safety-gym) and run ``pip install -e .`` in both directories to install the two environments. Note that we make changes (such as time-up settings) to the envs so they are different from the versions developed by original authors. You need to **install our repositories** to run DRPO codes.
3. run ``pip install -r requirements.txt``.
3. Set the ``ROOT_DIR`` in ``./src/defaults.py`` as ``/your/path/to/this/repository``. This is where experiments' logs and checkpoints will be placed.


## Run the code
Run
```bash
python main.py -c config/ENV.json
```
or
```bash
sh run-exp_name.sh
```
in the command line.

- **More envs** Now we only support ``ENV=cartpole-move, quadrotor, safetygym-car, safetygym-point``. But you are free to customize your own env as long as you implement it with ``check_done``, ``check_violation`` and ``get_constrained_values`` on top of basic gym envs. Remenber to put it in ``./src/env`` and add it in ``./src/shared.py``

- **Change hyper parameters** You are free to finetune hyper-parameters in three ways: (1) change values in different ``.py`` files; (2) change values in ``./config/ENV.json`` and (3) change values in the command line with ``python main.py -c config/ENV.json -s PAMRM VALUE``.  Use ``.`` to specify hierarchical structure in the config, e.g. ``-s alg_cfg.horizon 10``. The priorities of the three ways are from low to high (e.g., a value in (1) will be overrided by the value specified in (3)).

- **Experiments results** will be stored in ``./ENV/{time}_{alg_name}_{random_seed}``, together with configs, checkpoints, training and evaluation data.


## Test and visualize the trajectories (only for ``cartpole-move`` and ``quadrotor``)
- Check and run the command line in the ``./src/tester.py`` and the results will be stored in the corresponding logs directories.

- Now you can run python files in ``./src/viz_cartpole`` and ``./src/viz_quadrotor`` to see the learned multipliers, reachability certificates and the test trajectories. Images of ``cartpole-move`` will be stored in the tester directory in the logs while trajectories of ``quadrotor`` will be stored in ``./src/viz_quadrotor``.

## Plot the training curves.
1. Collect the results of each algorithm in ``./logs/ENV/ALGO/{time}_{alg_name}_{random_seed1}``, ``./logs/ENV/ALGO/{time}_{alg_name}_{random_seed2}``, etc.
2. See ``./src/viz_curves.ipynb`` and add your algorithms to ``alg_list`` in ``help_func()``.
3. ``plot_eval_results_of_all_alg_n_runs(ENV)`` and watch the curves.

## Contributing
When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with me before making a change. Also free to fork/star and make any changes.

## If you find our paper/codes helpful, welcome to cite:
```
@misc{yu2022safe,
  doi = {10.48550/ARXIV.2210.07553},
  url = {https://arxiv.org/abs/2210.07553},
  author = {Yu, Dongjie and Zou, Wenjun and Yang, Yujie and Ma, Haitong and Li, Shengbo Eben and Duan, Jingliang and Chen, Jianyu},  
  title = {Safe Model-Based Reinforcement Learning with an Uncertainty-Aware Reachability Certificate},
  publisher = {arXiv},
  year = {2022},
}
```
