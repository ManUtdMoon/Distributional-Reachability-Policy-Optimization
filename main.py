from pathlib import Path

import numpy as np
np.set_printoptions(precision=3, linewidth=120)

from src import cli
from src.defaults import ROOT_DIR
from src.log import default_log as log, TabularLog
from src.checkpoint import CheckpointableData, Checkpointer
from src.config import BaseConfig, Require
from src.torch_util import device
from src.shared import get_env
from src.smbpo import SMBPO


ROOT_DIR = Path(ROOT_DIR)
SAVE_PERIOD = 10


class Config(BaseConfig):
    env_name = Require(str)
    env_cfg = {}
    seed = 64578
    epochs = 600
    alg_cfg = SMBPO.Config()


def main(cfg):
    env_factory = lambda id=None: get_env(cfg.env_name, **{**cfg.env_cfg, **dict(id=id)})
    data = CheckpointableData()
    alg = SMBPO(cfg.alg_cfg, env_factory, data, cfg.epochs)
    alg.to(device)
    checkpointer = Checkpointer(alg, log.dir, 'ckpt_{}.pt')
    data_checkpointer = Checkpointer(data, log.dir, 'data.pt')

    # Check if existing run
    if data_checkpointer.try_load():
        log('Data load succeeded')
        loaded_epoch = checkpointer.load_latest(list(range(0, cfg.epochs, SAVE_PERIOD)))
        if isinstance(loaded_epoch, int):
            assert loaded_epoch == alg.epochs_completed
            log('Solver load succeeded')
        else:
            assert alg.epochs_completed == 0
            log('Solver load failed')
    else:
        log('Data load failed')

    if alg.epochs_completed == 0:
        alg.setup()
        eval_tabular_log = TabularLog(log.dir, 'eval.csv')
        # So that we can compare to the performance of randomly initialized policy
        eval_tabular_log.row(alg.evaluate())

    while alg.epochs_completed < cfg.epochs:
        log(f'Beginning epoch {alg.epochs_completed+1}')
        alg.epoch()
        eval_tabular_log.row(alg.evaluate())

        if alg.epochs_completed % SAVE_PERIOD == 0 or alg.epochs_completed < 10:
            checkpointer.save(alg.epochs_completed)
            data_checkpointer.save()


if __name__ == '__main__':
    cli.main(Config(), main)