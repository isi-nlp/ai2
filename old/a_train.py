from pathlib import Path
import random

import hydra
from loguru import logger
import numpy as np
import omegaconf
from pytorch_lightning import Trainer
import torch

from a_model import Classifier

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

# Save root path as hydra will create copies of this code in date specific folder
ROOT_PATH = Path(__file__).parent.absolute()


@hydra.main(config_path="config/train.yaml")
def train(config: omegaconf.Config):
    config = omegaconf.OmegaConf.to_container(config)
    logger.info(config)

    # If the training is deterministic for debugging purposes, we set the random seed
    if not isinstance(config['random_seed'], bool):
        logger.info(f"Running deterministic model with seed {config['random_seed']}")
        torch.manual_seed(config['random_seed'])
        np.random.seed(config['random_seed'])
        random.seed(config['random_seed'])
        if torch.cuda.is_available():
            torch.backends.cuda.deterministic = True
            torch.backends.cuda.benchmark = False

    # Initialize the classifier by arguments specified in config file
    model = Classifier(config)

    trainer = Trainer(
        gpus=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else None,
        log_gpu_memory="all",
        accumulate_grad_batches=config["accumulate_grad_batches"],
        max_epochs=config["max_epochs"],
        min_epochs=1,
        distributed_backend="dp",
        weights_summary='top',
        fast_dev_run=True,
    )
    trainer.fit(model)
    logger.success('Training Completed')


if __name__ == "__main__":
    train()
