from typing import *
import hydra
import torch
import random
import numpy as np
from pytorch_lightning import Trainer
from loguru import logger
from model import Classifier


@hydra.main(config_path="config.yaml")
def train(config):

    logger.info(config)

    np.random.seed(42)
    random.seed(42)

    if torch.cuda.is_available():
        torch.backends.cuda.deterministic = True
        torch.backends.cuda.benchmark = False

    model = Classifier(config)
    trainer = Trainer(
        gradient_clip_val = 0,
        num_nodes=1,
        gpus = None if not torch.cuda.is_available() else [i for i in range(torch.cuda.device_count())],
        log_gpu_memory=True,
        show_progress_bar=True,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        max_epochs=config["max_epochs"],
        min_epochs=1,
        val_check_interval=0.1,
        log_save_interval=100,
        row_log_interval=10,
        distributed_backend = "ddp",
        use_amp=config["use_amp"],
        weights_summary= 'top',
        amp_level='O2',
        num_sanity_val_steps=5,
        resume_from_checkpoint=None,
    )
    trainer.fit(model)

    pass

if __name__ == "__main__":
    train()
