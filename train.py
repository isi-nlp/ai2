import os
from pathlib import Path
import random

import hydra
from loguru import logger
import numpy as np
import omegaconf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
import torch

from eval import evaluate
from model import Classifier

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
    save_path = f"{config['model']}-{config['task_name']}-s{config['random_seed']}"
    if 'task_name2' in config:
        save_path = save_path + f"-{config['task_name2']}"
    if 'experiment_id' in config:  # If provided, overwrite save path with an experiment identifier
        save_path = config['experiment_id']

    if config['build_on_pretrained_model']:
        device = 'cpu' if not torch.cuda.is_available() else "cuda"
        checkpoint = torch.load(ROOT_PATH / config['build_on_pretrained_model'], map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        save_path += f"-pretrained_{config['build_on_pretrained_model'].split('/')[-1].split('.')[0]}"

    # Define the trainer along with its checkpoint and experiment instance
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(save_path, 'checkpoints'),  # Last part needed due to parsing logic
        verbose=True,
        save_top_k=1 if config['save_best_only'] else -1,
    )
    tt_logger = TestTubeLogger(
        save_dir=save_path,
        name=config['task_name'],
        version=0,
    )
    tt_logger.experiment.autosave = True
    trainer = Trainer(
        logger=tt_logger,
        checkpoint_callback=checkpoint,
        gradient_clip_val=0,
        gpus=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else None,
        log_gpu_memory="all",
        progress_bar_refresh_rate=1,
        check_val_every_n_epoch=1,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        max_epochs=config["max_epochs"],
        min_epochs=1,
        train_percent_check=1.0,
        val_percent_check=1.0,
        test_percent_check=1.0,
        log_save_interval=25,
        row_log_interval=25,
        distributed_backend="dp",
        precision=16 if config["use_amp"] else 32,
        weights_summary='top',
        num_sanity_val_steps=5,
    )
    trainer.fit(model)
    logger.success('Training Completed')

    if config['eval_after_training']:
        logger.info('Start model evaluation')
        # Evaluate the model with evaluate function from eval.py
        evaluate(a_classifier=model, output_path=save_path,
                 compute_device=('cpu' if not torch.cuda.is_available() else "cuda"),
                 val_x=ROOT_PATH / config["val_x"], val_y=ROOT_PATH / config["val_y"])


if __name__ == "__main__":
    train()
