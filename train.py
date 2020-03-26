from typing import *
import torch
import random
import numpy as np
from pytorch_lightning import Trainer
from loguru import logger
from model import Classifier
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import argparse
from test_tube import HyperOptArgumentParser, Experiment
import os
import yaml


def train(config):

    logger.info(config)

    seed = config['random_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.backends.cuda.deterministic = True
        torch.backends.cuda.benchmark = False

    model = Classifier(config)

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(config['save_path'], 'checkpoints'),
        save_best_only=config['save_best_only'] if 'save_best_only' in config else False,
        verbose=True,
    )
    exp = Experiment(
        name=config['task_name'],
        version=0,
        save_dir=config['save_path'],
        autosave=True,
    )
    trainer = Trainer(
        gradient_clip_val = 0,
        gpus = None if not torch.cuda.is_available() else [i for i in range(torch.cuda.device_count())],
        log_gpu_memory=True,
        show_progress_bar=True,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        max_nb_epochs=config["max_epochs"],
        min_nb_epochs=1,
        val_check_interval=0.02,
        log_save_interval=25,
        row_log_interval=25,
        distributed_backend = "dp",
        use_amp=config["use_amp"],
        nb_sanity_val_steps=5,
        checkpoint_callback=checkpoint,
        check_val_every_n_epoch=1.0,
        train_percent_check=1.0,
        val_percent_check=1.0,
        test_percent_check=1.0,
        experiment=exp,
    )
    trainer.fit(model)


def get_parser():
    def str2bool(v):
        v = v.lower()
        assert v == 'true' or v == 'false'
        return v.lower() == 'true'
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     conflict_handler='resolve')
    parser.add_argument('--config_file', type=str, required=True, help='path to the datasets')
    parser.add_argument('--verbose', type=str2bool, default=False,
                        help='if verbose')
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    with open(args.config_file, 'r') as ymlfile:
        config = yaml.load(ymlfile)
    train(config)