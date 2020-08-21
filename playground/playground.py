"""
Playground python file for transformer from scratch - mostly replicated the train.py file except for line 34 where
weights are initialized randomly again
"""
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TestTubeLogger

from eval import evaluate
from model import Classifier

# Save root path as hydra will create copies of this code in date specific folder
ROOT_PATH = Path(__file__).parent.absolute()


@hydra.main(config_path="playground.yaml")
def train(config):
    logger.info(config)

    # If the training is deterministic for debugging purposes, we set the random seed
    if not isinstance(config['random_seed'], bool):
        seed_everything(config['random_seed'])
        logger.info(f"Running deterministic model with seed {config['random_seed']}")

    # Initialize the classifier by arguments specified in config file
    model = Classifier(OmegaConf.to_container(config))
    logger.info('Initialized classifier.')
    model.embedder.init_weights()
    save_path = f"transformer_scratch-{config['task_name']}-s{config['random_seed']}"

    # Initialize the Test Tube for Trainer
    tt_logger = TestTubeLogger(
        save_dir=save_path,
        name=config['task_name'],
        version=0,
    )
    tt_logger.experiment.autosave = True

    # Trainer Call Back Functions
    early_stop = EarlyStopping(
        monitor='loss_minus_accuracy',
        mode='min',
        verbose=True,
        patience=3,
    )
    checkpoint = ModelCheckpoint(
        monitor='val_accuracy',
        mode='max',
        verbose=True,
        filepath=save_path + '/checkpoints/',
        save_top_k=int(config['save_top_N_models']),
        save_last=True
    )

    # Main Trainer
    trainer = Trainer(
        accumulate_grad_batches=config["accumulate_grad_batches"],
        deterministic=True if not isinstance(config['random_seed'], bool) else False,
        check_val_every_n_epoch=1,
        checkpoint_callback=checkpoint,
        distributed_backend="dp",
        early_stop_callback=False,
        gpus=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else None,
        gradient_clip_val=0,
        limit_test_batches=1.0,
        limit_val_batches=1.0,
        limit_train_batches=1.0,
        log_gpu_memory="all",
        log_save_interval=25,
        logger=tt_logger,
        max_epochs=config["max_epochs"],
        min_epochs=1,
        num_sanity_val_steps=5,
        precision=16 if config["precision"] == 'half' else 32,
        progress_bar_refresh_rate=config['progress_bar_refresh_rate'],
        row_log_interval=25,
        weights_summary='top',
        auto_lr_find=True,
    )
    trainer.fit(model)
    logger.success('Training Completed')

    # If config file request to evaluate after finish training
    if config['eval_after_training']:
        logger.info('Start model evaluation')
        # Evaluate the model with evaluate function from eval.py
        evaluate(a_classifier=model, output_path=save_path,
                 compute_device=('cpu' if not torch.cuda.is_available() else "cuda"),
                 with_progress_bar=False if config['progress_bar_refresh_rate'] == 0 else True,
                 val_x=ROOT_PATH / config["val_x"], val_y=ROOT_PATH / config["val_y"])


if __name__ == "__main__":
    train()
