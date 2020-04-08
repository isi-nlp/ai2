import os
import pathlib
import random
from typing import *

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from test_tube import Experiment
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Classifier

# Save root path as hydra will create copies of this code in a folder
ROOT_PATH = pathlib.Path(__file__).parent.absolute()


@hydra.main(config_path="config/train.yaml")
def train_w_eval(config):
    logger.info(config)

    # If the training is deterministic for debugging purposes, we set the random seed
    if config['random_seed']:
        logger.info(f"Running deterministic model with seed {config['random_seed']}")
        torch.manual_seed(config['random_seed'])
        np.random.seed(config['random_seed'])
        random.seed(config['random_seed'])
        if torch.cuda.is_available():
            torch.backends.cuda.deterministic = True
            torch.backends.cuda.benchmark = False

    # Initialize the classifier by arguments specified in config file
    model = Classifier(config)
    save_path = f"{config['model']}_{config['task_name']}"

    # Define the trainer along with its checkpoint and experiment instance
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(save_path, 'checkpoints'),
        save_best_only=config['save_best_only'],
        verbose=True,
    )
    exp = Experiment(
        name=config['task_name'],
        version=0,
        save_dir=save_path,
        autosave=True,
    )
    trainer = Trainer(
        gradient_clip_val=0,
        gpus=None if not torch.cuda.is_available() else [i for i in range(torch.cuda.device_count())],
        log_gpu_memory=True,
        show_progress_bar=True,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        max_nb_epochs=config["max_epochs"],
        min_nb_epochs=1,
        val_check_interval=0.02,
        log_save_interval=25,
        row_log_interval=25,
        distributed_backend="dp",
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

    # Evaluate the model after it was trained
    device = 'cpu' if not torch.cuda.is_available() else "cuda"
    model.to(device)
    model.eval()

    predictions: List[int] = []
    probabilities: List[List[float]] = []
    for batch in tqdm(DataLoader(model.dataloader(
            ROOT_PATH / config["val_x"],
            ROOT_PATH / config["val_y"]),
            batch_size=model.hparams["batch_size"] * 2,
            collate_fn=model.collate,
            shuffle=False)):

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        with torch.no_grad():
            logits = model.forward(batch)
        predictions.extend(torch.argmax(logits, dim=1).cpu().detach().numpy().tolist())
        probabilities.extend(F.softmax(logits, dim=-1).cpu().detach().numpy().tolist())
    predictions = [p + model.label_offset for p in predictions]

    # Save the prediction and the probability
    with open(f"{save_path}/preds.lst", "w+") as f:
        f.write("\n".join(map(str, predictions)))
    with open(f"{save_path}/proba.lst", "w+") as f:
        f.write("\n".join(map(lambda l: '\t'.join(map(str, l)), probabilities)))


if __name__ == "__main__":
    train_w_eval()
