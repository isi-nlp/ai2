import os
from pathlib import Path

import hydra
from loguru import logger
import omegaconf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
import torch

from autoresume import AutoResume
from eval import evaluate
from model import Classifier

# Save root path as hydra will create copies of this code in date specific folder
ROOT_PATH = Path(__file__).parent.absolute()


@hydra.main(config_path="config/train.yaml")
def train(config: omegaconf.Config):
    config = omegaconf.OmegaConf.to_container(config)
    logger.info(config)

    # Automatically generates random seed if none given
    config['random_seed'] = seed_everything(config['random_seed'])
    logger.info(f"Running deterministic model with seed {config['random_seed']}")

    # Initialize the classifier by arguments specified in config file
    save_path = config['save_path']
    if not save_path:
        save_path = f"{config['model']}-{config['task_name']}-s{config['random_seed']}"
    if config['build_on_pretrained_model']:
        logger.info(f"Loading model from {config['build_on_pretrained_model']}")
        device = 'cpu' if not torch.cuda.is_available() else "cuda"
        model = Classifier.load_from_checkpoint(ROOT_PATH / config['build_on_pretrained_model'], map_location=device)
        model.hparams = config
        save_path += f"-pretrained_{Path(config['build_on_pretrained_model']).stem}"
    else:
        model = Classifier(config)

    if config['resume_from_checkpoint']:
        checkpoint_path = ROOT_PATH / config['resume_from_checkpoint']
    else:
        checkpoint_path = None

    # Define the trainer along with its checkpoint and experiment instance
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(save_path, 'checkpoints', 'foo'),  # Last part needed due to parsing logic
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
        callbacks=[AutoResume()] if config['autoresume'] else None,
        gradient_clip_val=0,
        gpus=-1 if torch.cuda.is_available() else None,
        log_gpu_memory="all",
        progress_bar_refresh_rate=1,
        check_val_every_n_epoch=1,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        max_epochs=config["max_epochs"],
        min_epochs=1,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        limit_test_batches=1.0,
        log_save_interval=25,
        row_log_interval=25,
        distributed_backend="dp",
        precision=16 if config["use_amp"] else 32,
        weights_summary='top',
        num_sanity_val_steps=5,
        resume_from_checkpoint=checkpoint_path,
        benchmark=False,
        deterministic=True,
        fast_dev_run=True,  # TODO: Remove
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
