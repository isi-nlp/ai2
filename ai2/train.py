from datetime import datetime
import random

from vistautils.parameters_only_entrypoint import parameters_only_entry_point
from vistautils.parameters import Parameters
from loguru import logger
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
import torch

from ai2.eval import evaluate
from ai2.model import Classifier


# Date and time formats for saving dated/timed outputs
DATE_FORMAT = '%Y-%m-%d'
TIME_FORMAT = '%H-%M-%S'


def train(params: Parameters):
    # Load all expected parameters at the start so that if we're missing one, we crash immediately
    # instead of wasting time finishing half the script.
    save_path = params.optional_creatable_directory('save_path')
    save_by_date_and_parameters = params.boolean('save_by_date_and_parameters')
    save_best_only = params.boolean('save_best_only')
    build_on_pretrained_model = params.optional_existing_file('build_on_pretrained_model')

    model_name = params.string('model.model_name')
    task_name = params.string('task_name')
    task_name2 = params.optional_string('task_name2')
    architecture = params.string('architecture')
    train_data_slice = params.integer('train_data_slice')
    maybe_random_seed = params.get('random_seed', object)

    eval_after_training = params.boolean('eval_after_training')
    val_x_file = params.existing_file('val_x')
    val_y_file = params.existing_file('val_y')

    # If the training is deterministic for debugging purposes, we set the random seed
    if not isinstance(maybe_random_seed, bool):
        if not isinstance(maybe_random_seed, int):\
            raise RuntimeError(
                 "Random seed must be either false (i.e. no random seed)"
                 "or an integer seed!"
            )
        logger.info(f"Running deterministic model with seed {maybe_random_seed}")
        torch.manual_seed(maybe_random_seed)
        np.random.seed(maybe_random_seed)
        random.seed(maybe_random_seed)
        if torch.cuda.is_available():
            torch.backends.cuda.deterministic = True
            torch.backends.cuda.benchmark = False

    # Initialize the classifier by arguments specified in config file
    config = params.namespace('model').as_nested_dicts()
    config.update((k, v) for k, v in params.as_nested_dicts() if k != 'model')
    model = Classifier(config)
    logger.info('Initialized classifier.')

    if save_by_date_and_parameters:
        now = datetime.now()
        date = now.strftime(DATE_FORMAT)
        time = now.strftime(TIME_FORMAT)
        save_path = save_path / date / time / f"{model_name}_{task_name}-{train_data_slice}_{architecture}_s{maybe_random_seed}"
        if task_name2:
            save_path = save_path / f"_{task_name2}"

    if build_on_pretrained_model:
        logger.info('Loading pretrained checkpoint...')
        device = 'cpu' if not torch.cuda.is_available() else "cuda"
        checkpoint = torch.load(build_on_pretrained_model, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        save_path += f"_pretrained_{str(build_on_pretrained_model).split('/')[-1].split('.')[0]}"
    logger.info(f'Output directory: {save_path}')

    # Define the trainer along with its checkpoint and experiment instance
    checkpoint = ModelCheckpoint(
        filepath=str(save_path / 'checkpoints' / 'foo'),  # Last part needed due to parsing logic
        verbose=True,
        save_top_k=1 if save_best_only else -1,
    )
    tt_logger = TestTubeLogger(
        save_dir=str(save_path),
        name=task_name,
        version=0,
    )
    tt_logger.experiment.autosave = True
    # We pass the trainer parameters using the values from config (rather than params) to
    # better reflect the parameters we passed the model.
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

    if eval_after_training:
        logger.info('Start model evaluation')
        # Evaluate the model with evaluate function from eval.py
        evaluate(a_classifier=model, output_path=save_path, results_path=save_path / "results.txt",
                 compute_device=('cpu' if not torch.cuda.is_available() else "cuda"),
                 val_x=val_x_file, val_y=val_y_file)


if __name__ == "__main__":
    parameters_only_entry_point(train)
