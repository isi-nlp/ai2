#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-06 09:17:36
# @Author  : Chenghao Mou (chengham@isi.edu)
# @Link    : https://github.com/ChenghaoMou/ai2

# pylint: disable=unused-wildcard-import
# pylint: disable=no-member

import os
import random
import sys
from pathlib import Path
from typing import Dict
from functools import partial

import numpy as np
import torch
import yaml
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.arg_parse import add_default_args
from test_tube import HyperOptArgumentParser, Experiment

from huggingface import HuggingFaceClassifier
from textbook.utils import set_seed, get_default_hyperparameter


def main(hparams):
    curr_dir = Path("output") / f"{hparams.experiment_name}"

    log_dir = curr_dir / f"{hparams.model_type}-{hparams.model_weight}-log"

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    exp = Experiment(
        name=hparams.task_name,
        version=0,
        save_dir=log_dir,
        autosave=True,
    )

    model_save_path = curr_dir / f"{hparams.model_type}-{hparams.model_weight}-checkpoints" / hparams.task_name / str(
        exp.version) / ""
    logger.info(f"Saving model to {model_save_path}")

    running_config = yaml.safe_load(open(hparams.running_config_file, "r"))
    task_config = yaml.safe_load(open(hparams.task_config_file, 'r'))

    default_parameter = partial(get_default_hyperparameter, config=running_config,
                                task_name=hparams.task_name, model_type=hparams.model_type,
                                model_weight=hparams.model_weight)

    # Check parameter that might be specified in the script for search, if none, get default
    if hparams.batch_size is None:
        hparams.batch_size = default_parameter(field='batch_size')
    if hparams.learning_rate is None:
        hparams.learning_rate = float(default_parameter(field='lr'))

    hparams.max_nb_epochs = default_parameter(field='max_nb_epochs')
    hparams.initializer_range = float(default_parameter(field='initializer_range'))
    hparams.dropout = float(default_parameter(field='dropout'))
    hparams.max_seq_len = default_parameter(field='max_seq_len')
    hparams.seed = default_parameter(field='seed')
    hparams.weight_decay = default_parameter(field='weight_decay')
    hparams.warmup_steps = default_parameter(field='warmup_steps')
    hparams.adam_epsilon = float(default_parameter(field='adam_epsilon'))
    hparams.accumulate_grad_batches = default_parameter(field='accumulate_grad_batches')
    hparams.do_lower_case = default_parameter(field='do_lower_case')
    hparams.model_save_path = model_save_path
    hparams.output_dimension = task_config[hparams.task_name].get('output_dimension', 1)
    hparams.tokenizer_type = hparams.model_type if hparams.tokenizer_type is None else hparams.tokenizer_type
    hparams.tokenizer_weight = hparams.model_weight if hparams.tokenizer_weight is None else hparams.tokenizer_weight

    exp.argparse(hparams)
    exp.save()
    logger.info(hparams)
    set_seed(hparams.seed)

    # TODO: Change this to your own model
    model = HuggingFaceClassifier(hparams)

    print(model)

    early_stop = EarlyStopping(
        monitor=hparams.early_stop_metric,
        patience=hparams.early_stop_patience,
        verbose=True,
        mode=hparams.early_stop_mode
    )

    checkpoint = ModelCheckpoint(
        filepath=hparams.model_save_path,
        save_best_only=True,
        verbose=True,
        monitor=hparams.model_save_monitor_value,
        mode=hparams.model_save_monitor_mode
    )

    trainer = Trainer(
        experiment=exp,
        checkpoint_callback=checkpoint,
        early_stop_callback=early_stop,
        gradient_clip_val=hparams.gradient_clip_val,
        process_position=0,
        nb_gpu_nodes=1,
        gpus=[i for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None,
        log_gpu_memory=True,
        show_progress_bar=True,
        overfit_pct=0.0,
        track_grad_norm=hparams.track_grad_norm,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
        fast_dev_run=False,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        max_nb_epochs=hparams.max_nb_epochs,
        min_nb_epochs=hparams.min_nb_epochs,
        train_percent_check=hparams.train_percent_check,
        val_percent_check=hparams.val_percent_check,
        test_percent_check=hparams.val_percent_check,
        val_check_interval=hparams.val_check_interval,
        log_save_interval=hparams.log_save_interval,
        row_log_interval=hparams.row_log_interval,
        distributed_backend='dp',
        use_amp=hparams.use_amp,
        print_nan_grads=hparams.check_grad_nans,
        print_weights_summary=True,
        amp_level=hparams.amp_level,
        nb_sanity_val_steps=5,
    )

    trainer.fit(model)


if __name__ == '__main__':
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parent_parser = HyperOptArgumentParser(strategy='random_search', add_help=True)
    add_default_args(parent_parser, root_dir)

    # TODO: Change this to your own model
    parser = HuggingFaceClassifier.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    main(hyperparams)
