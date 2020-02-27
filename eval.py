#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-06 09:17:36
# @Author  : Chenghao Mou (chengham@isi.edu)
# @Link    : https://github.com/ChenghaoMou/ai2

# pylint: disable=unused-wildcard-import
# pylint: disable=no-member

import os
import sys

import torch
import numpy as np
import yaml
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.trainer_io import load_hparams_from_tags_csv
from pytorch_lightning.utilities.arg_parse import add_default_args
from test_tube import HyperOptArgumentParser
from torch.utils.data import DataLoader, RandomSampler
from huggingface import HuggingFaceClassifier
import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(hparams):

    # TODO: Change this model loader to your own.

    model = HuggingFaceClassifier.load_from_metrics(
        hparams=hparams,
        weights_path=hparams.weights_path,
        tags_csv=hparams.tags_csv,
        on_gpu=torch.cuda.is_available(),
        map_location=None
    )

    model = model.to(device)
    model.eval()
    results = []
    for batch in DataLoader(
            model.val_dataloader.dataset, shuffle=False, batch_size=4, collate_fn=model.collate_fn):

        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)
        batch["token_type_ids"] = batch["token_type_ids"].to(device)
        batch["y"] = batch["y"].to(device)
        with torch.no_grad():
            results.append(model.validation_step(batch, -1))

    logits = torch.cat([o['batch_logits'] for o in results], dim=0).reshape(-1, results[0]['batch_logits'].shape[1])
    pred = torch.argmax(logits, dim=-1).reshape(-1)


    with open(f"{hparams.task_name}-{hparams.model_weight}-predictions.lst", "w") as output_file:
        output_file.write("\n".join(map(str, (pred + model.task_config[hparams.task_name]['label_offset']).cpu().numpy().tolist())))

    stats = []
    for _ in range(100):
        results_ = [results[i] for i in np.random.random_integers(0, len(results)-1, size=len(results))]

        res = model.validation_end(results_)
        stats.append(res['val_acc'])

    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    logger.info(f'{alpha*100:.1f} confidence interval {lower*100:.1f} and {upper*100:.1f}, average: {np.mean(stats)*100:.1f}')


if __name__ == '__main__':
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parent_parser = HyperOptArgumentParser(strategy='random_search', add_help=True)
    add_default_args(parent_parser, root_dir)

    # TODO: Change this to your own model
    parser = HuggingFaceClassifier.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    main(hyperparams)
