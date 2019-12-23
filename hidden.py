#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-06 09:17:36
# @Author  : Chenghao Mou (chengham@isi.edu)
# @Link    : https://github.com/ChenghaoMou/ai2

# pylint: disable=unused-wildcard-import
# pylint: disable=no-member

# Roberta is better than bert in terms of commonsense
# if it is true:
#   training instance is helpful?
#   avg cls token embeds
#   piqa & siqa
#   question only
#   clean
#   visualization


import os
import sys
import json
import torch
import numpy as np
import yaml
import pickle
from glob import glob
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.trainer_io import load_hparams_from_tags_csv
from pytorch_lightning.utilities.arg_parse import add_default_args
from test_tube import HyperOptArgumentParser
from torch.utils.data import DataLoader, RandomSampler
from huggingface import HuggingFaceClassifier
import annoy
from annoy import AnnoyIndex
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def embed_dataset(model, dataset, embed_output, mapping_output):

    t = AnnoyIndex(model.encoder.dim, 'angular')
    mapping = {}
    count = 0

    if os.path.exists(embed_output) and os.path.exists(mapping_output):

        with open(mapping_output, "r") as f:
            mapping = json.loads(f.read())

        t.load(embed_output)

        return t, mapping

    for i, batch in tqdm(enumerate(DataLoader(dataset, shuffle=False, batch_size=4, collate_fn=model.collate_fn))):

        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)
        batch["token_type_ids"] = batch["token_type_ids"].to(device)
        batch["y"] = batch["y"].to(device)

        batch_size, num_choice, seq = batch["input_ids"].shape

        with torch.no_grad():

            embeds = model.intermediate(
                batch["input_ids"].view(-1, seq),
                batch["token_type_ids"].view(-1, seq),
                batch["attention_mask"].view(-1, seq))

            embeds = embeds.reshape(batch_size, num_choice, -1)
            embeds = embeds.cpu().detach().tolist()

            preds = model.validation_step(batch, -1)["batch_logits"]
            preds = torch.argmax(preds, dim=-1).squeeze().cpu().detach().numpy().tolist()
            if batch_size == 1:
                preds = [preds]

            for i in range(len(embeds)):
                for j in range(len(embeds[i])):
                    truth = batch["y"][i].item()
                    t.add_item(count, embeds[i][j])
                    mapping[str(count)] = {
                        "text": model.tokenizer.tokenizer.convert_tokens_to_string(batch["tokens"][i][j]),
                        "label-correct": truth == j,
                        "is_pred": j == preds[i],
                        "pred-correct": truth == preds[i],
                    }
                    count += 1

    t.build(100)
    t.save(embed_output)

    with open(mapping_output, "w") as f:
        f.write(json.dumps(mapping, indent=4))

    return t, mapping

def embed_dataset_sentence_transformer(embed_model, model, dataset, embed_output, mapping_output):

    t = AnnoyIndex(model.encoder.dim, 'angular')
    mapping = {}
    count = 0

    if os.path.exists(embed_output) and os.path.exists(mapping_output):

        with open(mapping_output, "r") as f:
            mapping = json.loads(f.read())

        t.load(embed_output)

        return t, mapping

    for i, batch in tqdm(enumerate(DataLoader(dataset, shuffle=False, batch_size=4, collate_fn=model.collate_fn))):

        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)
        batch["token_type_ids"] = batch["token_type_ids"].to(device)
        batch["y"] = batch["y"].to(device)

        sentences = []
        for i in range(len(batch['tokens'])):
            sentences.append([model.tokenizer.tokenizer.convert_tokens_to_string(batch['tokens'][i][j]).replace("<s>", "").replace("</s>", " ").replace("[CLS]", "").replace("[SEP]", " ") for j in range(len(batch['tokens'][i]))])

        batch_size, num_choice, seq = batch["input_ids"].shape

        with torch.no_grad():

            # embeds = model.intermediate(
            #     batch["input_ids"].view(-1, seq),
            #     batch["token_type_ids"].view(-1, seq),
            #     batch["attention_mask"].view(-1, seq))

            # embeds = embeds.reshape(batch_size, num_choice, -1)
            # embeds = embeds.cpu().detach().tolist()

            embeds = [embed_model.encode(e) for e in sentences]

            preds = model.validation_step(batch, -1)["batch_logits"]
            preds = torch.argmax(preds, dim=-1).squeeze().cpu().detach().numpy().tolist()
            if batch_size == 1:
                preds = [preds]

            for i in range(len(embeds)):
                for j in range(len(embeds[i])):
                    truth = batch["y"][i].item()
                    t.add_item(count, embeds[i][j])
                    mapping[str(count)] = {
                        "text": sentences[i][j],
                        "label-correct": truth == j,
                        "is_pred": j == preds[i],
                        "pred-correct": truth == preds[i],
                    }
                    count += 1

    t.build(100)
    t.save(embed_output)

    with open(mapping_output, "w") as f:
        f.write(json.dumps(mapping, indent=4))

    return t, mapping



def metric_closest(train_embed, train_mapping, val_embed, val_mapping, output, mode="tcorrect-correct"):

    val_mode, train_mode = mode.split("-")

    if val_mode == "tcorrect":
        # truth correct pairs
        def val_mode_filter(x): return x["label-correct"] == True
    elif val_mode == "twrong":
        def val_mode_filter(x): return x["label-correct"] == False
    elif val_mode == "pcorrect":
        def val_mode_filter(x): return x["is_pred"] and x["pred-correct"]
    elif val_mode == "pwrong":
        def val_mode_filter(x): return x["is_pred"] and x["pred-correct"] == False
    elif val_mode == "np":
        def val_mode_filter(x): return not x["is_pred"]

    if train_mode == "correct":
        def train_mode_filter(x): return x["label-correct"] == True
    elif train_mode == "wrong":
        def train_mode_filter(x): return x["label-correct"] == False

    val_indicies = [i for i in val_mapping if val_mode_filter(val_mapping[i])]
    train_indicies = [i for i in train_mapping if train_mode_filter(train_mapping[i])]
    # print(train_indicies)
    population = []

    result = {}

    for i in tqdm(val_indicies):
        # print(i)
        candidate_indices, candidate_distances = train_embed.get_nns_by_vector(
            val_embed.get_item_vector(int(i)), 500, include_distances=True)

        valid_indices, valid_distances = zip(*[(str(i), d)
                                               for i, d in sorted(zip(candidate_indices, candidate_distances)) if str(i) in train_indicies])

        population.extend(valid_distances)

        result[i] = {
            "val": val_mapping[i],
            "train": train_mapping[valid_indices[0]]
        }

    with open(f"{output}-{mode}.json", "w") as f:
        f.write(json.dumps(result, indent=4))

    p = sns.distplot(population, label=mode, hist=False, kde=True, kde_kws={'shade': False, 'linewidth': 3},)
    plt.legend()
    p.set_title(output)
    f = p.get_figure()
    f.savefig(f"{output}.png")


def main(hparams):

    # TODO: Change this model loader to your own.

    model1 = HuggingFaceClassifier.load_from_metrics(
        hparams=hparams,
        weights_path=hparams.weights_path,
        tags_csv=hparams.tags_csv,
        on_gpu=torch.cuda.is_available(),
        map_location=None
    )

    hparams.model_type = 'roberta' if hparams.model_type == "xlnet" else hparams.model_type
    model2 = SentenceTransformer(f'{hparams.model_type}-large-nli-stsb-mean-tokens')

    model2.to(device)
    model1.to(device)

    train_embed, train_mapping = embed_dataset_sentence_transformer(model2,
        model1, model1.train_dataloader.dataset, f"{hparams.model_type}-{hparams.task_name}-train.nn",
        f"{hparams.model_type}-{hparams.task_name}-train.json")

    val_embed, val_mapping = embed_dataset_sentence_transformer(model2,
        model1, model1.val_dataloader.dataset, f"{hparams.model_type}-{hparams.task_name}-val.nn",
        f"{hparams.model_type}-{hparams.task_name}-val.json")

    metric_closest(train_embed, train_mapping, val_embed, val_mapping,
                   f"{hparams.model_type}-{hparams.task_name}", "tcorrect-correct")
    metric_closest(train_embed, train_mapping, val_embed, val_mapping,
                   f"{hparams.model_type}-{hparams.task_name}", "twrong-correct")
    metric_closest(train_embed, train_mapping, val_embed, val_mapping,
                   f"{hparams.model_type}-{hparams.task_name}", "twrong-wrong")
    metric_closest(train_embed, train_mapping, val_embed, val_mapping,
                   f"{hparams.model_type}-{hparams.task_name}", "tcorrect-wrong")
    metric_closest(train_embed, train_mapping, val_embed, val_mapping,
                   f"{hparams.model_type}-{hparams.task_name}", "pwrong-wrong")
    metric_closest(train_embed, train_mapping, val_embed, val_mapping,
                   f"{hparams.model_type}-{hparams.task_name}", "pcorrect-wrong")


if __name__ == '__main__':
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parent_parser = HyperOptArgumentParser(strategy='random_search', add_help=True)
    add_default_args(parent_parser, root_dir)

    # TODO: Change this to your own model
    parser = HuggingFaceClassifier.add_model_specific_args(parent_parser)
    parser.add_argument("--save", type=str)
    hyperparams = parser.parse_args()

    main(hyperparams)
