#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-21 12:06:15
# @Author  : Chenghao Mou (chengham@isi.edu)

# pylint: disable=unused-wildcard-import
# pylint: disable=no-member

import os
import sys
import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.trainer_io import load_hparams_from_tags_csv
from pytorch_lightning.utilities.arg_parse import add_default_args
from test_tube import HyperOptArgumentParser
from huggingface import HuggingFaceClassifier
from captum.attr import IntegratedGradients
from captum.attr import InterpretableEmbeddingBase, TokenReferenceBase
from captum.attr import visualization
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# accumalate couple samples in this array for visualization purposes
vis_data_records_ig = []


def main(hparams):

    def interpret_example(model, example):
        model.eval()
        model.zero_grad()
        B, C, S = example['input_ids'].shape

        input_embedding = interpretable_embedding.indices_to_embeddings(example['input_ids'].reshape(-1, S))
        *_, H = input_embedding.shape

        pred = model.forward(input_ids=input_embedding.reshape(-1, S, H),
                             token_type_ids=example['token_type_ids'].reshape(-1, S),
                             attention_mask=example['attention_mask'].reshape(-1, S)).detach()
        pred_ind = torch.softmax(pred, dim=-1).detach()
        pred = pred.numpy()

        # generate reference for each sample
        reference_indices = token_reference.generate_reference(S, device=device).unsqueeze(0).expand(B * C, S)
        reference_embedding = interpretable_embedding.indices_to_embeddings(reference_indices)

        attributions_ig, delta = ig.attribute(
            input_embedding,
            reference_embedding,
            additional_forward_args=(example['token_type_ids'].reshape(-1, S),
                                     example['attention_mask'].reshape(-1, S)),
            n_steps=500, return_convergence_delta=True)

        print('pred: ', np.argmax(pred_ind), '(', ','.join('%.2f' % d for d in pred), ')', ', delta: ', abs(delta))

        for i in range(B*C):
            for j, token in enumerate(example["tokens"][i]):
                if token.startswith("[") and token.endswith("]"):
                    attributions_ig[i][j] = 0
            add_attributions_to_visualizer(
                attributions_ig[i].unsqueeze(0), example["tokens"][i],
                pred_ind[i].item(),
                i,
                example["y"].item(),
                delta[i], vis_data_records_ig)

    def add_attributions_to_visualizer(attributions, text, pred, pred_ind, label, delta, vis_data_records):
        attributions = attributions.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.detach().numpy()

        # storing couple samples in an array for visualization purposes
        vis_data_records.append(visualization.VisualizationDataRecord(
                                attributions,
                                pred,
                                pred_ind,
                                label,
                                "word",
                                attributions.sum(),
                                text[:len(attributions)],
                                delta))

    model = HuggingFaceClassifier.load_from_metrics(
        hparams=hparams,
        weights_path=hparams.weights_path,
        tags_csv=hparams.tags_csv,
        on_gpu=torch.cuda.is_available(),
        map_location=None
    )
    token_reference = TokenReferenceBase(reference_token_idx=model.tokenizer.pad)
    interpretable_embedding = configure_interpretable_embedding_layer(model, hparams.embedding_layer)
    ig = IntegratedGradients(model)

    for _, batch in enumerate(model.val_dataloader):

        example = {
            "tokens": batch["tokens"][0],
            "input_ids": batch["input_ids"][0].unsqueeze(0),
            "token_type_ids": batch["token_type_ids"][0].unsqueeze(0),
            "attention_mask": batch["attention_mask"][0].unsqueeze(0),
            "y": batch["y"][0].unsqueeze(0),
        }

        interpret_example(model, example)

    print('Visualize attributions based on Integrated Gradients')

    open(hparams.visualization_output, "w").write(visualization.visualize_text(vis_data_records_ig).data)


if __name__ == '__main__':
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parent_parser = HyperOptArgumentParser(strategy='random_search', add_help=True)
    add_default_args(parent_parser, root_dir)
    parent_parser.add_argument('--el', '--embedding_layer', type=str, help='Name of the embedding layer in the model',
                               default='encoder.model.embeddings.word_embeddings')
    parent_parser.add_argument('--vo', '--visualization_output', type=str, help='output visualization html file', default="vis.html")

    # TODO: Change this to your own model
    parser = HuggingFaceClassifier.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    main(hyperparams)
