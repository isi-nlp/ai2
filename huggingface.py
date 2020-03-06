#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-06 09:17:36
# @Author  : Chenghao Mou (chengham@isi.edu)
# @Link    : https://github.com/ChenghaoMou/ai2

# pylint: disable=unused-wildcard-import
# pylint: disable=no-member

from __future__ import annotations

import math
import os
from inspect import getfullargspec
from typing import *
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import yaml
from pytorch_lightning.root_module.root_module import LightningModule
from pytorch_lightning.trainer.trainer_io import load_hparams_from_tags_csv
from sklearn.metrics import accuracy_score
from test_tube import HyperOptArgumentParser
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler
from transformers import *

from textbook.dataset import ClassificationDataset, download
from textbook.interface import *
from textbook.utils import set_seed, get_default_hyperparameter

import scipy.stats
import random


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


# pylint: disable=no-member


TOKENIZERS = {
    'bert': BertTokenizer,
    'distilbert': DistilBertTokenizer,
    'xlm': XLMTokenizer,
    'xlnet': XLNetTokenizer,
    'roberta': RobertaTokenizer,
    'gpt': OpenAIGPTTokenizer,
    'gpt2': GPT2Tokenizer,
    'libert': BertTokenizer,
    'albert': AlbertTokenizer
}

MODELS = {
    'bert': BertModel,
    'distilbert': DistilBertModel,
    'xlm': XLMModel,
    'xlnet': XLNetModel,
    'roberta': RobertaModel,
    'roberta_mlm': RobertaForMaskedLM,
    'gpt': OpenAIGPTModel,
    'gpt2': GPT2Model,
    'albert': AlbertModel,
    #    'libert': LiBertModel
}


class HuggingFaceModelLoader(ModelLoader):

    def __init__(self, model: Union[Module, PreTrainedModel], model_type: str):
        super(HuggingFaceModelLoader, self).__init__(model)
        if model_type == 'roberta_mlm':
            self.lm_head = self.model.lm_head
            self.model = self.model.roberta

    def forward(self, **kwargs) -> Tuple:
        """Follow the convention of omnx, return tuple whenever possible.

        token_type_ids are used when a more is pretrained with {0, 1} token_type_ids.
        RoBERTa has the argument but does not support it yet.

        Returns:
            Tuple -- Tuple of returned values of forward.
        """
        signature = getfullargspec(self.model.forward)
        valid_args = {k: torch.zeros_like(v) if k == "token_type_ids" and getattr(self.model.config, 'type_vocab_size',
                                                                                  0) < 2 else v for k, v
                      in kwargs.items()
                      if k in signature.args}

        if "input_images" in signature.args:
            batch_size, seq_len = valid_args['input_ids'].shape
            valid_args['input_images'] = torch.zeros((batch_size, 3, seq_len, 84, 84)).to(
                valid_args['input_ids'].device)
            valid_args['dummy'] = True
        return self.model.forward(
            **valid_args
        )

    @classmethod
    def load(cls, model_type: str, model_weights: str, *args, **kargs) -> HuggingFaceModelLoader:
        assert model_type in MODELS, "Model type is not recognized."
        return HuggingFaceModelLoader(MODELS[model_type].from_pretrained(model_weights, cache_dir="./model_cache"),
                                      model_type=model_type)

    @property
    def dim(self) -> int:
        """Return the hidden dimension of the last layer.
        Returns:
            int -- Last layer's dimension.
        """
        return [p.size(0) for p in self.model.parameters()][-1]


class HuggingFaceTokenizerLoader(TokenizerLoader):

    @classmethod
    def load(cls, model_type: str, model_weights: str, *args, **kargs) -> HuggingFaceTokenizerLoader:
        assert model_type in TOKENIZERS, f"Tokenizer model type {model_type} is not recognized."
        return HuggingFaceTokenizerLoader(
            TOKENIZERS[model_type].from_pretrained(model_weights, *args, cache_dir="./model_cache", **kargs))
        # tokenizer_dir = "large_roberta"
        # return HuggingFaceTokenizerLoader(
        #     TOKENIZERS[model_type].from_pretrained(tokenizer_dir, *args, cache_dir="./model_cache", **kargs))

    @property
    def SEP(self) -> str:
        if self.tokenizer._sep_token is None:
            return ""
        return self.tokenizer._sep_token

    @property
    def sep(self) -> int:
        return self.token2id(self.SEP)

    @property
    def CLS(self) -> str:
        if self.tokenizer._cls_token is None:
            return ""
        return self.tokenizer._cls_token

    @property
    def cls(self) -> int:
        return self.token2id(self.CLS)

    @property
    def UNK(self) -> str:
        if self.tokenizer._unk_token is None:
            raise Exception('UNK token in tokenizer not found.')

        return self.tokenizer._unk_token

    @property
    def unk(self) -> int:
        return self.token2id(self.UNK)

    @property
    def PAD(self) -> str:
        if self.tokenizer._pad_token is None:
            return ""
        return self.tokenizer._pad_token

    @property
    def pad(self) -> int:
        return self.token2id(self.PAD)

    def token2id(self, token: str) -> int:
        return self.tokenizer.convert_tokens_to_ids([token])[0]

    def tokens2ids(self, tokens: List[str]) -> List[int]:
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)


class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, dataloaders, shuffle: bool = True):
        self.data: List = []
        for loader in dataloaders:
            for batch in loader:
                self.data.append(batch)
        if shuffle:
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class HuggingFaceClassifier(LightningModule):

    def __init__(self, hparams):

        super(HuggingFaceClassifier, self).__init__()
        self.hparams = hparams

        with open(self.hparams.task_config_file, 'r') as input_file:
            self.task_config = yaml.safe_load(input_file)

        with open(self.hparams.running_config_file, 'r') as input_file:
            self.running_config = yaml.safe_load(input_file)

        if not os.path.exists(self.hparams.output_dir):
            os.mkdir(self.hparams.output_dir)
        if self.hparams.task_name2 is not None:
            if not os.path.exists(self.hparams.output_dir2):
                os.mkdir(self.hparams.output_dir2)

        # TODO: Change it to your own model loader
        assert not (self.hparams.comet_cn_train100k is True and self.hparams.task2_separate_fc is True)
        if self.hparams.comet_cn_train100k:
            self.hparams.model_type = 'roberta_mlm'
        self.encoder = HuggingFaceModelLoader.load(self.hparams.model_type, self.hparams.model_weight)
        print(self.encoder)
        print(MODELS[self.hparams.model_type])
        if self.hparams.comet_cn_train100k:
            self.lm_head = self.encoder.lm_head
            self.max_e1 = 10
            self.max_r = 5
            self.max_e2 = 10
            self.cn_input_length = self.max_e1 + self.max_r + self.max_e2
            self.encoder_dim = self.encoder.dim
        self.encoder.train()
        self.dropout = nn.Dropout(self.hparams.dropout)

        if not self.hparams.comet_cn_train100k:
            self.linear = nn.Linear(self.encoder.dim, self.hparams.output_dimension)
            self.linear.weight.data.normal_(mean=0.0, std=self.hparams.initializer_range)
            self.linear.bias.data.zero_()
            if self.hparams.task2_separate_fc:
                self.linear2 = nn.Linear(self.encoder.dim, self.hparams.output_dimension)
                self.linear2.weight.data.normal_(mean=0.0, std=self.hparams.initializer_range)
                self.linear2.bias.data.zero_()
        else:
            self.linear = nn.Linear(self.encoder_dim, self.hparams.output_dimension)
            self.linear.weight.data.normal_(mean=0.0, std=self.hparams.initializer_range)
            self.linear.bias.data.zero_()

        # TODO: Change it to your own tokenizer loader
        self.tokenizer = HuggingFaceTokenizerLoader.load(
            self.hparams.tokenizer_type, self.hparams.tokenizer_weight, do_lower_case=self.hparams.do_lower_case)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, task_id=None):

        # if input_ids is not None and token_type_ids is not None and attention_mask is not None:
        #     logger.debug(f"Device: {next(self.encoder.model.parameters()).device}")
        #     logger.debug(f"Device: {input_ids.device} {token_type_ids.device} {attention_mask.device}")

        # TODO [Optional]: Change it to your own forward
        if not self.hparams.comet_cn_train100k:
            outputs = self.encoder.forward(
                **{'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask})
        elif self.hparams.comet_cn_train100k and task_id == 1:
            outputs = self.encoder.forward(
                **{'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask})
        elif self.hparams.comet_cn_train100k and task_id == 2:
            outputs = self.encoder.model.forward(
                **{'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask})
            sequence_output = outputs[0]
            prediction_scores = self.lm_head(sequence_output)
            return prediction_scores
        output = torch.mean(outputs[0], dim=1).squeeze()
        output = self.dropout(output)
        if self.hparams.task2_separate_fc and task_id == 2 and not self.hparams.comet_cn_train100k:
            logits = self.linear2(output)
        else:
            logits = self.linear(output)

        return logits.squeeze()

    def intermediate(self, input_ids, token_type_ids=None, attention_mask=None, task_id=None):

        # if input_ids is not None and token_type_ids is not None and attention_mask is not None:
        #     logger.debug(f"Device: {next(self.encoder.model.parameters()).device}")
        #     logger.debug(f"Device: {input_ids.device} {token_type_ids.device} {attention_mask.device}")

        # TODO [Optional]: Change it to your own forward
        with torch.no_grad():
            outputs = self.encoder.forward(
                **{'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask})
            output = torch.mean(outputs[0], dim=1).squeeze()

            return output

    def loss(self, labels, logits):
        l = F.cross_entropy(logits, labels, reduction='sum')
        return l

    def loss_mlm(self, labels, logits):
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        l = loss_fct(logits.view(-1, self.tokenizer.tokenizer.vocab_size), labels.view(-1))
        return l

    def training_step(self, data_batch, batch_i):

        task2 = False
        if not self.hparams.comet_cn_train100k:
            if 'task_id' in data_batch:
                if data_batch['task_id'] is not None:
                    if data_batch['task_id'][0] == 2:
                        task2 = True

        if self.hparams.comet_cn_train100k:
            if type(data_batch) is not dict:
                task2 = True

        if not self.hparams.comet_cn_train100k:

            B, _, S = data_batch['input_ids'].shape

            logits = self.forward(**{
                'input_ids': data_batch['input_ids'].reshape(-1, S),
                'token_type_ids': data_batch['token_type_ids'].reshape(-1, S),
                'attention_mask': data_batch['attention_mask'].reshape(-1, S),
                'task_id': 1 if not task2 else 2,
            })
            loss_val = self.loss(data_batch['y'].reshape(-1), logits.reshape(B, -1))

            # WARNING: If your loss is a scalar, add one dimension in the beginning for multi-gpu training!
            if self.trainer.use_dp:
                loss_val = loss_val.unsqueeze(0)

            train_res_dict = {
                'logits': logits.reshape(B, -1),
                'loss': loss_val / B,
            }

            if task2:
                train_res_dict['progress'] = {
                    'loss_2': loss_val / B,
                }

        else:

            if not task2:
                B, _, S = data_batch['input_ids'].shape

                logits = self.forward(**{
                    'input_ids': data_batch['input_ids'].reshape(-1, S),
                    'token_type_ids': data_batch['token_type_ids'].reshape(-1, S),
                    'attention_mask': data_batch['attention_mask'].reshape(-1, S),
                    'task_id': 1 if not task2 else 2,
                })
                loss_val = self.loss(data_batch['y'].reshape(-1), logits.reshape(B, -1))

                # WARNING: If your loss is a scalar, add one dimension in the beginning for multi-gpu training!
                if self.trainer.use_dp:
                    loss_val = loss_val.unsqueeze(0)

                train_res_dict = {
                    'logits': logits.reshape(B, -1),
                    'loss': loss_val / B,
                }

            else:
                input_ids, lm_labels, input_mask = data_batch
                B = input_ids.shape[0]
                logits = self.forward(**{
                    'input_ids': input_ids,
                    'token_type_ids': None,
                    'attention_mask': input_mask,
                    'task_id': 2,
                })
                loss_val = self.loss_mlm(lm_labels, logits)

                # WARNING: If your loss is a scalar, add one dimension in the beginning for multi-gpu training!
                if self.trainer.use_dp:
                    loss_val = loss_val.unsqueeze(0)

                train_res_dict = {
                    'logits': logits,
                    'loss': loss_val / B,
                }

                train_res_dict['progress'] = {
                    'comet_loss': loss_val / B,
                }

        return train_res_dict

    def validation_step(self, data_batch, batch_i, dataset_idx=None):

        task2 = False
        if not self.hparams.comet_cn_train100k:
            if 'task_id' in data_batch:
                if data_batch['task_id'] is not None:
                    if data_batch['task_id'][0] == 2:
                        task2 = True

        if self.hparams.comet_cn_train100k:
            if type(data_batch) is not dict:
                task2 = True

        if not self.hparams.comet_cn_train100k:
            B, _, S = data_batch['input_ids'].shape
            # print (data_batch['task_id'], batch_i, dataset_idx)

            logits = self.forward(**{
                'input_ids': data_batch['input_ids'].reshape(-1, S),
                'token_type_ids': data_batch['token_type_ids'].reshape(-1, S),
                'attention_mask': data_batch['attention_mask'].reshape(-1, S),
                'task_id': 1 if not task2 else 2,
            })
            loss_val = self.loss(data_batch['y'].reshape(-1), logits.reshape(B, -1))

            # WARNING: If your loss is a scalar, add one dimension in the beginning for multi-gpu training!

            if self.trainer and self.trainer.use_dp:
                loss_val = loss_val.unsqueeze(0)

            return {
                'batch_logits': logits.reshape(B, -1),
                'batch_loss': loss_val,
                'batch_truth': data_batch['y'].reshape(-1)
            }

        else:
            if not task2:
                B, _, S = data_batch['input_ids'].shape
                # print (data_batch['task_id'], batch_i, dataset_idx)

                logits = self.forward(**{
                    'input_ids': data_batch['input_ids'].reshape(-1, S),
                    'token_type_ids': data_batch['token_type_ids'].reshape(-1, S),
                    'attention_mask': data_batch['attention_mask'].reshape(-1, S),
                    'task_id': 1 if not task2 else 2,
                })
                loss_val = self.loss(data_batch['y'].reshape(-1), logits.reshape(B, -1))

                # WARNING: If your loss is a scalar, add one dimension in the beginning for multi-gpu training!

                if self.trainer and self.trainer.use_dp:
                    loss_val = loss_val.unsqueeze(0)

                return {
                    'batch_logits': logits.reshape(B, -1),
                    'batch_loss': loss_val,
                    'batch_truth': data_batch['y'].reshape(-1)
                }

            else:
                input_ids, lm_labels, input_mask = data_batch
                B = input_ids.shape[0]
                logits = self.forward(**{
                    'input_ids': input_ids,
                    'token_type_ids': None,
                    'attention_mask': input_mask,
                    'task_id': 2,
                })
                loss_val = self.loss_mlm(lm_labels, logits)

                # WARNING: If your loss is a scalar, add one dimension in the beginning for multi-gpu training!
                if self.trainer.use_dp:
                    loss_val = loss_val.unsqueeze(0)

                return {
                    'batch_logits': logits,
                    'batch_loss': loss_val,
                    'batch_truth': lm_labels
                }

    def test_step(self, data_batch, batch_i, dataset_idx=None):

        task2 = False
        if not self.hparams.comet_cn_train100k:
            if 'task_id' in data_batch:
                if data_batch['task_id'] is not None:
                    if data_batch['task_id'][0] == 2:
                        task2 = True

        if self.hparams.comet_cn_train100k:
            if type(data_batch) is not dict:
                task2 = True

        if not self.hparams.comet_cn_train100k:
            B, _, S = data_batch['input_ids'].shape

            logits = self.forward(**{
                'input_ids': data_batch['input_ids'].reshape(-1, S),
                'token_type_ids': data_batch['token_type_ids'].reshape(-1, S),
                'attention_mask': data_batch['attention_mask'].reshape(-1, S),
                'task_id': 1 if not task2 else 2,
            })

            return {
                'batch_logits': logits.reshape(B, -1),
            }

        else:
            if not task2:
                B, _, S = data_batch['input_ids'].shape

                logits = self.forward(**{
                    'input_ids': data_batch['input_ids'].reshape(-1, S),
                    'token_type_ids': data_batch['token_type_ids'].reshape(-1, S),
                    'attention_mask': data_batch['attention_mask'].reshape(-1, S),
                    'task_id': 1 if not task2 else 2,
                })

                return {
                    'batch_logits': logits.reshape(B, -1),
                }
            else:
                input_ids, lm_labels, input_mask = data_batch
                logits = self.forward(**{
                    'input_ids': input_ids,
                    'token_type_ids': None,
                    'attention_mask': input_mask,
                    'task_id': 2,
                })

                return {
                    'batch_logits': logits,
                }

    def validation_end(self, outputs):
        multi_dataset = False
        if type(outputs[0]) == list:
            multi_dataset = True

        if multi_dataset:
            truth = torch.cat([o['batch_truth'] for o in outputs[0]], dim=0).reshape(-1)
            logits = torch.cat([o['batch_logits'] for o in outputs[0]], dim=0).reshape(truth.shape[0],
                                                                                       outputs[0][0][
                                                                                           'batch_logits'].shape[1])
            loss_sum = torch.cat([o['batch_loss'].reshape(-1) for o in outputs[0]], dim=0).reshape(-1)
            loss_sum = torch.sum(loss_sum, dim=0).reshape(-1)

            assert truth.shape[0] == sum([o['batch_logits'].shape[0] for o in outputs[0]]), "Mismatch size"

            loss = self.loss(truth, logits)

            assert math.isclose(loss.item(), loss_sum.item(),
                                abs_tol=0.01), f"Loss not equal: {loss.item()} VS. {loss_sum.item()}"

            loss /= truth.shape[0]
            loss_sum /= truth.shape[0]

            proba = F.softmax(logits, dim=-1)
            pred = torch.argmax(proba, dim=-1).reshape(-1)
        else:
            truth = torch.cat([o['batch_truth'] for o in outputs], dim=0).reshape(-1)
            logits = torch.cat([o['batch_logits'] for o in outputs], dim=0).reshape(truth.shape[0],
                                                                                    outputs[0]['batch_logits'].shape[1])
            loss_sum = torch.cat([o['batch_loss'].reshape(-1) for o in outputs], dim=0).reshape(-1)
            loss_sum = torch.sum(loss_sum, dim=0).reshape(-1)

            assert truth.shape[0] == sum([o['batch_logits'].shape[0] for o in outputs]), "Mismatch size"

            loss = self.loss(truth, logits)

            assert math.isclose(loss.item(), loss_sum.item(),
                                abs_tol=0.01), f"Loss not equal: {loss.item()} VS. {loss_sum.item()}"

            loss /= truth.shape[0]
            loss_sum /= truth.shape[0]

            proba = F.softmax(logits, dim=-1)
            pred = torch.argmax(proba, dim=-1).reshape(-1)

        if multi_dataset and not self.hparams.comet_cn_train100k:
            truth2 = torch.cat([o['batch_truth'] for o in outputs[1]], dim=0).reshape(-1)
            logits2 = torch.cat([o['batch_logits'] for o in outputs[1]], dim=0).reshape(truth2.shape[0],
                                                                                        outputs[1][0][
                                                                                            'batch_logits'].shape[1])
            loss_sum2 = torch.cat([o['batch_loss'].reshape(-1) for o in outputs[1]], dim=0).reshape(-1)
            loss_sum2 = torch.sum(loss_sum2, dim=0).reshape(-1)

            assert truth2.shape[0] == sum([o['batch_logits'].shape[0] for o in outputs[1]]), "Mismatch size"

            loss2 = self.loss(truth2, logits2)

            assert math.isclose(loss2.item(), loss_sum2.item(),
                                abs_tol=0.01), f"Loss not equal: {loss.item()} VS. {loss_sum.item()}"

            loss2 /= truth2.shape[0]
            loss_sum2 /= truth2.shape[0]

            proba2 = F.softmax(logits2, dim=-1)
            pred2 = torch.argmax(proba2, dim=-1).reshape(-1)

        elif multi_dataset and self.hparams.comet_cn_train100k:
            # truth2 = torch.cat([o['batch_truth'] for o in outputs[1]], dim=0).reshape(-1)
            # truth2 = torch.cat([o['batch_truth'] for o in outputs[1]], dim=0)
            # bz = outputs[1][0]['batch_truth'].shape[0]
            # logits2 = torch.cat([o['batch_logits'] for o in outputs[1]], dim=0)
            loss_sum2 = torch.cat([o['batch_loss'].reshape(-1) for o in outputs[1]], dim=0).reshape(-1)
            length = loss_sum2.shape[0]
            loss_sum2 = torch.sum(loss_sum2, dim=0).reshape(-1)
            loss2 = loss_sum2

            loss2 /= length

            # proba2 = F.softmax(logits2, dim=-1)
            # pred2 = torch.argmax(proba2, dim=-1)

        with open(os.path.join(self.hparams.output_dir, "dev-labels.lst"), "w") as output_file:
            output_file.write("\n".join(map(str, (truth + self.task_config[self.hparams.task_name][
                'label_offset']).cpu().numpy().tolist())))

        with open(os.path.join(self.hparams.output_dir, "dev-predictions.lst"), "w") as output_file:
            output_file.write("\n".join(
                map(str, (pred + self.task_config[self.hparams.task_name]['label_offset']).cpu().numpy().tolist())))

        with open(os.path.join(self.hparams.output_dir, "dev-probabilities.lst"), "w") as output_file:
            output_file.write("\n".join(map(lambda l: '\t'.join(map(str, l)), proba.cpu().detach().numpy().tolist())))

        if multi_dataset and not self.hparams.comet_cn_train100k:
            with open(os.path.join(self.hparams.output_dir2, "dev-labels.lst"), "w") as output_file2:
                output_file2.write("\n".join(map(str, (truth2 + self.task_config[self.hparams.task_name2][
                    'label_offset']).cpu().numpy().tolist())))

            with open(os.path.join(self.hparams.output_dir2, "dev-predictions.lst"), "w") as output_file2:
                output_file2.write("\n".join(
                    map(str,
                        (pred2 + self.task_config[self.hparams.task_name2]['label_offset']).cpu().numpy().tolist())))

            with open(os.path.join(self.hparams.output_dir2, "dev-probabilities.lst"), "w") as output_file2:
                output_file2.write(
                    "\n".join(map(lambda l: '\t'.join(map(str, l)), proba2.cpu().detach().numpy().tolist())))

        stats = []
        predl = pred.cpu().detach().numpy().tolist()
        truthl = truth.cpu().detach().numpy().tolist()

        for _ in range(100):
            predl = pred.cpu().detach().numpy().tolist()

            indicies = np.random.randint(len(predl), size=len(predl))
            sampled_pred = [predl[i] for i in indicies]
            sampled_truth = [truthl[i] for i in indicies]
            stats.append(accuracy_score(sampled_truth, sampled_pred))

        _, lower, upper = mean_confidence_interval(stats, self.hparams.ci_alpha)

        if multi_dataset and not self.hparams.comet_cn_train100k:
            stats2 = []
            predl2 = pred2.cpu().detach().numpy().tolist()
            truthl2 = truth2.cpu().detach().numpy().tolist()

            for _ in range(10000):
                predl2 = pred2.cpu().detach().numpy().tolist()

                indicies2 = np.random.randint(len(predl2), size=len(predl2))
                sampled_pred2 = [predl2[i] for i in indicies2]
                sampled_truth2 = [truthl2[i] for i in indicies2]
                stats2.append(accuracy_score(sampled_truth2, sampled_pred2))

            _2, lower2, upper2 = mean_confidence_interval(stats2, self.hparams.ci_alpha)

        result_dict = {
            'val_loss': loss.item(),
            'val_acc': accuracy_score(truth.cpu().detach().numpy().tolist(), pred.cpu().detach().numpy().tolist()),
            'val_cil': lower,
            'val_ciu': upper,
        }

        if multi_dataset and not self.hparams.comet_cn_train100k:
            result_dict['val_loss2'] = loss2.item()
            result_dict['val_acc2'] = accuracy_score(truth2.cpu().detach().numpy().tolist(),
                                                     pred2.cpu().detach().numpy().tolist())
            result_dict['val_cil2'] = lower2
            result_dict['val_ciu2'] = upper2

        elif multi_dataset and self.hparams.comet_cn_train100k:
            result_dict['comet_loss'] = loss2.item()
            # print (loss2.item(), loss2.cpu().item(), np.exp(loss2.cpu().item())); raise
            ppl = np.exp(loss2.cpu()) if loss2.item() < 300 else np.inf
            result_dict['comet_ppl'] = ppl

        return result_dict

    def test_end(self, outputs):
        """
        Outputs has the appended output after each test step
        OPTIONAL
        :param outputs:
        :return: dic_with_metrics for tqdm
        """
        multi_dataset = False
        if type(outputs[0]) == list:
            multi_dataset = True

        if multi_dataset:
            logits = torch.cat([o[0]['batch_logits'] for o in outputs], dim=0).reshape(-1, outputs[0][0][
                'batch_logits'].shape[1])
            proba = F.softmax(logits, dim=-1)
            pred = torch.argmax(proba, dim=-1).reshape(-1)
        else:
            logits = torch.cat([o['batch_logits'] for o in outputs], dim=0).reshape(-1,
                                                                                    outputs[0]['batch_logits'].shape[1])
            proba = F.softmax(logits, dim=-1)
            pred = torch.argmax(proba, dim=-1).reshape(-1)

        with open(os.path.join(self.hparams.output_dir, "predictions.lst"), "w") as output_file:
            output_file.write("\n".join(map(str, (pred + self.task_config[self.hparams.task_name][
                'label_offset']).cpu().detach().numpy().tolist())))

        with open(os.path.join(self.hparams.output_dir, "probabilities.lst"), "w") as output_file:
            output_file.write("\n".join(map(lambda l: '\t'.join(map(str, l)), proba.cpu().detach().numpy().tolist())))

        if multi_dataset and not self.hparams.comet_cn_train100k:
            logits2 = torch.cat([o[1]['batch_logits'] for o in outputs], dim=0).reshape(-1, outputs[0][1][
                'batch_logits'].shape[1])
            proba2 = F.softmax(logits2, dim=-1)
            pred2 = torch.argmax(proba2, dim=-1).reshape(-1)

            with open(os.path.join(self.hparams.output_dir2, "predictions.lst"), "w") as output_file2:
                output_file2.write("\n".join(map(str, (pred2 + self.task_config[self.hparams.task_name2][
                    'label_offset']).cpu().detach().numpy().tolist())))

            with open(os.path.join(self.hparams.output_dir2, "probabilities.lst"), "w") as output_file2:
                output_file2.write(
                    "\n".join(map(lambda l: '\t'.join(map(str, l)), proba2.cpu().detach().numpy().tolist())))

        return {}

    def configure_optimizers(self):

        # Prepare optimizer and schedule (linear warmup and decay)
        t_total = len(self.train_dataloader) // self.hparams.accumulate_grad_batches * self.hparams.max_nb_epochs

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=t_total)

        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        dataset_name = "train"
        cache_dirs = download(self.task_config[self.hparams.task_name]['urls'], self.hparams.task_cache_dir)
        dataset = ClassificationDataset.load(cache_dir=cache_dirs[0] if isinstance(cache_dirs, list) else cache_dirs,
                                             file_mapping=self.task_config[self.hparams.task_name]['file_mapping'][
                                                 dataset_name],
                                             task_formula=self.task_config[self.hparams.task_name]['task_formula'],
                                             type_formula=self.task_config[self.hparams.task_name]['type_formula'],
                                             preprocessor=self.tokenizer,
                                             pretokenized=self.task_config[self.hparams.task_name].get('pretokenized',
                                                                                                       False),
                                             label_formula=self.task_config[self.hparams.task_name].get('label_formula',
                                                                                                        None),
                                             label_offset=self.task_config[self.hparams.task_name].get('label_offset',
                                                                                                       0),
                                             label_transform=self.task_config[self.hparams.task_name].get(
                                                 'label_transform', None),
                                             shuffle=self.task_config[self.hparams.task_name].get('shuffle', False),
                                             )

        if self.hparams.task_name2 is not None:
            cache_dirs2 = download(self.task_config[self.hparams.task_name2]['urls'], self.hparams.task_cache_dir)
            dataset2 = ClassificationDataset.load(
                cache_dir=cache_dirs2[0] if isinstance(cache_dirs2, list) else cache_dirs2,
                file_mapping=self.task_config[self.hparams.task_name2]['file_mapping'][dataset_name],
                task_formula=self.task_config[self.hparams.task_name2]['task_formula'],
                type_formula=self.task_config[self.hparams.task_name2]['type_formula'],
                preprocessor=self.tokenizer,
                pretokenized=self.task_config[self.hparams.task_name2].get('pretokenized', False),
                label_formula=self.task_config[self.hparams.task_name2].get('label_formula', None),
                label_offset=self.task_config[self.hparams.task_name2].get('label_offset', 0),
                label_transform=self.task_config[self.hparams.task_name2].get('label_transform', None),
                shuffle=self.task_config[self.hparams.task_name2].get('shuffle', False),
                task_id=2)

            dataloader = DataLoader(dataset, collate_fn=self.collate_fn,
                                    shuffle=True, batch_size=self.hparams.batch_size)
            dataloader2 = DataLoader(dataset2, collate_fn=self.collate_fn,
                                     shuffle=True, batch_size=self.hparams.batch_size)
            dataloaders = [dataloader, dataloader2]
            multidatasets = MultiTaskDataset(dataloaders)
            multi_dataloader = DataLoader(multidatasets,
                                          collate_fn=lambda examples: examples[0],
                                          shuffle=True, batch_size=1)
            return multi_dataloader

        if self.hparams.comet_cn_train100k:
            from mcs.comet_train_masked import pre_process_datasets
            from mcs.conceptnet_utils import load_comet_dataset
            from mcs.data_utils import tokenize_and_encode
            from torch.utils.data import TensorDataset

            mask_token_id = self.tokenizer.tokenizer.encode(self.tokenizer.tokenizer.mask_token)[0]
            cn_train_dataset = load_comet_dataset('mcs/train100k.txt',
                                                  rel_lang=True, sep=True, prefix="<s>")
            cn_encoded_datasets = tokenize_and_encode([cn_train_dataset], self.tokenizer.tokenizer)
            cn_tensor_datasets = pre_process_datasets(cn_encoded_datasets, self.cn_input_length,
                                                      self.max_e1, self.max_r, self.max_e2, mask_parts='e2',
                                                      mask_token=mask_token_id)
            cn_train_tensor_dataset = cn_tensor_datasets[0]
            cn_train_data = TensorDataset(*cn_train_tensor_dataset)
            cn_train_dataloader = DataLoader(cn_train_data, batch_size=self.hparams.batch_size, shuffle=True)

            dataloader = DataLoader(dataset, collate_fn=self.collate_fn,
                                    shuffle=True, batch_size=self.hparams.batch_size)

            dataloaders = [dataloader, cn_train_dataloader]
            multidatasets = MultiTaskDataset(dataloaders)
            multi_dataloader = DataLoader(multidatasets,
                                          collate_fn=lambda examples: examples[0],
                                          shuffle=True, batch_size=1)
            return multi_dataloader

        return DataLoader(dataset,
                          collate_fn=self.collate_fn,
                          shuffle=True, batch_size=self.hparams.batch_size)

    def collate_fn(self, examples):
        """Padding examples into a batch."""

        padding_value = self.tokenizer.pad

        tokens = []
        input_ids = []
        token_type_ids = []
        attention_mask = []
        y = None
        task_ids = None

        for example in examples:

            # print(examples)

            tokens.append(example['tokens'])
            example_input_ids = pad_sequence(
                [torch.from_numpy(np.asarray(x)) for x in example['input_ids']],
                batch_first=True, padding_value=padding_value).long()
            example_token_type_ids = pad_sequence(
                [torch.from_numpy(np.asarray(x)) for x in example['token_type_ids']],
                batch_first=True, padding_value=padding_value).long()
            example_attention_mask = pad_sequence(
                [torch.from_numpy(np.asarray(x)) for x in example['attention_mask']],
                batch_first=True, padding_value=padding_value).long()

            input_ids.append(example_input_ids[..., :self.hparams.max_seq_len].transpose(0, 1))
            token_type_ids.append(example_token_type_ids[..., :self.hparams.max_seq_len].transpose(0, 1))
            attention_mask.append(example_attention_mask[..., :self.hparams.max_seq_len].transpose(0, 1))
            if example['y'] is not None:
                y = [example['y']] if y is None else y + [example['y']]
            if 'task_id' in example:
                task_ids = [example['task_id']] if task_ids is None else task_ids + [example['task_id']]

        return {
            'tokens': tokens,
            'input_ids': pad_sequence(input_ids, batch_first=True, padding_value=padding_value).transpose(1, 2),
            'token_type_ids': pad_sequence(token_type_ids, batch_first=True, padding_value=padding_value).transpose(1,
                                                                                                                    2),
            'attention_mask': pad_sequence(attention_mask, batch_first=True, padding_value=padding_value).transpose(1,
                                                                                                                    2),
            'y': y if y is None else torch.from_numpy(np.asarray(y)).long(),
            'task_id': task_ids if task_ids is None else torch.from_numpy(np.asarray(task_ids)).long(),
        }

    @pl.data_loader
    def val_dataloader(self, sampling=False):
        dataset_name = "dev"
        cache_dirs = download(self.task_config[self.hparams.task_name]['urls'], self.hparams.task_cache_dir)
        dataset = ClassificationDataset.load(cache_dir=cache_dirs[-1] if isinstance(cache_dirs, list) else cache_dirs,
                                             file_mapping=self.task_config[self.hparams.task_name]['file_mapping'][
                                                 dataset_name],
                                             task_formula=self.task_config[self.hparams.task_name]['task_formula'],
                                             type_formula=self.task_config[self.hparams.task_name]['type_formula'],
                                             preprocessor=self.tokenizer,
                                             pretokenized=self.task_config[self.hparams.task_name].get('pretokenized',
                                                                                                       False),
                                             label_formula=self.task_config[self.hparams.task_name].get('label_formula',
                                                                                                        None),
                                             label_offset=self.task_config[self.hparams.task_name].get('label_offset',
                                                                                                       0),
                                             label_transform=self.task_config[self.hparams.task_name].get(
                                                 'label_transform',
                                                 None),
                                             shuffle=self.task_config[self.hparams.task_name].get('shuffle', False), )

        if self.hparams.task_name2 is not None:
            cache_dirs2 = download(self.task_config[self.hparams.task_name2]['urls'], self.hparams.task_cache_dir)
            dataset2 = ClassificationDataset.load(
                cache_dir=cache_dirs2[-1] if isinstance(cache_dirs2, list) else cache_dirs2,
                file_mapping=self.task_config[self.hparams.task_name2]['file_mapping'][dataset_name],
                task_formula=self.task_config[self.hparams.task_name2]['task_formula'],
                type_formula=self.task_config[self.hparams.task_name2]['type_formula'],
                preprocessor=self.tokenizer,
                pretokenized=self.task_config[self.hparams.task_name2].get('pretokenized', False),
                label_formula=self.task_config[self.hparams.task_name2].get('label_formula', None),
                label_offset=self.task_config[self.hparams.task_name2].get('label_offset', 0),
                label_transform=self.task_config[self.hparams.task_name2].get('label_transform',
                                                                              None),
                shuffle=self.task_config[self.hparams.task_name2].get('shuffle', False),
                task_id=2, )

        if not sampling:
            dataloader = DataLoader(dataset,
                                    collate_fn=self.collate_fn,
                                    shuffle=False, batch_size=self.hparams.batch_size)
        else:
            dataloader = DataLoader(dataset, collate_fn=self.collate_fn,
                                    sampler=RandomSampler(dataset, replacement=True),
                                    shuffle=False, batch_size=self.hparams.batch_size)

        if self.hparams.comet_cn_train100k:
            from mcs.comet_train_masked import pre_process_datasets
            from mcs.conceptnet_utils import load_comet_dataset
            from mcs.data_utils import tokenize_and_encode
            from torch.utils.data import TensorDataset

            mask_token_id = self.tokenizer.tokenizer.encode(self.tokenizer.tokenizer.mask_token)[0]
            cn_val_dataset = load_comet_dataset('mcs/dev1.txt',
                                                rel_lang=True, sep=True, prefix="<s>")
            cn_encoded_datasets = tokenize_and_encode([cn_val_dataset], self.tokenizer.tokenizer)
            cn_tensor_datasets = pre_process_datasets(cn_encoded_datasets, self.cn_input_length,
                                                      self.max_e1, self.max_r, self.max_e2, mask_parts='e2',
                                                      mask_token=mask_token_id)
            cn_val_tensor_dataset = cn_tensor_datasets[0]
            cn_val_data = TensorDataset(*cn_val_tensor_dataset)
            cn_val_dataloader = DataLoader(cn_val_data, batch_size=self.hparams.batch_size, shuffle=True)
            return [dataloader, cn_val_dataloader]

        if self.hparams.task_name2 is not None:
            if not sampling:
                dataloader2 = DataLoader(dataset2,
                                         collate_fn=self.collate_fn,
                                         shuffle=False, batch_size=self.hparams.batch_size)
            else:
                dataloader2 = DataLoader(dataset2, collate_fn=self.collate_fn,
                                         sampler=RandomSampler(dataset2, replacement=True),
                                         shuffle=False, batch_size=self.hparams.batch_size)
            return [dataloader, dataloader2]

        return dataloader

    @pl.data_loader
    def test_dataloader(self):

        if self.hparams.test_input_dir is None:
            return self.val_dataloader

        dataset = ClassificationDataset.load(cache_dir=self.hparams.test_input_dir,
                                             file_mapping={'input_x': None},
                                             task_formula=self.task_config[self.hparams.task_name]['task_formula'],
                                             type_formula=self.task_config[self.hparams.task_name]['type_formula'],
                                             preprocessor=self.tokenizer,
                                             pretokenized=self.task_config[self.hparams.task_name].get('pretokenized',
                                                                                                       False),
                                             label_formula=self.task_config[self.hparams.task_name].get('label_formula',
                                                                                                        None),
                                             label_offset=self.task_config[self.hparams.task_name].get('label_offset',
                                                                                                       0),
                                             label_transform=self.task_config[self.hparams.task_name].get(
                                                 'label_transform',
                                                 None),
                                             shuffle=self.task_config[self.hparams.task_name].get('shuffle', False), )

        if self.hparams.task_name2 is not None:
            dataset2 = ClassificationDataset.load(cache_dir=self.hparams.test_input_dir2,
                                                  file_mapping={'input_x': None},
                                                  task_formula=self.task_config[self.hparams.task_name2][
                                                      'task_formula'],
                                                  type_formula=self.task_config[self.hparams.task_name2][
                                                      'type_formula'],
                                                  preprocessor=self.tokenizer,
                                                  pretokenized=self.task_config[self.hparams.task_name2].get(
                                                      'pretokenized', False),
                                                  label_formula=self.task_config[self.hparams.task_name2].get(
                                                      'label_formula', None),
                                                  label_offset=self.task_config[self.hparams.task_name2].get(
                                                      'label_offset', 0),
                                                  label_transform=self.task_config[self.hparams.task_name2].get(
                                                      'label_transform',
                                                      None),
                                                  shuffle=self.task_config[self.hparams.task_name2].get('shuffle',
                                                                                                        False),
                                                  task_id=2, )

        self.hparams.batch_size = 4
        print(self.hparams.batch_size)
        dataloader = DataLoader(dataset,
                                collate_fn=self.collate_fn,
                                shuffle=False, batch_size=self.hparams.batch_size)

        if self.hparams.comet_cn_train100k:
            from mcs.comet_train_masked import pre_process_datasets
            from mcs.conceptnet_utils import load_comet_dataset
            from mcs.data_utils import tokenize_and_encode
            from torch.utils.data import TensorDataset

            mask_token_id = self.tokenizer.tokenizer.encode(self.tokenizer.tokenizer.mask_token)[0]
            cn_val_dataset = load_comet_dataset('mcs/dev1.txt',
                                                rel_lang=True, sep=True, prefix="<s>")
            cn_encoded_datasets = tokenize_and_encode([cn_val_dataset], self.tokenizer.tokenizer)
            cn_tensor_datasets = pre_process_datasets(cn_encoded_datasets, self.cn_input_length,
                                                      self.max_e1, self.max_r, self.max_e2, mask_parts='e2',
                                                      mask_token=mask_token_id)
            cn_val_tensor_dataset = cn_tensor_datasets[0]
            cn_val_data = TensorDataset(*cn_val_tensor_dataset)
            cn_val_dataloader = DataLoader(cn_val_data, batch_size=self.hparams.batch_size, shuffle=True)
            return [dataloader, cn_val_dataloader]

        if self.hparams.task_name2 is not None:
            dataloader2 = DataLoader(dataset2,
                                     collate_fn=self.collate_fn,
                                     shuffle=False, batch_size=self.hparams.batch_size)
            return [dataloader, dataloader2]

        return dataloader

    @classmethod
    def load_from_metrics(cls, hparams, weights_path, tags_csv, on_gpu, map_location=None):

        prev_hparams = load_hparams_from_tags_csv(tags_csv)
        prev_hparams.__dict__.update(hparams.__dict__)
        hparams.__dict__.update({k: v for k, v in prev_hparams.__dict__.items() if k not in hparams.__dict__})
        hparams.__setattr__('on_gpu', on_gpu)

        if on_gpu:
            if map_location is not None:
                checkpoint = torch.load(weights_path, map_location=map_location)
            else:
                checkpoint = torch.load(weights_path)
        else:
            checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)

        running_config = yaml.safe_load(open(hparams.running_config_file, "r"))
        task_config = yaml.safe_load(open(hparams.task_config_file, 'r'))

        default_parameter = partial(get_default_hyperparameter, config=running_config,
                                    task_name=hparams.task_name, model_type=hparams.model_type,
                                    model_weight=hparams.model_weight)

        hparams.max_nb_epochs = default_parameter(field='max_nb_epochs')
        hparams.learning_rate = float(default_parameter(field='lr'))
        hparams.initializer_range = float(default_parameter(field='initializer_range'))
        hparams.dropout = float(default_parameter(field='dropout'))
        hparams.batch_size = default_parameter(field='batch_size')
        hparams.max_seq_len = default_parameter(field='max_seq_len')
        hparams.seed = default_parameter(field='seed')
        hparams.weight_decay = float(default_parameter(field='weight_decay'))
        hparams.warmup_steps = default_parameter(field='warmup_steps')
        hparams.adam_epsilon = float(default_parameter(field='adam_epsilon'))
        hparams.accumulate_grad_batches = default_parameter(field='accumulate_grad_batches')

        hparams.do_lower_case = task_config[hparams.task_name].get('do_lower_case', False)
        hparams.tokenizer_type = hparams.model_type if hparams.tokenizer_type is None else hparams.tokenizer_type
        hparams.tokenizer_weight = hparams.model_weight if hparams.tokenizer_weight is None else hparams.tokenizer_weight

        set_seed(hparams.seed)

        model = cls(hparams)
        model.load_state_dict(checkpoint['state_dict'])

        model.on_load_checkpoint(checkpoint)

        return model

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover

        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser], add_help=False)

        # param overwrites
        parser.set_defaults(gradient_clip_val=1.0,
                            model_save_monitor_value='val_acc',
                            model_save_monitor_mode='max',
                            early_stop_metric='val_loss',
                            early_stop_patience=10,
                            early_stop_mode='min',
                            val_check_interval=0.02,
                            max_nb_epochs=3
                            )

        running_group = parser.add_argument_group(title='Training/Evaluation options')
        model_group = parser.add_argument_group(title='Model options')
        tokenizer_group = parser.add_argument_group(title='Tokenizer options')
        task_group = parser.add_argument_group(title='Task options')

        # Add arguments to those groups

        model_group.add_argument('--model_type', type=str, required=True)
        model_group.add_argument('--model_weight', type=str, required=True)
        model_group.add_argument('--ci_alpha', type=float, default=0.95)

        tokenizer_group.add_argument('--tokenizer_type', type=str, default=None)
        tokenizer_group.add_argument('--tokenizer_weight', type=str, default=None)

        task_group.add_argument('--task_name',
                                choices=['alphanli', 'snli', 'hellaswag', 'physicaliqa', 'physicaliqa-10pc', 'physicaliqa-25pc', 'socialiqa',
                                         'vcrqa', 'vcrqr', 'physicaliqa-carved', 'physicaliqa-carved-25pc'],
                                required=True)
        task_group.add_argument('--task_name2', default=None,
                                choices=['atomic_attr_qa_random_name',
                                         'atomic_attr_qa',
                                         'atomic_which_one_qa',
                                         'atomic_temporal_qa',
                                         'cn_all_cs',
                                         'cn_all_cs_10k',
                                         'cn_all_cs_10k_2',
                                         'cn_all_cs_20k',
                                         'cn_all_cs_20k_2',
                                         'cn_all_cs_40k',
                                         'cn_all_cs_40k_2',
                                         'cn_all_cs_50k',
                                         'cn_all_cs_30k', 'cn_physical_cs_relaxed', 'cn_physical_cs_narrow',
                                         'cn_all_cs_10k',
                                         'cn_all_cs_20k',
                                         'cn_physical_10k',
                                         'cn_carved_10k',
                                         ],
                                required=False)
        task_group.add_argument('--task2_separate_fc', type=bool, required=False, default=False)
        task_group.add_argument('--comet_cn_train100k', type=bool, required=False, default=False)
        task_group.add_argument('--task_config_file', type=str, required=True)
        task_group.add_argument('--task_cache_dir', type=str, required=True)

        running_group.add_argument('--running_config_file', type=str, required=True)

        parser.add_argument('--experiment_name', type=str, required=True, default=None)
        parser.add_argument('--test_input_dir', type=str, required=False, default=None)
        parser.add_argument('--output_dir', type=str, required=False, default=None)
        parser.add_argument('--output_dir2', type=str, required=False, default=None)
        parser.add_argument('--weights_path', type=str, required=False, default=None)
        parser.add_argument('--tags_csv', type=str, required=False, default=None)

        return parser
