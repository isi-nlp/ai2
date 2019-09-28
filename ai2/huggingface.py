from __future__ import annotations

import math
import os

from typing import *

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from pytorch_lightning.root_module.root_module import LightningModule
from transformers import *
from sklearn.metrics import accuracy_score
from test_tube import HyperOptArgumentParser
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.nn import Module

from ai2.dataset import AI2Dataset, download
from ai2.interface import *


TOKENIZERS = {
    'bert': BertTokenizer,
    'xlm': XLMTokenizer,
    'xlnet': XLNetTokenizer,
    'roberta': RobertaTokenizer,
    'gpt': OpenAIGPTTokenizer,
    'gpt2': GPT2Tokenizer
}

MODELS = {
    'bert': BertModel,
    'xlm': XLMModel,
    'xlnet': XLNetModel,
    'roberta': RobertaModel,
    'gpt': OpenAIGPTModel,
    'gpt2': GPT2Model
}


class HuggingFaceModelLoader(ModelLoader):

    def __init__(self, model: Union[Module, PreTrainedModel]):
        super(HuggingFaceModelLoader, self).__init__(model)

    def forward(self, **kwargs) -> Tuple:
        """Follow the convention of omnx, return tuple whenever possible.

        token_type_ids are used when a more is pretrained with {0, 1} token_type_ids. RoBERTa has the argument but does not support it yet.

        Returns:
            Tuple -- Tuple of returned values of forwading.
        """
        signature = getfullargspec(self.model.forward)
        return self.model.forward(
            **
            {k: None if k == "token_type_ids" and getattr(self.model.config, 'type_vocab_size', 0) < 2 else v for k, v
             in kwargs.items()
             if k in signature.args})

    @classmethod
    def load(cls, model_type: str, model_weights: str) -> HuggingFaceModelLoader:
        assert model_type in MODELS, "Model type is not recognized."
        return HuggingFaceModelLoader(MODELS[model_type].from_pretrained(model_weights, cache_dir="./model_cache"))


class HuggingFaceTokenizerLoader(TokenizerLoader):

    @classmethod
    def load(cls, model_type: str, model_weights: str, *args, **kargs) -> HuggingFaceTokenizerLoader:
        assert model_type in TOKENIZERS, f"Tokenizer model type {model_type} is not recognized."
        return HuggingFaceTokenizerLoader(
            TOKENIZERS[model_type].from_pretrained(model_weights, *args, cache_dir="./model_cache", **kargs))

    @property
    def SEP(self) -> str:
        if self.tokenizer._sep_token is None:
            return self.UNK
        return self.tokenizer._sep_token

    @property
    def sep(self) -> int:
        return self.token2id(self.SEP)

    @property
    def CLS(self) -> str:
        if self.tokenizer._cls_token is None:
            return self.UNK
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
            return self.UNK
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

        # TODO: Change to your own model loader
        self.encoder = HuggingFaceModelLoader.load(self.hparams.model_type, self.hparams.model_weight)
        self.encoder.train()

        self.dropout = nn.Dropout(self.hparams.dropout)
        self.linear = nn.Linear(self.encoder.dim, self.hparams.output_dimension)

        self.linear.weight.data.normal_(mean=0.0, std=self.hparams.initializer_range)
        self.linear.bias.data.zero_()

        # TODO: Change to your own tokenizer loader
        self.tokenizer = HuggingFaceTokenizerLoader.load(
            self.hparams.tokenizer_type, self.hparams.tokenizer_weight, do_lower_case=self.hparams.do_lower_case)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):

        # if input_ids is not None and token_type_ids is not None and attention_mask is not None:
        #     logger.debug(f"Device: {next(self.encoder.model.parameters()).device}")
        #     logger.debug(f"Device: {input_ids.device} {token_type_ids.device} {attention_mask.device}")

        # TODO [Optional]: Change to your own forward
        outputs = self.encoder.forward(
            **{'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask})
        output = torch.mean(outputs[0], dim=1).squeeze()
        logits = self.dropout(output)
        logits = self.linear(logits)

        return logits.squeeze()

    def loss(self, labels, logits):
        l = F.cross_entropy(logits, labels, reduction='sum')
        return l

    def training_step(self, data_batch, batch_i):

        B, C, S = data_batch['input_ids'].shape

        logits = self.forward(**{
            'input_ids': data_batch['input_ids'].reshape(-1, S),
            'token_type_ids': data_batch['token_type_ids'].reshape(-1, S),
            'attention_mask': data_batch['attention_mask'].reshape(-1, S),
        })
        loss_val = self.loss(data_batch['y'].reshape(-1), logits.reshape(B, -1))

        proba = F.softmax(logits, dim=-1)
        pred = torch.argmax(proba, dim=-1).reshape(-1)

        # WARNING: If your loss is a scalar, add one dimension in the beginning for multi-gpu training!
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)

        return {
            'logits': logits.reshape(B, -1),
            'loss': loss_val / B,
        }

    def validation_step(self, data_batch, batch_i):
        B, C, S = data_batch['input_ids'].shape

        logits = self.forward(**{
            'input_ids': data_batch['input_ids'].reshape(-1, S),
            'token_type_ids': data_batch['token_type_ids'].reshape(-1, S),
            'attention_mask': data_batch['attention_mask'].reshape(-1, S),
        })
        loss_val = self.loss(data_batch['y'].reshape(-1), logits.reshape(B, -1))

        # WARNING: If your loss is a scalar, add one dimension in the beginning for multi-gpu training!

        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)

        return {
            'batch_logits': logits.reshape(B, -1),
            'batch_loss': loss_val,
            'batch_truth': data_batch['y'].reshape(-1)
        }

    def test_step(self, data_batch, batch_i):
        B, C, S = data_batch['input_ids'].shape

        logits = self.forward(**{
            'input_ids': data_batch['input_ids'].reshape(-1, S),
            'token_type_ids': data_batch['token_type_ids'].reshape(-1, S),
            'attention_mask': data_batch['attention_mask'].reshape(-1, S),
        })

        return {
            'batch_logits': logits.reshape(B, -1),
        }

    def validation_end(self, outputs):

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

        with open(os.path.join(self.hparams.output_dir, "dev-labels.lst"), "w") as output_file:
            output_file.write("\n".join(map(str, (truth + self.task_config[self.hparams.task_name][
                'label_offset']).cpu().numpy().tolist())))

        with open(os.path.join(self.hparams.output_dir, "dev-predictions.lst"), "w") as output_file:
            output_file.write("\n".join(
                map(str, (pred + self.task_config[self.hparams.task_name]['label_offset']).cpu().numpy().tolist())))

        with open(os.path.join(self.hparams.output_dir, "dev-probabilities.lst"), "w") as output_file:
            output_file.write("\n".join(map(lambda l: '\t'.join(map(str, l)), proba.cpu().detach().numpy().tolist())))

        return {
            'val_loss': loss.item(),
            'val_acc': accuracy_score(truth.cpu().detach().numpy().tolist(), pred.cpu().detach().numpy().tolist()),
        }

    def test_end(self, outputs):
        """
        Outputs has the appended output after each test step
        OPTIONAL
        :param outputs:
        :return: dic_with_metrics for tqdm
        """
        logits = torch.cat([o['batch_logits'] for o in outputs], dim=0).reshape(-1, outputs[0]['batch_logits'].shape[1])
        proba = F.softmax(logits, dim=-1)
        pred = torch.argmax(proba, dim=-1).reshape(-1)

        with open(os.path.join(self.hparams.output_dir, "predictions.lst"), "w") as output_file:
            output_file.write("\n".join(map(str, (pred + self.task_config[self.hparams.task_name][
                'label_offset']).cpu().detach().numpy().tolist())))

        with open(os.path.join(self.hparams.output_dir, "probabilities.lst"), "w") as output_file:
            output_file.write("\n".join(map(lambda l: '\t'.join(map(str, l)), proba.cpu().detach().numpy().tolist())))

        return {}

    def configure_optimizers(self):

        # Prepare optimizer and schedule (linear warmup and decay)
        t_total = len(self.tng_dataloader) // self.hparams.accumulate_grad_batches * self.hparams.max_nb_epochs

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=self.hparams.warmup_steps, t_total=t_total)

        return [optimizer], [scheduler]

    @pl.data_loader
    def tng_dataloader(self):
        dataset_name = "train"
        cache_dirs = download(self.task_config[self.hparams.task_name]['urls'], self.hparams.task_cache_dir)
        dataset = AI2Dataset.load(cache_dir=cache_dirs[0] if isinstance(cache_dirs, list) else cache_dirs,
                                  file_mapping=self.task_config[self.hparams.task_name]['file_mapping'][dataset_name],
                                  task_formula=self.task_config[self.hparams.task_name]['task_formula'],
                                  type_formula=self.task_config[self.hparams.task_name]['type_formula'],
                                  preprocessor=self.tokenizer,
                                  pretokenized=self.task_config[self.hparams.task_name].get('pretokenized', False),
                                  label_formula=self.task_config[self.hparams.task_name].get('label_formula', None),
                                  label_offset=self.task_config[self.hparams.task_name].get('label_offset', 0),
                                  label_transform=self.task_config[self.hparams.task_name].get('label_transform', None),
                                  )

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

        return {
            'tokens': tokens,
            'input_ids': pad_sequence(input_ids, batch_first=True, padding_value=padding_value).transpose(1, 2),
            'token_type_ids': pad_sequence(token_type_ids, batch_first=True, padding_value=padding_value).transpose(1,
                                                                                                                    2),
            'attention_mask': pad_sequence(attention_mask, batch_first=True, padding_value=padding_value).transpose(1,
                                                                                                                    2),
            'y': y if y is None else torch.from_numpy(np.asarray(y)).long(),
        }

    @pl.data_loader
    def val_dataloader(self):
        dataset_name = "dev"
        cache_dirs = download(self.task_config[self.hparams.task_name]['urls'], self.hparams.task_cache_dir)
        # TODO: improve download file mapping
        dataset = AI2Dataset.load(cache_dir=cache_dirs[-1] if isinstance(cache_dirs, list) else cache_dirs,
                                  file_mapping=self.task_config[self.hparams.task_name]['file_mapping'][dataset_name],
                                  task_formula=self.task_config[self.hparams.task_name]['task_formula'],
                                  type_formula=self.task_config[self.hparams.task_name]['type_formula'],
                                  preprocessor=self.tokenizer,
                                  pretokenized=self.task_config[self.hparams.task_name].get('pretokenized', False),
                                  label_formula=self.task_config[self.hparams.task_name].get('label_formula', None),
                                  label_offset=self.task_config[self.hparams.task_name].get('label_offset', 0),
                                  label_transform=self.task_config[self.hparams.task_name].get('label_transform',
                                                                                               None), )

        return DataLoader(dataset,
                          collate_fn=self.collate_fn,
                          shuffle=False, batch_size=self.hparams.batch_size)

    @pl.data_loader
    def test_dataloader(self):

        if self.hparams.test_input_dir is None:
            return self.val_dataloader

        dataset_name = "test"
        dataset = AI2Dataset.load(cache_dir=self.hparams.test_input_dir,
                                  file_mapping={'input_x': None},
                                  task_formula=self.task_config[self.hparams.task_name]['task_formula'],
                                  type_formula=self.task_config[self.hparams.task_name]['type_formula'],
                                  preprocessor=self.tokenizer,
                                  pretokenized=self.task_config[self.hparams.task_name].get('pretokenized', False),
                                  label_formula=self.task_config[self.hparams.task_name].get('label_formula', None),
                                  label_offset=self.task_config[self.hparams.task_name].get('label_offset', 0),
                                  label_transform=self.task_config[self.hparams.task_name].get('label_transform',
                                                                                               None), )

        return DataLoader(dataset,
                          collate_fn=self.collate_fn,
                          shuffle=False, batch_size=self.hparams.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover

        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser], add_help=False)

        # param overwrites
        parser.set_defaults(gradient_clip=1.0,
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

        tokenizer_group.add_argument('--tokenizer_type', type=str, default=None)
        tokenizer_group.add_argument('--tokenizer_weight', type=str, default=None)

        task_group.add_argument('--task_name',
                                choices=['alphanli', 'snli', 'hellaswag', 'physicaliqa', 'socialiqa', 'vcrqa', 'vcrqr'],
                                required=True)
        task_group.add_argument('--task_config_file', type=str, required=True)
        task_group.add_argument('--task_cache_dir', type=str, required=True)

        running_group.add_argument('--running_config_file', type=str, required=True)

        parser.add_argument('--test_input_dir', type=str, required=False, default=None)
        parser.add_argument('--output_dir', type=str, required=False, default=None)
        parser.add_argument('--weights_path', type=str, required=False, default=None)
        parser.add_argument('--tags_csv', type=str, required=False, default=None)

        return parser