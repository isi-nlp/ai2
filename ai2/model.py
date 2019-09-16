import os

import torch
import yaml
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import torch.nn as nn
import pandas as pd

from collections import OrderedDict

from test_tube import HyperOptArgumentParser
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning.root_module.root_module import LightningModule

from ai2.interface import HuggingFaceModelLoader, HuggingFaceTokenizerLoader
from ai2.dataset import AI2Dataset, download


class HuggingFaceClassifier(LightningModule):

    def __init__(self, hparams):

        super(HuggingFaceClassifier, self).__init__()
        self.hparams = hparams

        with open(self.hparams.task_config_file, 'r') as input_file:
            self.task_config = yaml.safe_load(input_file)

        with open(self.hparams.running_config_file, 'r') as input_file:
            self.running_config = yaml.safe_load(input_file)

        self.hparams.learning_rate = float(self.running_config.get(
            self.hparams.model_type, {}).get(
            self.hparams.model_weight, self.running_config['default']).get('lr'))

        self.hparams.initializer_range = float(self.running_config.get(
            self.hparams.model_type, {}).get(
            self.hparams.model_weight, self.running_config['default']).get('initializer_range'))

        self.hparams.dropout = float(self.running_config.get(
            self.hparams.model_type, {}).get(
            self.hparams.model_weight, self.running_config['default']).get('dropout'))

        self.hparams.batch_size = self.running_config.get(
            self.hparams.model_type, {}).get(
            self.hparams.model_weight, self.running_config['default']).get('batch_size')

        self.hparams.max_seq_len = self.running_config.get(
            self.hparams.model_type, {}).get(
            self.hparams.model_weight, self.running_config['default']).get('max_seq_len')

        self.hparams.do_lower_case = self.task_config[self.hparams.task_name].get('do_lower_case', False)

        if not os.path.exists(self.hparams.output_dir):
            os.mkdir(self.hparams.output_dir)

        self.build_model()

    def build_model(self):

        self.model = HuggingFaceModelLoader.load(self.hparams.model_type, self.hparams.model_weight)
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.linear = nn.Linear(self.model.dim, 1)

        self.linear.weight.data.normal_(mean=0.0, std=self.hparams.initializer_range)
        self.linear.bias.data.zero_()

        self.tokenizer = HuggingFaceTokenizerLoader.load(
            self.hparams.tokenizer_type, self.hparams.tokenizer_weight, do_lower_case=self.hparams.do_lower_case)

    def forward(self, **kargs):

        outputs = self.model.forward(**kargs)
        output = torch.mean(outputs[0], dim=1).squeeze()
        logits = self.dropout(output)
        logits = self.linear(logits)

        return logits.squeeze()

    def loss(self, labels, logits):
        l = F.cross_entropy(logits, labels)
        return l

    def training_step(self, data_batch, batch_i):

        B, C, S = data_batch['input_ids'].shape

        logits = self.forward(**{
            'input_ids': data_batch['input_ids'].reshape(-1, S),
            'token_type_ids': data_batch['token_type_ids'].reshape(-1, S),
            'attention_mask': data_batch['attention_mask'].reshape(-1, S),
        })

        return {
            'logits': logits.reshape(B, C),
            'loss': self.loss(data_batch['y'].reshape(-1), logits.reshape(B, C))
        }

    # def on_epoch_end(self):
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     outputs = []

    #     for i, batch in tqdm(enumerate(self.tng_dataloader), total=len(self.tng_dataloader)):
    #         for key, val in batch.items():
    #             batch[key] = val.to(device)
    #             outputs.append(self.validation_step(batch, i))

    #     logits = torch.cat([o['logits'] for o in outputs], dim=0)
    #     truth = torch.cat([o['truth'] for o in outputs], dim=0)
    #     # loss = self.loss(truth.reshape(-1), logits)
    #     proba = F.softmax(logits, dim=-1)
    #     pred = torch.argmax(proba, dim=-1).reshape(-1)

    #     with open(os.path.join(self.hparams.output_dir, "train-labels.lst"), "w") as output_file:
    #         output_file.write("\n".join(map(str,  truth.reshape(-1).cpu().numpy().tolist())))

    #     with open(os.path.join(self.hparams.output_dir, "train-predictions.lst"), "w") as output_file:
    #         output_file.write("\n".join(map(str, (pred + self.task_config[self.hparams.task_name]['label_offset']).cpu().numpy().tolist())))

    #     with open(os.path.join(self.hparams.output_dir, "train-probabilities.lst"), "w") as output_file:
    #         output_file.write("\n".join(map(lambda l: '\t'.join(l), proba.cpu().detach().numpy().tolist())))

    def validation_step(self, data_batch, batch_i):
        B, C, S = data_batch['input_ids'].shape

        logits = self.forward(**{
            'input_ids': data_batch['input_ids'].reshape(-1, S),
            'token_type_ids': data_batch['token_type_ids'].reshape(-1, S),
            'attention_mask': data_batch['attention_mask'].reshape(-1, S),
        })

        return {
            'logits': logits.reshape(B, C),
            'truth': data_batch['y'],
            'val_batch_loss': self.loss(data_batch['y'].reshape(-1), logits.reshape(B, C))
        }

    def test_step(self, data_batch, batch_i):
        B, C, S = data_batch['input_ids'].shape

        logits = self.forward(**{
            'input_ids': data_batch['input_ids'].reshape(-1, S),
            'token_type_ids': data_batch['token_type_ids'].reshape(-1, S),
            'attention_mask': data_batch['attention_mask'].reshape(-1, S),
        })

        return {
            'logits': logits.reshape(B, C),
        }

    def validation_end(self, outputs):

        logits = torch.cat([o['logits'] for o in outputs], dim=0)
        truth = torch.cat([o['truth'] for o in outputs], dim=0)
        loss = self.loss(truth.reshape(-1), logits)
        proba = F.softmax(logits, dim=-1)
        pred = torch.argmax(proba, dim=-1).reshape(-1)

        with open(os.path.join(self.hparams.output_dir, "dev-labels.lst"), "w") as output_file:
            output_file.write("\n".join(map(str,  truth.reshape(-1).cpu().numpy().tolist())))

        with open(os.path.join(self.hparams.output_dir, "dev-predictions.lst"), "w") as output_file:
            output_file.write("\n".join(map(str, (pred + self.task_config[self.hparams.task_name]['label_offset']).cpu().numpy().tolist())))

        with open(os.path.join(self.hparams.output_dir, "dev-probabilities.lst"), "w") as output_file:
            output_file.write("\n".join(map(lambda l: '\t'.join(l), proba.cpu().detach().numpy().tolist())))

        return {
            'val_loss': loss.item(),
            'val_acc': (pred == truth).sum() / len(truth)
        }

    def test_end(self, outputs):
        """
        Outputs has the appended output after each test step
        OPTIONAL
        :param outputs:
        :return: dic_with_metrics for tqdm
        """
        logits = torch.cat([o['logits'] for o in outputs], dim=0)
        proba = F.softmax(logits, dim=-1)
        pred = torch.argmax(proba, dim=-1).reshape(-1)

        with open(os.path.join(self.hparams.output_dir, "predictions.lst"), "w") as output_file:
            output_file.write("\n".join(map(str, (pred + self.task_config[self.hparams.task_name]['label_offset']).cpu().detach().numpy().tolist())))

        with open(os.path.join(self.hparams.output_dir, "probabilities.lst"), "w") as output_file:
            output_file.write("\n".join(map(lambda l: '\t'.join(l), proba.cpu().detach().numpy().tolist())))

        return {}

    def configure_optimizers(self):

        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

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
                                  label_offset=self.task_config[self.hparams.task_name].get('label_offset', 0))

        return DataLoader(dataset,
                          collate_fn=self.collate_fn,
                          shuffle=True, batch_size=self.hparams.batch_size)

    def collate_fn(self, examples):

        padding_value = self.tokenizer.pad

        tokens = []
        input_ids = []
        token_type_ids = []
        attention_mask = []
        y = None

        for example in examples:

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
            'token_type_ids': pad_sequence(token_type_ids, batch_first=True, padding_value=padding_value).transpose(1, 2),
            'attention_mask': pad_sequence(attention_mask, batch_first=True, padding_value=padding_value).transpose(1, 2),
            'y': y if y is None else torch.from_numpy(np.asarray(y)).long(),
        }

    @pl.data_loader
    def val_dataloader(self):
        dataset_name = "dev"
        cache_dirs = download(self.task_config[self.hparams.task_name]['urls'], self.hparams.task_cache_dir)
        dataset = AI2Dataset.load(cache_dir=cache_dirs[0] if isinstance(cache_dirs, list) else cache_dirs,
                                  file_mapping=self.task_config[self.hparams.task_name]['file_mapping'][dataset_name],
                                  task_formula=self.task_config[self.hparams.task_name]['task_formula'],
                                  type_formula=self.task_config[self.hparams.task_name]['type_formula'],
                                  preprocessor=self.tokenizer,
                                  pretokenized=self.task_config[self.hparams.task_name].get('pretokenized', False),
                                  label_formula=self.task_config[self.hparams.task_name].get('label_formula', None),
                                  label_offset=self.task_config[self.hparams.task_name].get('label_offset', 0))

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
                                  label_offset=self.task_config[self.hparams.task_name].get('label_offset', 0))

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
                            val_check_interval=0.05
                            )

        runing_group = parser.add_argument_group(title='Training/Evaluation options')
        model_group = parser.add_argument_group(title='Model options')
        tokenizer_group = parser.add_argument_group(title='Tokenizer options')
        task_group = parser.add_argument_group(title='Task options')

        # Add arguments to those groups

        model_group.add_argument('--model_type', type=str, required=True)
        model_group.add_argument('--model_weight', type=str, required=True)

        tokenizer_group.add_argument('--tokenizer_type', type=str, default=None)
        tokenizer_group.add_argument('--tokenizer_weight', type=str, default=None)

        task_group.add_argument('--task_name', choices=['alphanli', 'hellaswag', 'physicaliqa', 'socialiqa', 'vcrqa', 'vcrqr'], required=True)
        task_group.add_argument('--task_config_file', type=str, required=True)
        task_group.add_argument('--task_cache_dir', type=str, required=True)

        runing_group.add_argument('--running_config_file', type=str, required=True)

        parser.add_argument('--test_input_dir', type=str, required=False, default=None)
        parser.add_argument('--output_dir', type=str, required=False, default=None)
        parser.add_argument('--weights_path', type=str, required=False, default=None)
        parser.add_argument('--tags_csv', type=str, required=False, default=None)

        return parser
