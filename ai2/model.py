import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pytorch_lightning.root_module.root_module import LightningModule
from sklearn.metrics import accuracy_score
from test_tube import HyperOptArgumentParser
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from ai2.dataset import AI2Dataset, download
from ai2.interface import HuggingFaceModelLoader, HuggingFaceTokenizerLoader


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
            self.hparams.model_weight, self.running_config['default'].get('lr')))

        self.hparams.initializer_range = float(self.running_config.get(
            self.hparams.model_type, {}).get(
            self.hparams.model_weight, self.running_config['default'].get('initializer_range')))

        self.hparams.dropout = float(self.running_config.get(
            self.hparams.model_type, {}).get(
            self.hparams.model_weight, self.running_config['default'].get('dropout')))

        self.hparams.batch_size = self.running_config.get(
            self.hparams.model_type, {}).get(
            self.hparams.model_weight, self.running_config['default'].get('batch_size'))

        self.hparams.max_seq_len = self.running_config.get(
            self.hparams.model_type, {}).get(
            self.hparams.model_weight, self.running_config['default'].get('max_seq_len'))

        self.hparams.do_lower_case = self.task_config[self.hparams.task_name].get('do_lower_case', False)

        if not os.path.exists(self.hparams.output_dir):
            os.mkdir(self.hparams.output_dir)

        self.encoder = HuggingFaceModelLoader.load(self.hparams.model_type, self.hparams.model_weight)
        self.encoder.train()
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.linear = nn.Linear(self.encoder.dim, 1)

        self.linear.weight.data.normal_(mean=0.0, std=self.hparams.initializer_range)
        self.linear.bias.data.zero_()

        self.tokenizer = HuggingFaceTokenizerLoader.load(
            self.hparams.tokenizer_type, self.hparams.tokenizer_weight, do_lower_case=self.hparams.do_lower_case)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):

        # if input_ids is not None and token_type_ids is not None and attention_mask is not None:
        #     logger.debug(f"Device: {next(self.encoder.model.parameters()).device}")
        #     logger.debug(f"Device: {input_ids.device} {token_type_ids.device} {attention_mask.device}")

        outputs = self.encoder.forward(
            **{'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask})
        output = torch.mean(outputs[0], dim=1).squeeze()
        logits = self.dropout(output)
        logits = self.linear(logits)

        return logits.squeeze()

    def loss(self, labels, logits):
        l = F.cross_entropy(logits, labels, reduction='mean')
        return l

    def training_step(self, data_batch, batch_i):

        B, C, S = data_batch['input_ids'].shape

        logits = self.forward(**{
            'input_ids': data_batch['input_ids'].reshape(-1, S),
            'token_type_ids': data_batch['token_type_ids'].reshape(-1, S),
            'attention_mask': data_batch['attention_mask'].reshape(-1, S),
        })
        loss_val = self.loss(data_batch['y'].reshape(-1), logits.reshape(B, C))
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)

        return {
            'logits': logits.reshape(B, C),
            'loss': loss_val
        }

    def validation_step(self, data_batch, batch_i):
        B, C, S = data_batch['input_ids'].shape

        logits = self.forward(**{
            'input_ids': data_batch['input_ids'].reshape(-1, S),
            'token_type_ids': data_batch['token_type_ids'].reshape(-1, S),
            'attention_mask': data_batch['attention_mask'].reshape(-1, S),
        })
        loss_val = self.loss(data_batch['y'].reshape(-1), logits.reshape(B, C))
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)

        return {
            'batch_logits': logits.reshape(B, C),
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
            'batch_logits': logits.reshape(B, C),
        }

    def validation_end(self, outputs):
        truth = torch.cat([o['batch_truth'] for o in outputs], dim=0).reshape(-1)
        logits = torch.cat([o['batch_logits'] for o in outputs], dim=0).reshape(len(truth),
                                                                                outputs[0]['batch_logits'].shape[1])

        loss = self.loss(truth, logits)
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
                            val_check_interval=0.02,
                            max_nb_epochs=3
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

        task_group.add_argument('--task_name',
                                choices=['alphanli', 'hellaswag', 'physicaliqa', 'socialiqa', 'vcrqa', 'vcrqr'],
                                required=True)
        task_group.add_argument('--task_config_file', type=str, required=True)
        task_group.add_argument('--task_cache_dir', type=str, required=True)

        runing_group.add_argument('--running_config_file', type=str, required=True)

        parser.add_argument('--test_input_dir', type=str, required=False, default=None)
        parser.add_argument('--output_dir', type=str, required=False, default=None)
        parser.add_argument('--weights_path', type=str, required=False, default=None)
        parser.add_argument('--tags_csv', type=str, required=False, default=None)

        return parser
