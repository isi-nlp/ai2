# encoding: utf-8
# Created by chenghaomou at 9/7/19
# Contact: mouchenghao at gmail dot com
# Description: Model template

"""
Example template for defining a system
"""
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.root_module.root_module import LightningModule
from pytorch_transformers import *
from sklearn.metrics import accuracy_score
from test_tube import HyperOptArgumentParser
from torch import optim
from torch.utils.data import DataLoader

from ai2.interface import PyTorchTransformer, PyTorchTransformerTokenizer
from ai2.utils import AI2Dataset, Formulator, Preprocessor

MODELS = {
    'bert': BertModel,
    'xlm': XLMModel,
    'xlnet': XLNetModel,
    'roberta': RobertaModel,
    'gpt': OpenAIGPTModel,
    'gpt2': GPT2Model
}

TOKENIZERS = {
    'bert': BertTokenizer,
    'xlm': XLMTokenizer,
    'xlnet': XLNetTokenizer,
    'roberta': RobertaTokenizer,
    'gpt': OpenAIGPTTokenizer,
    'gpt2': GPT2Tokenizer

}


class Classifier(LightningModule):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super(Classifier, self).__init__()
        self.hparams = hparams
        self.formulator = Formulator({
            'premise': self.hparams.task_premise.split('+'),
            'hypotheses': self.hparams.task_hypotheses.split('|')
        })
        self.tokenizer = PyTorchTransformerTokenizer.load(TOKENIZERS[self.hparams.tokenizer_type],
                                                          self.hparams.tokenizer_weight)
        self.preprocessor = Preprocessor(self.tokenizer, self.formulator, lambda y: y - self.hparams.task_offset)

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Layout model
        :return:
        """
        self.base = PyTorchTransformer.load(MODELS[self.hparams.model_type], self.hparams.model_weight)
        self.dropout = nn.Dropout(self.hparams.train_dropout)
        self.linear = nn.Linear(self.base.dimension, 1)
        self.linear.weight.data.normal_(mean=0.0, std=self.hparams.train_initializer_range)
        self.linear.bias.data.zero_()
        # ---------------------

    # TRAINING
    # ---------------------
    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """

        B, C, S = input_ids.shape

        output = self.base.forward(input_ids=input_ids.reshape(B * C, S),
                                   token_type_ids=token_type_ids.reshape(B * C, S),
                                   attention_mask=attention_mask.reshape(B * C, S))[0]

        if len(output.shape) == 3:
            output = output.mean(dim=1).reshape((B * C, -1))

        output = self.dropout(output)
        logits = self.linear(output).reshape((-1, C))

        return logits

    def loss(self, labels, logits):
        l = F.cross_entropy(logits, labels, reduction='sum', ignore_index=self.tokenizer.pad_index)
        return l

    def training_step(self, data_batch, batch_i):
        """
        Lightning calls this inside the training loop
        :param data_batch:
        :return:
        """
        # forward pass
        input_ids, y, token_type_ids, attention_mask = data_batch['input_ids'], data_batch['y'], data_batch[
            'token_type_ids'], data_batch['attention_mask']
        y_hat = self.forward(input_ids, token_type_ids, attention_mask)
        loss_val = self.loss(y, y_hat)

        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)

        output = OrderedDict({
            'loss': loss_val
        })

        return output

    def validation_step(self, data_batch, batch_i):
        """
        Lightning calls this inside the validation loop
        :param data_batch:
        :return:
        """
        input_ids, y, token_type_ids, attention_mask = data_batch['input_ids'], data_batch['y'], data_batch[
            'token_type_ids'], data_batch['attention_mask']
        y_hat = self.forward(input_ids, token_type_ids, attention_mask)
        loss_val = self.loss(y, y_hat)

        # acc
        labels_hat = torch.argmax(y_hat, dim=-1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        output = OrderedDict({
            'batch_loss': loss_val,
            'batch_acc': val_acc,
            'truth': y,
            'pred': labels_hat,
            'prob': F.softmax(y_hat, dim=-1)
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        truth = torch.cat([output['truth'] for output in outputs], dim=0).reshape(-1).cpu().detach().numpy().tolist()
        pred = torch.cat([output['pred'] for output in outputs], dim=0).reshape(-1).cpu().detach().numpy().tolist()

        prob = torch.cat([output['prob'] for output in outputs], dim=0).cpu().detach().numpy().tolist()

        tqdm_dic = {'val_acc': accuracy_score(truth, pred)}
        return tqdm_dic

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.train_learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    @pl.data_loader
    def tng_dataloader(self):

        return DataLoader(
            AI2Dataset.load(self.hparams.dataset_cache_dir,
                            mapping={self.hparams.dataset_train_x: 'input', self.hparams.dataset_train_y: 'label'},
                            transform=self.preprocessor),
            batch_size=self.hparams.train_batch_size,
            collate_fn=self.preprocessor.collate,
            shuffle=True
        )

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            AI2Dataset.load(self.hparams.dataset_cache_dir,
                            mapping={self.hparams.dataset_dev_x: 'input', self.hparams.dataset_dev_y: 'label'},
                            transform=self.preprocessor),
            batch_size=self.hparams.train_batch_size,
            collate_fn=self.preprocessor.collate,
            shuffle=False
        )

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :return:
        """
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser], add_help=False)

        # param overwrites
        parser.set_defaults(gradient_clip=1.0)

        training_group = parser.add_argument_group(title='Training options')
        base_model_group = parser.add_argument_group(title='Model options')
        base_tokenizer_group = parser.add_argument_group(title='Tokenizer options')
        dataset_group = parser.add_argument_group(title='Dataset options')
        task_group = parser.add_argument_group(title='Task options')

        # Add arguments to those groups

        training_group.add_argument('--train_learning_rate', type=float, default=2e-5)
        training_group.add_argument('--train_batch_size', type=int, default=8)
        training_group.add_argument('--train_dropout', type=float, default=0.1)
        training_group.add_argument('--train_initializer_range', type=float, default=0.02)

        base_model_group.add_argument('--model_type', type=str, default='bert', required=True)
        base_model_group.add_argument('--model_weight', type=str, default='bert-base-cased', required=True)

        base_tokenizer_group.add_argument('--tokenizer_type', type=str, default='bert', required=True)
        base_tokenizer_group.add_argument('--tokenizer_weight', type=str, default='bert-base-cased', required=True)

        dataset_group.add_argument('--dataset_cache_dir', type=str, required=True)
        dataset_group.add_argument('--dataset_train_x', type=str, required=True)
        dataset_group.add_argument('--dataset_train_y', type=str, required=True)
        dataset_group.add_argument('--dataset_dev_x', type=str, required=True)
        dataset_group.add_argument('--dataset_dev_y', type=str, required=True)

        task_group.add_argument('--task_name', choices=['anli', 'hellaswag', 'physicaliqa', 'socialiqa'], required=True)
        task_group.add_argument('--task_premise', type=str, required=True)
        task_group.add_argument('--task_hypotheses', type=str, required=True)
        task_group.add_argument('--task_offset', type=int, required=True)

        return parser
