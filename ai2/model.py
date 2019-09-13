import os
from collections import OrderedDict
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning.root_module.root_module import LightningModule
from interface import HuggingFaceModelLoader, HuggingFaceTokenizerLoader


class HuggingFaceClassifier(LightningModule):

    def __init__(self, hparams):

        super(Classifier, self).__init__()
        self.hparams = hparams
        self.build_model()

    def build_model(self):

        self.model = HuggingFaceModelLoader(self.hparams.model_type, self.hparams.model_weight)
        self.dropout = nn.dropout(self.hparams.dropout)
        self.linear = nn.Linear(self.model.dim, 1)

    def forward(self, **kargs):

        output = self.model.forward(**kargs)
        logits = self.dropout(output)
        logits = self.linear(logits)

        return logits

    def loss(self, labels, logits):
        l = F.cross_entropy(logits, labels)
        return l

    def training_step(self, data_batch, batch_i):
        pass

    def validation_step(self, data_batch, batch_i):
        pass

    def validation_end(self, outputs):
        pass

    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    @pl.data_loader
    def tng_dataloader(self):
        pass

    @pl.data_loader
    def val_dataloader(self):
        pass

    @pl.data_loader
    def test_dataloader(self):
        pass

    @staticmethod
    def add_model_specific_args(parent_parser):
        pass
