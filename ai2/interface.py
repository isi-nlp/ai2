from __future__ import annotations
from inspect import getfullargspec
from abc import ABC
from dataclasses import dataclass
from torch.nn import Module
from typing import Union
from pytorch_transformers.modeling_utils import PreTrainedModel
from pytorch_transformers.tokenization_utils import PreTrainedTokenizer
from inspect import getfullargspec
import abc
import os
from collections import OrderedDict
import torch.nn as nn
import torch
import torch.nn.functional as F
from test_tube import HyperOptArgumentParser
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning.root_module.root_module import LightningModule
from pytorch_transformers import *
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


@dataclass
class ModelLoader(ABC):

    model: Union[Module, PretrainedModel]

    @classmethod
    @abc.abstractmethod
    def load(cls, *args, **kwargs) -> ModelLoader:
        raise NotImplementedError('Load is not implemented.')

    def forward(self, **kwargs) -> Tuple:
        """Follow the convention of omnx, return tuple whenever possible.

        Returns:
            Tuple -- Tuple of returned values of forwading.
        """
        signature = getfullargspec(self.base.forward)
        return self.base.forward(**{k: v for k, v in kwargs.items() if k in signature.args})

    @property
    def dim(self) -> int:
        """Return the hidden dimension of the last layer.

        Returns:
            int -- Last layer's dimension.
        """
        return [p.size(0) for p in self.model.parameters()][-1]


@dataclass
class TokenizerLoader(ABC):

    tokenizer: Union[object, PretrainedTokenizer]

    @classmethod
    @abc.abstractmethod
    def load(cls, *args, **kwargs) -> TokenizerLoader:
        raise NotImplementedError("Load is not implemented.")

    @property
    @abc.abstractmethod
    def SEP(self) -> str:
        raise NotImplementedError("SEP is not implemented.")

    @property
    def sep(self) -> int:
        return self.token2id(self.SEP)

    @property
    @abc.abstractmethod
    def CLS(self) -> str:
        raise NotImplementedError("CLS is not implemented.")

    @property
    def cls(self) -> int:
        return self.token2id(self.CLS)

    @property
    @abc.abstractmethod
    def UNK(self) -> str:
        raise NotImplementedError("UNK is not implemented.")

    @property
    def unk(self) -> int:
        return self.token2id(self.UNK)

    @property
    @abc.abstractmethod
    def PAD(self) -> str:
        raise NotImplementedError("PAD is not implemented.")

    @property
    def pad(self) -> int:
        return self.token2id(self.PAD)

    @abc.abstractmethod
    def token2id(self, token: str) -> int:
        raise NotImplementedError("token2id is not implemented.")

    def tokens2ids(self, tokens: List[str]) -> List[int]:
        return [self.token2id(token) for token in tokens]

    @abc.abstractmethod
    def tokenize(self, text: str) -> List[str]:
        raise NotImplemented('tokenize is not implemented')


class HuggingFaceModelLoader(ModelLoader):

    @classmethod
    def load(cls, model_type: str, model_weights: str) -> HuggingFaceModelLoader:
        assert model_type in MODELS, "Model type is not recognized."
        return HuggingFaceModelLoader(MODELS[model_type].from_pretrained(model_weights))


class HuggingFaceTokenizerLoader(TokenizerLoader):

    @classmethod
    def load(cls, model_type: str, model_weights: str, *args, **kargs) -> HuggingFaceTokenizerLoader:
        assert model_type in TOKENIZERS, "Tokenizer model type is not recognized."
        return HuggingFaceTokenizerLoader(TOKENIZERS[model_type].from_pretrained(model_weights, *args, **kargs))

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
            raise Error('UNK token in tokenizer not found.')

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

    def __init__(self, hparams):

        super(ClassifierModel, self).__init__()
        self.hparams = hparams
        self.build_model()

    @abc.abstractmethod
    def build_model(self):

        raise NotImplementedError('build_model is not implemented.')

        # model_params = {k: v for k, v in self.hparams.items() if k.startswith('model')}
        # self.encoder = ModelLoader.load(**model_params)

        # tokenizer_params = {k: v for k, v in self.hparams.items() if k.startswith('tokenizer')}
        # self.tokenizer = TokenizerLoader.load(**tokenizer_params)

        # self.dropout = nn.Dropout(self.hparams.dropout)
        # self.linear = nn.Linear(self.encoder.dim, 1)

    # ---------------------
    # TRAINING
    # ---------------------
    @abc.abstractmethod
    def forward(self, **kargs):

        output = self.encoder.forward(**kargs)
        logits = self.dropout(output)
        logits = self.linear(logits)

        return logits

    @abc.abstractmethod
    def loss(self, labels, logits):
        l = F.cross_entropy(logits, labels)
        return l

    @abc.abstractmethod
    def training_step(self, data_batch, batch_i):
        pass

    @abc.abstractmethod
    def validation_step(self, data_batch, batch_i):
        pass

    @abc.abstractmethod
    def validation_end(self, outputs):
        pass

    # ---------------------
    # TRAINING SETUP
    # ---------------------

    @abc.abstractmethod
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
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
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip=5.0)

        return parser
