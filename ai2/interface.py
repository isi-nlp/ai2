from __future__ import annotations

import abc
from abc import ABC
from dataclasses import dataclass
from inspect import getfullargspec
from typing import *

from torch.nn import Module
from transformers import PreTrainedTokenizer, PreTrainedModel


class ModelLoader(ABC, Module):

    def __init__(self, model: Union[Module, PreTrainedModel]):
        super(ModelLoader, self).__init__()
        self.model = model

    @classmethod
    @abc.abstractmethod
    def load(cls, model_type: str, model_weights: str, *args, **kwargs) -> ModelLoader:
        raise NotImplementedError('Load is not implemented.')

    def forward(self, **kwargs) -> Tuple:
        """Follow the convention of omnx, return tuple whenever possible.

        Returns:
            Tuple -- Tuple of returned values of forwading.
        """
        signature = getfullargspec(self.model.forward)
        return self.model.forward(**{k: v for k, v in kwargs.items() if k in signature.args})

    @property
    def dim(self) -> int:
        """Return the hidden dimension of the last layer.

        Returns:
            int -- Last layer's dimension.
        """
        return [p.size(0) for p in self.model.parameters()][-1]


@dataclass
class TokenizerLoader(ABC):
    tokenizer: Union[object, PreTrainedTokenizer]

    @classmethod
    @abc.abstractmethod
    def load(cls, model_type: str, model_weights: str, *args, **kwargs) -> TokenizerLoader:
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


__all__ = ['ModelLoader', 'TokenizerLoader']
