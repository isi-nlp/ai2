from __future__ import annotations

import abc
from abc import ABC
from dataclasses import dataclass
from typing import *

from torch.nn import Module


class ModelLoader(ABC, Module):

    def __init__(self, model: object):
        super(ModelLoader, self).__init__()
        self.model = model

    @classmethod
    @abc.abstractmethod
    def load(cls, model_type: str, model_weights: str, *args, **kwargs) -> ModelLoader:
        raise NotImplementedError('ModelLoader: load is not implemented.')

    def forward(self, **kwargs) -> Tuple:
        raise NotImplementedError('ModelLoader: forward is not implemented.')

    @property
    def dim(self) -> int:
        raise NotImplementedError('ModelLoader: dim is not implemented.')


@dataclass
class TokenizerLoader(ABC):
    tokenizer: object

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
        raise NotImplementedError('tokenize is not implemented')


__all__ = ['ModelLoader', 'TokenizerLoader']
