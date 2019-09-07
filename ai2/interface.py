# encoding: utf-8
# Created by chenghaomou at 9/7/19
# Contact: mouchenghao at gmail dot com
# Description: Base abstract classes for model and tokenizer


from __future__ import annotations

import abc
from dataclasses import dataclass
from inspect import getfullargspec
from typing import *

from torch.nn import Module


@dataclass
class BaseModel(abc.ABC):
    base: Module

    @classmethod
    @abc.abstractmethod
    def load(cls, *args, **kwargs) -> BaseModel:
        pass

    def forward(self, **kwargs) -> Tuple:
        signature = getfullargspec(self.base.forward)
        return self.base.forward(**{k: v for k, v in kwargs.items() if k in signature.args})

    @property
    def dimension(self):
        return [p.size(0) for p in self.base.parameters()][-1]


@dataclass
class BaseTokenizer(abc.ABC):
    base: object

    @classmethod
    @abc.abstractmethod
    def load(cls, *args, **kwargs) -> BaseTokenizer:
        pass

    @property
    @abc.abstractmethod
    def sep_index(self):
        pass

    @property
    @abc.abstractmethod
    def sep_token(self):
        pass

    @property
    @abc.abstractmethod
    def unk_index(self):
        pass

    @property
    @abc.abstractmethod
    def unk_token(self):
        pass

    @property
    @abc.abstractmethod
    def cls_index(self):
        pass

    @property
    @abc.abstractmethod
    def cls_token(self):
        pass

    @property
    @abc.abstractmethod
    def pad_index(self):
        pass

    @property
    @abc.abstractmethod
    def pad_token(self):
        pass

    @abc.abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass

    @abc.abstractmethod
    def token2id(self, token: str) -> int:
        pass

    @abc.abstractmethod
    def tokens2ids(self, tokens: List[str]) -> List[int]:
        pass


class PyTorchTransformer(BaseModel):

    @classmethod
    def load(cls, model_class: Callable, model_path: str) -> PyTorchTransformer:
        return PyTorchTransformer(model_class.from_pretrained(model_path))


class PyTorchTransformerTokenizer(BaseTokenizer):

    @classmethod
    def load(cls, model_class: Callable, model_path: str) -> PyTorchTransformerTokenizer:
        return PyTorchTransformerTokenizer(model_class.from_pretrained(model_path, do_lower_case=False))

    @property
    def sep_index(self):
        return self.token2id(self.sep_token)

    @property
    def sep_token(self):
        if self.base._sep_token is None:
            return self.unk_token
        return self.base._sep_token

    @property
    def unk_index(self):
        return self.token2id(self.unk_token)

    @property
    def unk_token(self):
        assert self.base._unk_token is not None, f"UNK token not found"
        return self.base._unk_token

    @property
    def cls_index(self):
        return self.token2id(self.cls_token)

    @property
    def cls_token(self):
        if self.base._cls_token is None:
            return self.base._unk_token
        return self.base._cls_token

    @property
    def pad_index(self):
        return self.token2id(self.pad_token)

    @property
    def pad_token(self):
        if self.base._pad_token is None:
            return self.base._unk_token
        return self.base._pad_token

    def tokenize(self, text: str) -> List[str]:
        return self.base.tokenize(text)

    def token2id(self, token: str) -> int:
        return self.base.convert_tokens_to_ids([token])[0]

    def tokens2ids(self, tokens: List[str]) -> List[int]:
        return self.base.convert_tokens_to_ids(tokens)
