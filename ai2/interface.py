from __future__ import annotations

import abc
from abc import ABC
from dataclasses import dataclass
from inspect import getfullargspec
from typing import *

from pytorch_transformers import *
from torch.nn import Module

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


class ModelLoader(ABC, Module):

    def __init__(self, model: Union[Module, PreTrainedModel]):
        super(ModelLoader, self).__init__()
        self.model = model

    @classmethod
    @abc.abstractmethod
    def load(cls, *args, **kwargs) -> ModelLoader:
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

    def __init__(self, model: Union[Module, PreTrainedModel]):
        super(HuggingFaceModelLoader, self).__init__(model)

    def forward(self, **kwargs) -> Tuple:
        """Follow the convention of omnx, return tuple whenever possible.

        Returns:
            Tuple -- Tuple of returned values of forwading.
        """
        signature = getfullargspec(self.model.forward)
        return self.model.forward(
            **{k: None if k == "token_type_ids" and self.model.config.type_vocab_size < 2 else v for k, v in kwargs.items() if k in signature.args})

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
