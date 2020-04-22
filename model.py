import random
import os
import pathlib
from typing import *
from itertools import cycle

import torch
import pytorch_lightning as pl
import torch.nn as nn
import pandas as pd
import numpy as np
from loguru import logger
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaModel
from tensorboardX import SummaryWriter
import abc
from abc import ABC
from torch.nn import Module
from transformers import *
from inspect import getfullargspec
from transformers import T5Tokenizer, T5ForConditionalGeneration

random.seed(0)

MODELS = {
    'bert': BertModel,
    'distilbert': DistilBertModel,
    'xlm': XLMModel,
    'xlnet': XLNetModel,
    'roberta': RobertaModel,
    'roberta_mlm': RobertaForMaskedLM,
    'gpt': OpenAIGPTModel,
    'gpt2': GPT2Model,
    't5': T5ForConditionalGeneration,
    #    'libert': LiBertModel
}


class ClassificationDataset(Dataset):

    def __init__(self, instances):
        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]


class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, dataloaders, shuffle: bool = True):
        self.data: List = []
        for loader in dataloaders:
            for batch in loader:
                self.data.append(batch)
        if shuffle:
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class ModelLoader(ABC, Module):

    def __init__(self, model: object):
        super(ModelLoader, self).__init__()
        self.model = model

    @classmethod
    @abc.abstractmethod
    def load(cls, model_type: str, model_weights: str, *args, **kwargs):
        raise NotImplementedError('ModelLoader: load is not implemented.')

    def forward(self, **kwargs) -> Tuple:
        raise NotImplementedError('ModelLoader: forward is not implemented.')

    @property
    def dim(self) -> int:
        raise NotImplementedError('ModelLoader: dim is not implemented.')


class HuggingFaceModelLoader(ModelLoader):

    def __init__(self, model: Union[Module, PreTrainedModel], model_type: str):
        super(HuggingFaceModelLoader, self).__init__(model)
        if model_type == 'roberta_mlm':
            self.lm_head = self.model.lm_head
            self.model = self.model.roberta

    @classmethod
    def load(cls, model_type: str, model_weights: str, *args, **kargs):
        assert model_type in MODELS, "Model type is not recognized."
        return HuggingFaceModelLoader(MODELS[model_type].from_pretrained(model_weights, cache_dir="./model_cache"),
                                      model_type=model_type)

    @property
    def dim(self) -> int:
        """Return the hidden dimension of the last layer.
        Returns:
            int -- Last layer's dimension.
        """
        return [p.size(0) for p in self.model.parameters()][-1]


class Classifier(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.hparams = config
        self.root_path = pathlib.Path(__file__).parent.absolute()
        self.embedder = AutoModel.from_pretrained(config["model"], cache_dir=self.root_path / "model_cache")
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"], cache_dir=self.root_path / "model_cache", use_fast=False)
        self.embedder.train()
        self.dropout = nn.Dropout(config["dropout"])

        self.label_offset = 0

        self.classifier = nn.Linear(self.embedder.config.hidden_size, 1, bias=True)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
        self.classifier.weight.data.normal_()
        self.classifier.bias.data.zero_()

        if "task_name2" in self.hparams:
            self.classifier2 = nn.Linear(self.embedder.config.hidden_size, 1, bias=True)
            self.classifier2.weight.data.normal_()
            self.classifier2.bias.data.zero_()

    def forward(self, batch):
        if False:
            print(batch["input_ids"])
            for i in range(len(batch["input_ids"])):
                dd = batch["input_ids"][i]
                print(self.tokenizer.decode(dd))

        assert len(batch["input_ids"].shape) == 2, "LM only take two-dimensional input"
        assert len(batch["attention_mask"].shape) == 2, "LM only take two-dimensional input"
        assert len(batch["token_type_ids"].shape) == 2, "LM only take two-dimensional input"

        batch["token_type_ids"] = None if "roberta" in self.hparams["model"] or "lm_finetuned" \
                                          in self.hparams["model"] else batch["token_type_ids"]

        results = self.embedder(input_ids=batch["input_ids"],
                                decoder_input_ids=batch["input_ids"],)

        print(len(results))
        token_embeddings, a = results
        print(token_embeddings.shape)
        print(a)
        print(a.shape)
        output = torch.mean(token_embeddings, dim=1).squeeze()
        print(output.shape)
        output = self.dropout(output)
        print(output.shape)
        if batch["task_id"] == 2:
            logits = self.classifier2(output).squeeze(dim=1)
        elif batch["task_id"] == 0:
            logits = self.classifier(output).squeeze(dim=1)
        else:
            raise
        print(logits.shape)
        logits = logits.reshape(-1, batch["num_choice"])
        print(logits.shape)
        return logits

    def training_step(self, batch, batch_idx, task_id=None):

        logits = self.forward(batch)
        loss = self.loss(logits, batch["labels"])
        if self.trainer and self.trainer.use_dp:
            loss = loss.unsqueeze(0)

        if batch["task_id"] == 2:
            return {
                "loss": loss,
                "progress": {
                    "loss2": loss
                },
            }

        return {
            "loss": loss,
        }

    def validation_step(self, batch, batch_idx, task_id=None):
        logits = self.forward(batch)
        loss = self.loss(logits, batch["labels"])
        if self.trainer and self.trainer.use_dp:
            loss = loss.unsqueeze(0)
        return {
            'val_loss': loss,
            "val_batch_logits": logits,
            "val_batch_labels": batch["labels"],
        }

    def validation_end(self, outputs):

        print(len(outputs))
        if "task_name2" in self.hparams:
            val_loss_mean = torch.stack([o['val_loss'] for o in outputs[0]]).mean()
            val_logits = torch.cat([o["val_batch_logits"] for o in outputs[0]])
            val_labels = torch.cat([o["val_batch_labels"] for o in outputs[0]])
            correct = torch.sum(val_labels == torch.argmax(val_logits, dim=1))
            val_accuracy = torch.tensor(float(correct)) / (val_labels.shape[0] * 1.0)

            val_loss_mean2 = torch.stack([o['val_loss'] for o in outputs[1]]).mean()
            val_logits2 = torch.cat([o["val_batch_logits"] for o in outputs[1]])
            val_labels2 = torch.cat([o["val_batch_labels"] for o in outputs[1]])
            correct2 = torch.sum(val_labels2 == torch.argmax(val_logits2, dim=1))
            val_accuracy2 = torch.tensor(float(correct2)) / (val_labels2.shape[0] * 1.0)

            return {
                'val_loss': val_loss_mean,
                "val_accuracy": val_accuracy,
                'val_loss2': val_loss_mean2,
                "val_accuracy2": val_accuracy2,
            }

        else:
            val_loss_mean = torch.stack([o['val_loss'] for o in outputs]).mean()
            val_logits = torch.cat([o["val_batch_logits"] for o in outputs])
            val_labels = torch.cat([o["val_batch_labels"] for o in outputs])
            correct = torch.sum(val_labels == torch.argmax(val_logits, dim=1))
            val_accuracy = torch.tensor(float(correct)) / (val_labels.shape[0] * 1.0)
            return {
                'val_loss': val_loss_mean,
                "val_accuracy": val_accuracy,
            }

    def configure_optimizers(self):

        # t_total = len(self.train_dataloader()) // self.hparams["accumulate_grad_batches"] * self.hparams["max_epochs"]
        t_total = len(self.train_dataloader) // self.hparams["accumulate_grad_batches"] * self.hparams["max_epochs"]

        optimizer = AdamW(self.parameters(), lr=float(self.hparams["learning_rate"]),
                          eps=float(self.hparams["adam_epsilon"]))

        return optimizer

    @pl.data_loader
    def train_dataloader(self):
        dataloader = DataLoader(self.dataloader(self.root_path / self.hparams["train_x"],
                                                self.root_path / self.hparams["train_y"]),
                                batch_size=self.hparams["batch_size"],
                                collate_fn=self.collate, shuffle=True)

        if "train2_x" in self.hparams:
            dataloader2 = DataLoader(self.dataloader(self.root_path / self.hparams["train2_x"],
                                                     self.root_path / self.hparams["train2_y"], task_id=2),
                                     batch_size=self.hparams["batch_size"],
                                     collate_fn=self.collate, shuffle=True)

            dataloaders = [dataloader, dataloader2]
            multidatasets = MultiTaskDataset(dataloaders)
            multi_dataloader = DataLoader(multidatasets,
                                          collate_fn=lambda examples: examples[0],
                                          shuffle=True, batch_size=1)
            return multi_dataloader

        return dataloader

    @pl.data_loader
    def val_dataloader(self):
        dataloader = DataLoader(self.dataloader(self.root_path / self.hparams["val_x"],
                                                self.root_path / self.hparams["val_y"]),
                                batch_size=self.hparams["batch_size"],
                                collate_fn=self.collate)

        if "val2_x" in self.hparams:
            dataloader2 = DataLoader(self.dataloader(self.root_path / self.hparams["val2_x"],
                                                     self.root_path / self.hparams["val2_y"], task_id=2),
                                     batch_size=self.hparams["batch_size"],
                                     collate_fn=self.collate, shuffle=False)

            dataloaders = [dataloader, dataloader2]
            return dataloaders

        return dataloader

    def dataloader(self, x_path: Union[str, pathlib.Path],
                   y_path: Union[str, pathlib.Path] = None,
                   task_id=None):

        df = pd.read_json(x_path, lines=True)
        if y_path:
            labels = pd.read_csv(y_path, sep='\t', header=None).values.tolist()
            self.label_offset = np.asarray(labels).min()
            df["label"] = np.asarray(labels) - self.label_offset

        if task_id is None:
            task_id_str = ""
        else:
            task_id_str = str(task_id)

        df["text"] = df.apply(self.transform(self.hparams["formula{}".format(task_id_str)]), axis=1)
        df["task_id"] = task_id if task_id is not None else 0
        print(df.head())
        return ClassificationDataset(df[["text", "label", "task_id"]].to_dict("records"))

    @staticmethod
    def transform(formula):

        def warpper(row):
            context, choices = formula.split("->")
            context = context.split("+")
            choices = choices.split("|")

            context = " ".join(row[x.strip()] for x in context)
            choices = row[choices[0]] if len(choices) == 0 else [row[x.strip()] for x in choices]
            return list(zip(cycle([context]), choices))

        return warpper

    def collate(self, examples):

        batch_size = len(examples)
        num_choice = len(examples[0]["text"])
        task_id = examples[0]["task_id"]

        pairs = [pair for example in examples for pair in example["text"]]
        results = self.tokenizer.batch_encode_plus(pairs,
                                                   add_special_tokens=True, max_length=self.hparams["max_length"],
                                                   return_tensors='pt', return_attention_masks=True,
                                                   pad_to_max_length=True)

        assert results["input_ids"].shape[0] == batch_size * num_choice, \
            f"Invalid shapes {results['input_ids'].shape} {batch_size, num_choice}"

        batch = {
            "input_ids": results["input_ids"],
            "attention_mask": results["attention_mask"],
            "token_type_ids": results["token_type_ids"],
            "labels": torch.LongTensor([e["label"] for e in examples]) if "label" in examples[0] else None,
            "num_choice": num_choice,
            "task_id": task_id
        }

        # TODO: provide associated tree ids here
        batch['additional_position_ids'] = torch.zeros_like(results["input_ids"])

        return batch