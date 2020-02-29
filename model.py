
import os
import pathlib
from typing import *
from itertools import cycle

import torch
import pytorch_lightning as pl
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from loguru import logger
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

class ClassificationDataset(Dataset):

    def __init__(self, instances):

        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]


class Classifier(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.hparams = config
        self.root_path = pathlib.Path(__file__).parent.absolute()
        self.embedder = AutoModel.from_pretrained(config["model"], cache_dir=self.root_path / "model_cache")
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"], cache_dir=self.root_path / "model_cache", use_fast=False)

        self.embedder.train()
        self.classifier = nn.Linear(self.embedder.config.hidden_size, 1, bias=True)

        self.loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

        self.classifier.weight.data.normal_(mean=0.0, std=self.embedder.config.initializer_range)
        self.classifier.bias.data.zero_()

    def forward(self, batch):


        assert len(batch["input_ids"].shape) == 2, "LM only take two-dimensional input"
        assert len(batch["attention_mask"].shape) == 2, "LM only take two-dimensional input"
        assert len(batch["token_type_ids"].shape) == 2, "LM only take two-dimensional input"


        results = self.embedder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], token_type_ids=batch["token_type_ids"])

        token_embeddings, *_ = results
        logits = self.classifier(token_embeddings.mean(dim=1)).squeeze(dim=1)
        logits = logits.reshape(-1, batch["num_choice"])

        return logits

    def training_step(self, batch, batch_idx):

        logits = self.forward(batch)
        loss = self.loss(logits, batch["labels"])
        if self.trainer and self.trainer.use_dp:
            loss = loss.unsqueeze(0)
        return {
            "loss": loss
        }

    def validation_step(self, batch, batch_idx):
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

        val_loss_mean = torch.stack([o['val_loss'] for o in outputs]).mean()
        val_logits = torch.cat([o["val_batch_logits"] for o in outputs])
        val_labels = torch.cat([o["val_batch_labels"] for o in outputs])
        return {
            'val_loss': val_loss_mean,
            "progress_bar": {
                "val_accuracy": torch.sum(val_labels == torch.argmax(val_logits, dim=1)) / (val_labels.shape[0] * 1.0)
            }
        }

    def configure_optimizers(self):

        t_total = len(self.train_dataloader()) // self.hparams["accumulate_grad_batches"] * self.hparams["max_epochs"]

        optimizer = AdamW(self.parameters(), lr=float(self.hparams["learning_rate"]), eps=float(self.hparams["adam_epsilon"]))

        return optimizer

    @pl.data_loader
    def train_dataloader(self):

        return DataLoader(self.dataloader(self.root_path / self.hparams["train_x"], self.root_path / self.hparams["train_y"]), batch_size=self.hparams["batch_size"], collate_fn=self.collate)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dataloader(self.root_path / self.hparams["val_x"], self.root_path / self.hparams["val_y"]), batch_size=self.hparams["batch_size"], collate_fn=self.collate)


    def dataloader(self, x_path: Union[str, pathlib.Path], y_path: Union[str, pathlib.Path] = None):

        df = pd.read_json(x_path, lines=True)
        if y_path:
            labels = pd.read_csv(y_path, sep='\t', header=None).values.tolist()
            self.label_offset = np.asarray(labels).min()
            df["label"] = np.asarray(labels) - self.label_offset

        df["text"] = df.apply(self.transform(self.hparams["formula"]), axis=1)
        print(df.head())
        return ClassificationDataset(df[["text", "label"]].to_dict("records"))


    @staticmethod
    def transform(formula):

        def warpper(row):

            context, choices = formula.split("->")
            # (context + question -> answerA|answerB|answerC)
            # (obs1 + obs2 -> hyp1|hyp2)
            # (ctx_a + ctx_b -> ending_options)
            # (goal -> sol1|sol2)
            context = context.split("+")
            choices = choices.split("|")

            context = " ".join(row[x.strip()] for x in context)
            choices = row[choices[0]] if len(choices) == 0 else [row[x.strip()] for x in choices]
            return list(zip(cycle([context]), choices))

        return warpper


    def collate(self, examples):

        batch_size = len(examples)
        num_choice = len(examples[0]["text"])

        pairs = [pair for example in examples for pair in example["text"]]
        results = self.tokenizer.batch_encode_plus(pairs, add_special_tokens=True, max_length=self.hparams["max_length"], return_tensors='pt', return_attention_masks=True, pad_to_max_length=True)

        assert results["input_ids"].shape[0] == batch_size * num_choice, f"Invalid shapes {results['input_ids'].shape} {batch_size, num_choice}"

        return {
            "input_ids": results["input_ids"],
            "attention_mask": results["attention_mask"],
            "token_type_ids": results["token_type_ids"],
            "labels": torch.LongTensor([e["label"] for e in examples]) if "label" in examples[0] else None,
            "num_choice": num_choice
        }



