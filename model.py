import pathlib
from itertools import cycle
from typing import *

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AdamW


# Extending the dataset module provided by the PyTorch module to build the dataset class for AI2 dataset.
class ClassificationDataset(Dataset):

    def __init__(self, instances):
        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]


# Classifier class, with methods support training itself and use it's model to classify
class Classifier(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.hparams = config
        self.root_path = pathlib.Path(__file__).parent.absolute()
        self.label_offset = 0

        # Load Transformer model from cache files (encoder and tokenizer)
        self.embedder = AutoModel.from_pretrained(config["model"], cache_dir=self.root_path / "model_cache")
        self.tokenizer = \
            AutoTokenizer.from_pretrained(config["model"], cache_dir=self.root_path / "model_cache", use_fast=False)
        self.embedder.train()

        # Create the one layer feed forward neural net for classification purpose after the encoder
        self.classifier = nn.Linear(self.embedder.config.hidden_size, 1, bias=True)
        self.classifier.weight.data.normal_(mean=0.0, std=self.embedder.config.initializer_range)
        self.classifier.bias.data.zero_()
        self.loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

    # Given a batch output the it's forward result
    def forward(self, batch):
        assert len(batch["input_ids"].shape) == 2, "LM only take two-dimensional input"
        assert len(batch["attention_mask"].shape) == 2, "LM only take two-dimensional input"
        assert len(batch["token_type_ids"].shape) == 2, "LM only take two-dimensional input"

        batch["token_type_ids"] = None if "roberta" in self.hparams["model"] else batch["token_type_ids"]

        # Embed the given batch
        results = self.embedder(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                token_type_ids=batch["token_type_ids"])
        token_embeddings, *_ = results

        # Feed through the feed forward network to get the final logits of each classification label
        logits = self.classifier(token_embeddings.mean(dim=1)).squeeze(dim=1)
        logits = logits.reshape(-1, batch["num_choice"])
        return logits

    # Given a batch and the desired feature, this function embeds the batch and return the embedding representation
    def retrieve_embedding(self, batch, feature):

        batch_size = len(batch['labels'])
        num_choices = batch['num_choice']
        batch["token_type_ids"] = None if "roberta" in self.hparams["model"] else batch["token_type_ids"]

        with torch.no_grad():
            # Embed the given batch
            results = self.embedder(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    token_type_ids=batch["token_type_ids"])
            token_embeddings, *_ = results

            if feature.split('_')[0] == 'AVG':
                per_story_per_choice_embed = token_embeddings.mean(dim=1).reshape(batch_size, num_choices, -1)
                if feature == 'AVG_MEAN':
                    per_story_avg_embed = per_story_per_choice_embed.mean(dim=1)
                    return per_story_avg_embed.cpu().detach().numpy()
                # TODO: Implement AVG_CORRECT
                # TODO: Implement AVG_NULL
            # TODO: Implement CLS_*
            else:
                raise NotImplementedError("Feature for embedding calculation is not yet implemented")

    # Custom data loader
    def dataloader(self, x_path: Union[str, pathlib.Path], y_path: Union[str, pathlib.Path] = None):
        df = pd.read_json(x_path, lines=True)

        # If given labels are given we will parse it into the dataset
        if y_path:
            labels = pd.read_csv(y_path, sep='\t', header=None).values.tolist()
            self.label_offset = np.asarray(labels).min()
            df["label"] = np.asarray(labels) - self.label_offset

        # Transform the text based on the formula
        df["text"] = df.apply(self.transform(self.hparams["formula"]), axis=1)

        print(df.head())
        return ClassificationDataset(df[["text", "label"]].to_dict("records"))

    # Lambda function that parse in the formulas of how to read in training data
    @staticmethod
    def transform(formula):

        # Apply to a pandas data frame row
        def wrapper(row):
            context, choices = formula.split("->")
            # alphanli:     (obs1 + obs2 -> hyp1|hyp2)
            # hellaswag:    (ctx_a + ctx_b -> ending_options)
            # c_hellaswag:  (ctx_a + ctx_b -> opt0|opt1|opt2|opt3)
            # physicaliqa:  (goal -> sol1|sol2)
            # socialiqa:    (context + question -> answerA|answerB|answerC)
            context = context.strip().split("+")
            choices = choices.strip().split("|")

            context = " ".join(row[a_context.strip()] for a_context in context)

            # # If we are processing the original hellaswag, the options are encoded in nested list, so we flatten it
            # if choices == ['ending_options']:
            #     choices = row['ending_options']
            # else:
            #     choices = [row[a_choice.strip()] for a_choice in choices]

            return list(zip(cycle([context]), choices))

        return wrapper

    # Collate function used by data loader objects
    def collate(self, examples):

        batch_size = len(examples)
        num_choice = len(examples[0]["text"])

        pairs = [pair for example in examples for pair in example["text"]]
        results = self.tokenizer.batch_encode_plus(pairs, add_special_tokens=True,
                                                   max_length=self.hparams["max_length"], return_tensors='pt',
                                                   return_attention_masks=True, pad_to_max_length=True)

        assert results["input_ids"].shape[0] == batch_size * num_choice, \
            f"Invalid shapes {results['input_ids'].shape} {batch_size, num_choice}"

        return {"input_ids": results["input_ids"],
                "attention_mask": results["attention_mask"],
                "token_type_ids": results["token_type_ids"],
                "labels": torch.LongTensor([e["label"] for e in examples]) if "label" in examples[0] else None,
                "num_choice": num_choice}

    # Data loader methods to return train and validation data sets
    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(
            self.dataloader(self.root_path / self.hparams["train_x"], self.root_path / self.hparams["train_y"]),
            batch_size=self.hparams["batch_size"], collate_fn=self.collate)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            self.dataloader(self.root_path / self.hparams["val_x"], self.root_path / self.hparams["val_y"]),
            batch_size=self.hparams["batch_size"], collate_fn=self.collate)

    # Extend PyTorch Lightning methods
    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.loss(logits, batch["labels"])
        if self.trainer and self.trainer.use_dp:
            loss = loss.unsqueeze(0)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.loss(logits, batch["labels"])
        if self.trainer and self.trainer.use_dp:
            loss = loss.unsqueeze(0)
        return {'val_loss': loss,
                "val_batch_logits": logits,
                "val_batch_labels": batch["labels"]}

    def validation_end(self, outputs):
        val_loss_mean = torch.stack([o['val_loss'] for o in outputs]).mean()
        val_logits = torch.cat([o["val_batch_logits"] for o in outputs])
        val_labels = torch.cat([o["val_batch_labels"] for o in outputs])
        correct = torch.sum(val_labels == torch.argmax(val_logits, dim=1))
        val_accuracy = torch.tensor(float(correct)) / (val_labels.shape[0] * 1.0)
        return {'val_loss': val_loss_mean, "val_accuracy": val_accuracy}

    def configure_optimizers(self):
        t_total = len(self.train_dataloader) // self.hparams["accumulate_grad_batches"] * self.hparams["max_epochs"]
        optimizer = AdamW(self.parameters(), lr=float(self.hparams["learning_rate"]),
                          eps=float(self.hparams["adam_epsilon"]))
        return optimizer
