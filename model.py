from itertools import cycle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, AutoModel, AutoTokenizer


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

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.root_path = Path(__file__).parent.absolute()
        self.label_offset = 0

        # Load Transformer model from cache files (encoder and tokenizer)
        self.embedder = AutoModel.from_pretrained(hparams["model"], cache_dir=self.root_path / "model_cache")
        self.tokenizer = \
            AutoTokenizer.from_pretrained(hparams["model"], cache_dir=self.root_path / "model_cache", use_fast=False)
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
        return logits

    # Custom data loader
    def dataloader(self, x_path: Union[str, Path], y_path: Union[str, Path] = None):
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
        parsed_context, parsed_choices = formula.split("->")
        # alphanli:     (obs1 + obs2 -> hyp1|hyp2)
        # hellaswag:    (ctx_a + ctx_b -> ending_options)
        # physicaliqa:  (goal -> sol1|sol2)
        # socialiqa:    (context + question -> answerA|answerB|answerC)
        parsed_context = parsed_context.strip().split("+")
        parsed_choices = parsed_choices.strip().split("|")

        # Apply to a pandas data frame row
        def wrapper(row):
            context = " ".join(row[a_context.strip()] for a_context in parsed_context)

            # If we are processing hellaswag, the options are encoded in nested list, so we will flatten it
            if len(parsed_choices) == 1:
                choices = row[parsed_choices[0]]
            else:
                choices = [row[a_choice.strip()] for a_choice in parsed_choices]

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
                "num_choice": torch.LongTensor([num_choice] * batch_size)}

    # Data loader methods to return train and validation data sets
    def train_dataloader(self):
        return DataLoader(
            self.dataloader(self.root_path / self.hparams["train_x"], self.root_path / self.hparams["train_y"]),
            batch_size=self.hparams["batch_size"], collate_fn=self.collate)

    def val_dataloader(self):
        return DataLoader(
            self.dataloader(self.root_path / self.hparams["val_x"], self.root_path / self.hparams["val_y"]),
            batch_size=self.hparams["batch_size"], collate_fn=self.collate)

    # Extend PyTorch Lightning methods
    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        return {"out": logits,
                "labels": batch["labels"],
                "num_choice": batch["num_choice"]}

    def training_step_end(self, batch_parts_outputs):
        logits = batch_parts_outputs["out"]
        num_choice = batch_parts_outputs["num_choice"].flatten()[0].item()
        logits = logits.reshape(-1, num_choice)
        loss = self.loss(logits, batch_parts_outputs["labels"])
        return {"loss": loss,
                "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        return {"out": logits,
                "labels": batch["labels"],
                "num_choice": batch["num_choice"]}

    def validation_step_end(self, batch_parts_outputs):
        logits = batch_parts_outputs["out"]
        num_choice = batch_parts_outputs["num_choice"].flatten()[0].item()
        logits = logits.reshape(-1, num_choice)
        loss = self.loss(logits, batch_parts_outputs["labels"])
        return {"val_loss": loss,
                "val_batch_logits": logits,
                "val_batch_labels": batch_parts_outputs["labels"]}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([o['val_loss'] for o in outputs]).mean()
        val_logits = torch.cat([o["val_batch_logits"] for o in outputs])
        val_labels = torch.cat([o["val_batch_labels"] for o in outputs])
        correct = torch.sum(val_labels == torch.argmax(val_logits, dim=1))
        val_accuracy = torch.tensor(float(correct)) / (val_labels.shape[0] * 1.0)
        return {'log': {'val_loss': val_loss_mean, "val_accuracy": val_accuracy}}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=float(self.hparams["learning_rate"]),
                          eps=float(self.hparams["adam_epsilon"]))
        return optimizer
