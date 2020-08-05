from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AdamW

from utilities.HelperLibrary import transform


# Extending the dataset module provided by the PyTorch module to build the dataset class for AI2 dataset.
class ClassificationDataset(Dataset):

    def __init__(self, instances):
        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]


# Classifier class, with methods support training itself and use it's model to classify
class Classifier(LightningModule):

    def __init__(self, config):
        super().__init__()
        self.hparams = config
        self.root_path = Path(__file__).parent.absolute()
        self.label_offset = 0

        # Load Pretrained Transformer model from cache files (encoder and tokenizer)
        self.embedder = AutoModel.from_pretrained(config["model"], cache_dir=self.root_path / "model_cache")
        self.tokenizer = \
            AutoTokenizer.from_pretrained(config["model"], cache_dir=self.root_path / "model_cache", use_fast=False)
        self.embedder.train()

        # Create the one layer feed forward neural net for classification purpose after the encoder
        self.dropout = nn.Dropout(p=config["classifier_dropout"] if 'classifier_dropout' in config else 0)
        self.classifier = nn.Linear(self.embedder.config.hidden_size, 1, bias=True)
        self.classifier.weight.data.normal_(mean=0.0, std=self.embedder.config.initializer_range)
        self.classifier.bias.data.zero_()

        # Define Loss Function for training
        self.loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

    # Given a tokenized batch, this is the forward step to encode the batch and pass it through the forward classifier
    def forward(self, batch):
        assert len(batch["input_ids"].shape) == 2, "LM only take two-dimensional input"
        assert len(batch["attention_mask"].shape) == 2, "LM only take two-dimensional input"
        assert len(batch["token_type_ids"].shape) == 2, "LM only take two-dimensional input"

        batch["token_type_ids"] = None if "roberta" in self.hparams["model"] else batch["token_type_ids"]

        # Embed the given batch (Using the average of all token embeddings as the representation for the batch)
        results = self.embedder(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                token_type_ids=batch["token_type_ids"])
        token_embeddings, *_ = results
        mean_embeddings = torch.mean(token_embeddings, dim=1).squeeze()

        # Feed through the feed forward network to get the final logits of each classification label
        logits = self.classifier(self.dropout(mean_embeddings)).squeeze(dim=1)
        return logits

    # Custom data loader - load stories and labels into Pandas and create the Classification Dataset
    def dataloader(self, x_path: Union[str, Path], y_path: Union[str, Path] = None):

        df = pd.read_json(x_path, lines=True)
        col_list = ["text"]

        # If given labels are given we will parse it into the dataset along with the story
        if y_path:
            labels = pd.read_csv(y_path, sep='\t', header=None).values.tolist()
            self.label_offset = np.amin(np.asarray(labels))
            df["label"] = np.asarray(labels) - self.label_offset
            col_list.append('label')

        # Transform the text based on the formula
        df["text"] = df.apply(transform(self.hparams["formula"]), axis=1)

        return ClassificationDataset(df[col_list].to_dict("records"))

    # Collate function used by data loader objects, turns text into tokens and their respective token ids
    def collate(self, examples):
        batch_size = len(examples)
        num_choice = len(examples[0]["text"])
        batch = {"labels": torch.LongTensor([e["label"] for e in examples]) if "label" in examples[0] else None,
                 "num_choice": num_choice, "batch_size": batch_size}
        pairs = [pair for example in examples for pair in example["text"]]

        # Tokenize the given batch
        results = self.tokenizer.batch_encode_plus(pairs, truncation='longest_first',
                                                   add_special_tokens=True, max_length=self.hparams["max_length"],
                                                   return_tensors='pt', return_attention_mask=True,
                                                   return_token_type_ids=True, pad_to_max_length=True)
        assert results["input_ids"].shape[0] == batch_size * num_choice, \
            f"Invalid shapes {results['input_ids'].shape} {batch_size, num_choice}"

        batch["input_ids"] = results["input_ids"]
        batch["attention_mask"] = results["attention_mask"]
        batch["token_type_ids"] = results["token_type_ids"]

        return batch

    # Data loader methods to return train and validation data sets
    def train_dataloader(self):
        return DataLoader(
            self.dataloader(self.root_path / self.hparams["train_x"], self.root_path / self.hparams["train_y"]),
            batch_size=self.hparams["batch_size"], collate_fn=self.collate)

    def val_dataloader(self):
        return DataLoader(
            self.dataloader(self.root_path / self.hparams["val_x"], self.root_path / self.hparams["val_y"]),
            batch_size=self.hparams["batch_size"], collate_fn=self.collate)

    # Extend PyTorch Lightning methods for training and validation steps
    def training_step(self, batch, batch_idx, task_id=None):
        logits = self.forward(batch)
        return {"out": logits,
                "labels": batch["labels"],
                "num_choice": batch["num_choice"]}

    def training_step_end(self, batch_parts_outputs):
        logits = batch_parts_outputs["out"]
        num_choice = batch_parts_outputs["num_choice"]
        logits = logits.reshape(-1, num_choice)
        loss = self.loss(logits, batch_parts_outputs["labels"])
        return {"loss": loss,
                "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx, task_id=None):
        logits = self.forward(batch)
        return {"out": logits,
                "labels": batch["labels"],
                "num_choice": batch["num_choice"]}

    def validation_step_end(self, batch_parts_outputs):
        logits = batch_parts_outputs["out"]
        num_choice = batch_parts_outputs["num_choice"]
        logits = logits.reshape(-1, num_choice)
        loss = self.loss(logits, batch_parts_outputs["labels"])
        return {"val_loss": loss,
                "val_batch_logits": logits,
                "val_batch_labels": batch_parts_outputs["labels"]}

    # At the end of each validation epoch, this information is used to determine early stopping
    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([o['val_loss'] for o in outputs]).mean()
        val_logits = torch.cat([o["val_batch_logits"] for o in outputs])
        val_labels = torch.cat([o["val_batch_labels"] for o in outputs])
        correct = torch.sum(torch.eq(val_labels, torch.argmax(val_logits, dim=1)))
        val_accuracy = torch.tensor(float(correct)) / (val_labels.shape[0] * 1.0)
        return {'val_loss': val_loss_mean, "val_accuracy": val_accuracy,
                'combined_metric': val_loss_mean - val_accuracy}

    # Initialize Adam Optimizer: https://arxiv.org/pdf/1412.6980.pdf
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=float(self.hparams["learning_rate"]),
                     eps=float(self.hparams["adam_epsilon"]))
