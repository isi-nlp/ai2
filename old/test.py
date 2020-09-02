import argparse
import random

import numpy as np
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW


class ClassificationDataset(Dataset):

    def __init__(self, num_features, num_choices, num_instances):
        instances = []
        for _ in range(num_instances):
            instance = {
                "text": [[random.gauss(0, 1) for _ in range(num_features)]
                         for _ in range(num_choices)],
                "label": random.randint(0, num_choices - 1),
            }
            instances.append(instance)
            self.instances = instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]


# Classifier class, with methods support training itself and use it's model to classify
class Classifier(pl.LightningModule):

    def __init__(self, batch_size, num_choices, num_instances):
        super().__init__()

        self.batch_size = batch_size
        self.num_choices = num_choices
        self.num_instances = num_instances

        self.num_features = 100

        # Create the one layer feed forward neural net for classification purpose after the encoder
        self.classifier = nn.Linear(self.num_features, 1, bias=True)
        self.classifier.weight.data.normal_()
        self.classifier.bias.data.zero_()
        self.loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

    # Given a batch output the it's forward result
    def forward(self, batch):
        # Feed through the feed forward network to get the final logits of each classification label
        logits = self.classifier(batch['inputs']).squeeze(dim=1)
        return logits

    # Collate function used by data loader objects
    def collate(self, examples):
        batch_size = len(examples)
        num_choice = len(examples[0]["text"])

        return {"inputs": torch.FloatTensor([e["text"] for e in examples]),
                "labels": torch.LongTensor([e["label"] for e in examples]),
                "num_choice": torch.LongTensor([num_choice] * batch_size)}

    # Data loader methods to return train and validation data sets
    def train_dataloader(self):
        return DataLoader(ClassificationDataset(self.num_features, self.num_choices, self.num_instances),
                          batch_size=self.batch_size, collate_fn=self.collate)

    def val_dataloader(self):
        return DataLoader(ClassificationDataset(self.num_features, self.num_choices, self.num_instances),
                          batch_size=self.batch_size, collate_fn=self.collate)

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
        optimizer = AdamW(self.parameters(), lr=2e-6, eps=1e-7)
        return optimizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--accumulate-grad-batches', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=4)
    args = p.parse_args()

    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cuda.deterministic = True
    torch.backends.cuda.benchmark = False

    # Initialize the classifier by arguments specified in config file
    model = Classifier(batch_size=args.batch_size, num_choices=4, num_instances=11)

    trainer = Trainer(
        gradient_clip_val=0,
        gpus=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else None,
        log_gpu_memory="all",
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_epochs=4,
        min_epochs=1,
        distributed_backend="dp",
        weights_summary='top',
        fast_dev_run=True,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
