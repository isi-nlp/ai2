import itertools

from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
from pytorch_lightning.profiler import AdvancedProfiler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, AutoModel, AutoTokenizer


class ClassificationDataset(Dataset):
    def __init__(self, data_size: int):
        self.data_size = data_size

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return {
            "text": [
                ("When boiling butter, when it's ready, you can", "Pour it onto a plate"),
                ("When boiling butter, when it's ready, you can", "Pour it into a jar"),
            ],
            "label": 1,
        }


class Classifier(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # Load Transformer model from cache files (encoder and tokenizer)
        self.embedder = AutoModel.from_pretrained("roberta-large", cache_dir="model_cache")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "roberta-large", cache_dir="model_cache", use_fast=False
        )
        self.embedder.train()

        # Create the one layer feed forward neural net for classification purpose after the encoder
        self.classifier = nn.Linear(self.embedder.config.hidden_size, 1, bias=True)
        self.classifier.weight.data.normal_(mean=0.0, std=self.embedder.config.initializer_range)
        self.classifier.bias.data.zero_()
        self.loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

    def forward(self, batch):
        token_embeddings, *_ = self.embedder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        logits = self.classifier(token_embeddings.mean(dim=1)).squeeze(dim=1)
        return logits

    def collate(self, examples):
        num_choices = [len(example["text"]) for example in examples]

        pairs = [pair for example in examples for pair in example["text"]]
        results = self.tokenizer.batch_encode_plus(
            pairs,
            add_special_tokens=True,
            max_length=self.hparams["max_length"],
            return_tensors="pt",
            return_attention_masks=True,
            pad_to_max_length=True,
        )

        return {
            "input_ids": results["input_ids"],
            "attention_mask": results["attention_mask"],
            "labels": [e["label"] for e in examples],
            "num_choice": num_choices,
        }

    def generic_dataloader(self, data_size: int):
        return DataLoader(
            ClassificationDataset(data_size),
            batch_size=self.hparams["batch_size"],
            collate_fn=self.collate,
        )

    def train_dataloader(self):
        return self.generic_dataloader(10_000)

    def val_dataloader(self):
        return self.generic_dataloader(1_000)

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        return {"out": logits, "labels": batch["labels"], "num_choice": batch["num_choice"]}

    def training_step_end(self, batch_parts_outputs):
        logits = batch_parts_outputs["out"]
        num_choices = batch_parts_outputs["num_choice"]
        labels = torch.tensor(batch_parts_outputs["labels"], dtype=torch.long, device=self.device)
        logits = logits.split(num_choices)
        labels = labels.split(1)
        loss = torch.stack(
            list(self.loss(l1.reshape(1, -1), l2) for l1, l2 in zip(logits, labels))
        ).mean()
        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        return {"out": logits, "labels": batch["labels"], "num_choice": batch["num_choice"]}

    def validation_step_end(self, batch_parts_outputs):
        logits = batch_parts_outputs["out"]
        num_choices = batch_parts_outputs["num_choice"]
        labels = torch.tensor(batch_parts_outputs["labels"], dtype=torch.long, device=self.device)
        logits = logits.split(num_choices)
        labels = labels.split(1)
        loss = torch.stack(
            list(self.loss(l1.reshape(1, -1), l2) for l1, l2 in zip(logits, labels))
        ).mean()
        return {"val_loss": loss, "val_batch_logits": logits, "val_batch_labels": labels}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([o["val_loss"] for o in outputs]).mean()
        val_logits = tuple(itertools.chain.from_iterable(o["val_batch_logits"] for o in outputs))
        val_answers = torch.stack([torch.argmax(logits) for logits in val_logits])
        val_labels = torch.cat(
            tuple(itertools.chain.from_iterable(o["val_batch_labels"] for o in outputs))
        )
        correct = torch.sum(val_labels == val_answers)
        val_accuracy = torch.tensor(float(correct)) / (val_labels.shape[0] * 1.0)
        return {"log": {"val_loss": val_loss_mean, "val_accuracy": val_accuracy}}

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=float(self.hparams["learning_rate"]),
            eps=float(self.hparams["adam_epsilon"]),
        )
        return optimizer


def main() -> None:
    config = {
        "learning_rate": 2e-6,
        "adam_epsilon": 10e-8,
        "batch_size": 4,
        "max_length": 128,
    }

    seed_everything(42)

    model = Classifier(config)

    trainer = Trainer(
        logger=False,
        checkpoint_callback=False,
        gpus=-1 if torch.cuda.is_available() else None,
        accumulate_grad_batches=16,
        max_epochs=4,
        min_epochs=1,
        distributed_backend="dp",
        num_sanity_val_steps=5,
        benchmark=False,
        deterministic=True,
        profiler=AdvancedProfiler(output_filename="simple_trans251"),
        # fast_dev_run=True,  # TODO: Remove
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
