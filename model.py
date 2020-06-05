import random
from itertools import cycle
from pathlib import Path
from typing import Union, List

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
        self.dropout = nn.Dropout(hparams["dropout"])

        # Create the one layer feed forward neural net for classification purpose after the encoder
        self.classifier = nn.Linear(self.embedder.config.hidden_size, 1, bias=True)
        self.classifier.weight.data.normal_(mean=0.0, std=self.embedder.config.initializer_range)
        self.classifier.bias.data.zero_()
        self.loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

        if "task_name2" in self.hparams:
            self.classifier2 = nn.Linear(self.embedder.config.hidden_size, 1, bias=True)
            self.classifier2.weight.data.normal_(mean=0.0, std=self.embedder.config.initializer_range)
            self.classifier2.bias.data.zero_()

    # Given a batch output the it's forward result
    def forward(self, batch):
        assert len(batch["input_ids"].shape) == 2, "LM only take two-dimensional input"
        assert len(batch["attention_mask"].shape) == 2, "LM only take two-dimensional input"
        assert len(batch["token_type_ids"].shape) == 2, "LM only take two-dimensional input"

        batch["token_type_ids"] = None if "roberta" in self.hparams["model"] or "lm_finetuned" \
                                          in self.hparams["model"] else batch["token_type_ids"]
        results = self.embedder(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                token_type_ids=batch["token_type_ids"])

        if 't5' in self.hparams["model"]:
            results = self.embedder(input_ids=batch["input_ids"],
                                    decoder_input_ids=batch["input_ids"], )

        token_embeddings, *_ = results

        if self.hparams['architecture'] == 'embed_all_sep_mean':
            # Get the mean of part of the embedding that corresponds to the answer
            mean_dims = token_embeddings.shape[0], token_embeddings.shape[2]
            mean_embeddings = torch.zeros(mean_dims)
            for i in range(mean_dims[0]):
                q_i, q_j = batch['question_positions'][i]
                a_i, a_j = batch['answer_positions'][i]
                token_embeddings_for_sequence_i = token_embeddings[i, :, :].squeeze()
                question_seq = token_embeddings_for_sequence_i[q_i:q_j + 1, :]
                ans_seq = token_embeddings_for_sequence_i[a_i:a_j + 1, :]
                combined = torch.cat((question_seq, ans_seq), 0)  # concat context and answer
                combined_mean = torch.mean(combined, dim=0).squeeze()  # mean of the question and the correct answer
                mean_embeddings[i, :] = combined_mean
                mean_embeddings = mean_embeddings.to(torch.device('cuda'))
        else:
            mean_embeddings = torch.mean(token_embeddings, dim=1).squeeze()
        output = self.dropout(mean_embeddings)
        if batch["task_id"] == 2:
            logits = self.classifier2(output).squeeze(dim=1)
        elif batch["task_id"] == 0:
            logits = self.classifier(output).squeeze(dim=1)
        else:
            raise
        return logits

    # Custom data loader
    def dataloader(self, x_path: Union[str, Path], y_path: Union[str, Path] = None, task_id=None, data_slice=100):
        df = pd.read_json(x_path, lines=True)

        # If given labels are given we will parse it into the dataset
        if y_path:
            labels = pd.read_csv(y_path, sep='\t', header=None).values.tolist()
            self.label_offset = np.asarray(labels).min()
            df["label"] = np.asarray(labels) - self.label_offset

        task_id_str = "" if task_id is None else str(task_id)

        # Transform the text based on the formula
        df["text"] = df.apply(self.transform(self.hparams["formula{}".format(task_id_str)]), axis=1)
        df["task_id"] = task_id if task_id is not None else 0
        # Get the first n elements, if data set slicing is specified
        df = df[:int(len(df.index) * (data_slice / 100))]
        col_list = ["text", "task_id", "question_context"]
        # We use the context in embed_all_sep_mean architecture
        df["question_context"] = df["text"].apply(lambda x: x[0][0].split(' - ')[0])
        if 'label' in df.columns:
            col_list.append('label')
        # pd.set_option('display.max_columns', None)
        # pd.options.display.max_colwidth = 1000
        print(df.head())
        print(len(df.index))

        return ClassificationDataset(df[col_list].to_dict("records"))

    # Lambda function that parse in the formulas of how to read in training data
    def transform(self, formula):
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

            # Include answers in context - The approach is same for all QA
            if self.hparams['architecture'] in ['include_answers_in_context', 'embed_all_sep_mean']:
                context = context + ' - ' + ' - '.join(choices)
            return list(zip(cycle([context]), choices))

        return wrapper

    @staticmethod
    def find_sub_list(sl, l):
        sll = len(sl)
        for ind in (i for i, e in enumerate(l) if e == sl[0]):
            if l[ind:ind + sll] == sl:
                return ind, ind + sll - 1

    # Collate function used by data loader objects
    def collate(self, examples):

        batch_size = len(examples)
        num_choice = len(examples[0]["text"])
        task_id = examples[0]["task_id"]
        batch = {
            "labels": torch.LongTensor([e["label"] for e in examples]) if "label" in examples[0] else None,
            "num_choice": torch.LongTensor([num_choice] * batch_size),
            "task_id": task_id
        }

        # Reformat Multiple choice parsing
        # We create single embedding, but takes sub representations later/ We need to know i
        if self.hparams['architecture'] == 'embed_all_sep_mean':
            #E.g: Piqa: Goal + Answers , Goal, Answer
            context_answer_pairs = [(c, example['question_context'], a) for example in examples for c, a in example["text"]]
            # We just keep the context, i.e question and all the answers (goal+answers), and not the correct answer
            pairs = [pair[0] for pair in context_answer_pairs]
            question_positions = []
            answer_positions = [] # Positions of the answer within the entire context
            # We also want to know just the token representation of context and correct ans, we get the subsequence
            for pair in context_answer_pairs:
                entire_qa_tokens = self.tokenizer.tokenize(pair[0]) # This is the entire QA tokenized
                question_tokens = self.tokenizer.tokenize(pair[1]) # This is just the Q tokenixed
                question_positions.append(self.find_sub_list(question_tokens, entire_qa_tokens))
                answers_tokens = self.tokenizer.tokenize('Answer ' + pair[2])[1:]
                answer_positions.append(self.find_sub_list(answers_tokens, entire_qa_tokens))
            batch['question_positions'] = question_positions
            batch['answer_positions'] = answer_positions
        else:
            pairs = [pair for example in examples for pair in example["text"]]

        results = self.tokenizer.batch_encode_plus(pairs,
                                                   add_special_tokens=True, max_length=self.hparams["max_length"],
                                                   return_tensors='pt', return_attention_masks=True,
                                                   pad_to_max_length=True)
        assert results["input_ids"].shape[0] == batch_size * num_choice, \
            f"Invalid shapes {results['input_ids'].shape} {batch_size, num_choice}"

        batch["input_ids"] = results["input_ids"]
        batch["attention_mask"] = results["attention_mask"]
        batch["token_type_ids"] = results["token_type_ids"]

        # TODO: provide associated tree ids here
        batch['additional_position_ids'] = torch.zeros_like(results["input_ids"])

        return batch

    # Data loader methods to return train and validation data sets
    def train_dataloader(self):
        dataloader = DataLoader(self.dataloader(self.root_path / self.hparams["train_x"],
                                                self.root_path / self.hparams["train_y"],
                                                data_slice=self.hparams["train_data_slice"]),
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

    # Extend PyTorch Lightning methods
    def training_step(self, batch, batch_idx, task_id=None):
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

    def validation_step(self, batch, batch_idx, task_id=None):
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
        optimizer = AdamW(self.parameters(), lr=float(self.hparams["learning_rate"]),
                          eps=float(self.hparams["adam_epsilon"]))
        return optimizer
