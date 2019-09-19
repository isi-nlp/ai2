import requests
from loguru import logger
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import *
import torch
import os
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pytorch_lightning.root_module.root_module import LightningModule
from sklearn.metrics import accuracy_score
from test_tube import HyperOptArgumentParser
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import os
import pathlib
import sys

import torch
import yaml
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.arg_parse import add_default_args
from test_tube import HyperOptArgumentParser, Experiment
from eval import load_from_metrics


@dataclass
class JointDataset(Dataset):

    x: Union[List[List[float]], torch.Tensor]
    y: Optional[Union[List[int], torch.Tensor]] = None

    @classmethod
    def load(cls, x_paths: List[str], y_path: str) -> JointDataset:

        dataset = []
        labels = []

        for x_path in x_paths:

            with open(x_path) as f:
                for i, line in enumerate(f):
                    if len(dataset) <= i:
                        dataset.append([])
                    dataset[i].extend(map(floar, line.strip('\r\n ').split('\t')))
        if y_path:
            with open(y_path) as f:
                for line in f:
                    labels.append(int(line.strip('\r\n ')))

        return ProbabilityDataset(dataset, labels)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):

        return {
            'x': self.x[index],
            'y': self.y[index]
        }


class JointModel(LightningModule):

    def __init__(self, hparams):

        super(JointModel, self).__init__()
        self.hparams = hparams

        if not os.path.exists(self.hparams.output_dir):
            os.mkdir(self.hparams.output_dir)

        self.model = nn.Seuqential(
            nn.Linear(self.hparams.input_dimension, self.hparams.hidden_dimension),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_dimension, self.hparams.output_dimension),
        )

        self.linear.weight.data.normal_(mean=0.0, std=self.hparams.initializer_range)
        self.linear.bias.data.zero_()

    def forward(self, input_tensor):

        logits = self.model.forward(input_tensor)

        return logits.squeeze()

    def loss(self, labels, logits):
        l = F.cross_entropy(logits, labels, reduction='mean')
        return l

    def training_step(self, data_batch, batch_i):

        B, H = data_batch['input_tensor'].shape

        logits = self.forward(**{
            'input_tensor': data_batch['input_tensor'].reshape(-1, H),
        })
        loss_val = self.loss(data_batch['y'].reshape(-1), logits.reshape(B, -1))

        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)

        return {
            'logits': logits.reshape(B, -1),
            'loss': loss_val
        }

    def validation_step(self, data_batch, batch_i):
        B, H = data_batch['input_tensor'].shape

        logits = self.forward(**{
            'input_tensor': data_batch['input_tensor'].reshape(-1, H),
        })
        loss_val = self.loss(data_batch['y'].reshape(-1), logits.reshape(B, -1))

        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)

        return {
            'batch_logits': logits.reshape(B, -1),
            'batch_loss': loss_val,
            'batch_truth': data_batch['y']
        }

    def test_step(self, data_batch, batch_i):
        B, H = data_batch['input_tensor'].shape

        logits = self.forward(**{
            'input_tensor': data_batch['input_tensor'].reshape(-1, H),
        })

        return {
            'batch_logits': logits.reshape(B, -1),
        }

    def validation_end(self, outputs):
        truth = torch.cat([o['batch_truth'] for o in outputs], dim=0).reshape(-1)
        logits = torch.cat([o['batch_logits'] for o in outputs], dim=0).reshape(len(truth),
                                                                                outputs[0]['batch_logits'].shape[1])

        loss = self.loss(truth, logits)
        proba = F.softmax(logits, dim=-1)
        pred = torch.argmax(proba, dim=-1).reshape(-1)

        with open(os.path.join(self.hparams.output_dir, "dev-labels.lst"), "w") as output_file:
            output_file.write("\n".join(map(str, (truth + self.hparams.offset).cpu().numpy().tolist())))

        with open(os.path.join(self.hparams.output_dir, "dev-predictions.lst"), "w") as output_file:
            output_file.write("\n".join(
                map(str, (pred + self.hparams.offset).cpu().numpy().tolist())))

        with open(os.path.join(self.hparams.output_dir, "dev-probabilities.lst"), "w") as output_file:
            output_file.write("\n".join(map(lambda l: '\t'.join(map(str, l)), proba.cpu().detach().numpy().tolist())))

        return {
            'val_loss': loss.item(),
            'val_acc': accuracy_score(truth.cpu().detach().numpy().tolist(), pred.cpu().detach().numpy().tolist()),
        }

    def test_end(self, outputs):
        """
        Outputs has the appended output after each test step
        OPTIONAL
        :param outputs:
        :return: dic_with_metrics for tqdm
        """
        logits = torch.cat([o['batch_logits'] for o in outputs], dim=0).reshape(-1, outputs[0]['batch_logits'].shape[1])
        proba = F.softmax(logits, dim=-1)
        pred = torch.argmax(proba, dim=-1).reshape(-1)

        with open(os.path.join(self.hparams.output_dir, "predictions.lst"), "w") as output_file:
            output_file.write("\n".join(map(str, (pred + self.hparams.offset).cpu().detach().numpy().tolist())))

        with open(os.path.join(self.hparams.output_dir, "probabilities.lst"), "w") as output_file:
            output_file.write("\n".join(map(lambda l: '\t'.join(map(str, l)), proba.cpu().detach().numpy().tolist())))

        return {}

    def configure_optimizers(self):

        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @pl.data_loader
    def tng_dataloader(self):

        dataset = ProbabilityDataset.load(self.hparams.input_x, self.hparams.input_y)
        return DataLoader(dataset,
                          shuffle=True,
                          batch_size=self.hparams.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        dataset = ProbabilityDataset.load(self.hparams.dev_input_x, self.hparams.dev_input_y)
        return DataLoader(dataset,
                          shuffle=False,
                          batch_size=self.hparams.batch_size)

    @pl.data_loader
    def test_dataloader(self):

        if self.hparams.test_input_x is None:
            return self.val_dataloader

        dataset = ProbabilityDataset.load(self.hparams.test_input_x, None)
        return DataLoader(dataset,
                          shuffle=False,
                          batch_size=self.hparams.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover

        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser], add_help=False)

        # param overwrites
        parser.set_defaults(gradient_clip=5.0,
                            model_save_monitor_value='val_acc',
                            model_save_monitor_mode='max',
                            early_stop_metric='val_loss',
                            early_stop_patience=10,
                            early_stop_mode='min',
                            val_check_interval=0.02,
                            max_nb_epochs=300
                            )
        parser.add_argument('--input_dimention', type=int, required=True)
        parser.add_argument('--hidden_dimension', type=int, required=True)
        parser.add_argument('--output_dimension', type=int, required=True)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--initializer_range', type=float, default=0.02)
        parser.add_argument('--offset', type=int, required=True)
        parser.add_argument('--learning_rate', type=float, default=1)

        parser.add_argument('--input_x', nargs='+', required=True)
        parser.add_argument('--input_y', type=str, required=True)
        parser.add_argument('--dev_input_x', nargs='+', required=True)
        parser.add_argument('--dev_input_y', type=str, required=True)
        parser.add_argument('--test_input_x', nargs='+', required=False, default=None)

        parser.add_argument('--output_dir', type=str, required=True, default=None)

        parser.add_argument('--weights_path', type=str, required=False, default=None)
        parser.add_argument('--tags_csv', type=str, required=False, default=None)

        return parser


def main(hparams):

    if hparams.test_input_x is not None:
        model = load_from_metrics(
            hparams=hparams,
            model_cls=JointModel,
            weights_path=hparams.weights_path,
            tags_csv=hparams.tags_csv,
            on_gpu=torch.cuda.is_available(),
            map_location=None
        )

        trainer = Trainer()
        trainer.test(model)

    else:

        curr_dir = "joint-output"

        log_dir = os.path.join(curr_dir, f"log")
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

        exp = Experiment(
            name=f"joint-{hparams.task_name}",
            version=0,
            save_dir=log_dir,
            autosave=True,
        )

        logger.info(f"{hparams}")

        exp.argparse(hparams)
        exp.save()

        # build model
        model = JointModel(hparams)

        # callbacks
        early_stop = EarlyStopping(
            monitor=hparams.early_stop_metric,
            patience=hparams.early_stop_patience,
            verbose=True,
            mode=hparams.early_stop_mode
        )

        model_save_path = os.path.join(curr_dir,
                                       f"joint-{hparams.task_name}-checkpoints",
                                       str(exp.version))

        checkpoint = ModelCheckpoint(
            filepath=model_save_path,
            save_best_only=True,
            verbose=True,
            monitor=hparams.model_save_monitor_value,
            mode=hparams.model_save_monitor_mode
        )

        # configure trainer
        trainer = Trainer(
            experiment=exp,
            checkpoint_callback=checkpoint,
            early_stop_callback=early_stop,
            gradient_clip=hparams.gradient_clip,
            process_position=0,
            nb_gpu_nodes=1,
            gpus=[i for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None,
            show_progress_bar=True,
            overfit_pct=0.0,
            track_grad_norm=hparams.track_grad_norm,
            check_val_every_n_epoch=hparams.check_val_every_n_epoch,
            fast_dev_run=hparams.fast_dev_run,
            accumulate_grad_batches=hparams.accumulate_grad_batches,
            max_nb_epochs=hparams.max_nb_epochs,
            min_nb_epochs=hparams.min_nb_epochs,
            train_percent_check=hparams.train_percent_check,
            val_percent_check=hparams.val_percent_check,
            test_percent_check=hparams.val_percent_check,
            val_check_interval=hparams.val_check_interval,
            log_save_interval=hparams.log_save_interval,
            add_log_row_interval=hparams.add_log_row_interval,
            distributed_backend='dp',
            use_amp=hparams.use_amp,
            print_nan_grads=hparams.check_grad_nans,
            print_weights_summary=True,
            amp_level=hparams.amp_level,
            nb_sanity_val_steps=5,
        )

        # train model
        trainer.fit(model)


if __name__ == '__main__':
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parent_parser = HyperOptArgumentParser(strategy='random_search', add_help=True)
    add_default_args(parent_parser, root_dir)

    parser = JointModel.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    main(hyperparams)
