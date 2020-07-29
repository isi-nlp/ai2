"""Callback for auto-resuming jobs to bypass the Slurm time limit.

To use the callback, include it in the list provided to `callbacks` when instantiating the
`Trainer` class.
"""
from typing import NoReturn

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.base import Callback


class AutoResume(Callback):
    """This callback exits the program and requeues it on the Slurm cluster after each epoch."""

    def __init__(self):
        pass

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> NoReturn:
        """Requeues the current program when the epoch ends.

        Args:
            trainer: Trainer running callback.
            pl_module: Module being trained.
        """
        trainer.sig_handler(None, None)
        exit()
