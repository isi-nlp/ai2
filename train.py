import os
import pathlib
import sys

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.arg_parse import add_default_args
from test_tube import HyperOptArgumentParser, Experiment

from ai2.model import HuggingFaceClassifier


def main(hparams):
    curr_dir = "output"

    log_dir = os.path.join(curr_dir, hparams.task_name, f"{hparams.model_type}-{hparams.model_weight}-log")
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    hparams.tokenizer_type = hparams.model_type if hparams.tokenizer_type is None else hparams.tokenizer_type
    hparams.tokenizer_weight = hparams.model_weight if hparams.tokenizer_weight is None else hparams.tokenizer_weight

    exp = Experiment(
        name=hparams.task_name,
        version=0,
        save_dir=log_dir,
        autosave=True,
    )

    running_config = yaml.safe_load(open(hparams.running_config_file, "r"))

    # set the hparams for the experiment
    exp.argparse(hparams)
    exp.save()

    # build model
    model = HuggingFaceClassifier(hparams)

    # callbacks
    early_stop = EarlyStopping(
        monitor=hparams.early_stop_metric,
        patience=hparams.early_stop_patience,
        verbose=True,
        mode=hparams.early_stop_mode
    )

    model_save_path = os.path.join(curr_dir, hparams.task_name,
                                   f"{hparams.model_type}-{hparams.model_weight}-checkpoints", str(exp.version))

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
        gpus=[i for i in range(torch.cuda.device_count())],
        show_progress_bar=True,
        overfit_pct=0.0,
        track_grad_norm=hparams.track_grad_norm,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
        fast_dev_run=hparams.fast_dev_run,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        max_nb_epochs=running_config.get(
            hparams.model_type, {}).get(
            hparams.model_weight, running_config['default']).get('max_epochs'),
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

    parser = HuggingFaceClassifier.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    main(hyperparams)
