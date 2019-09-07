# encoding: utf-8
# Created by chenghaomou at 9/7/19
# Contact: mouchenghao at gmail dot com
# Description: Finetuning a model
import os
import sys

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.arg_parse import add_default_args
from test_tube import HyperOptArgumentParser, Experiment

from ai2.model import Classifier


def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    :return:
    """
    # init experiment
    log_dir = os.path.dirname(os.path.realpath(__file__))
    exp = Experiment(
        name=hparams.task_name,
        debug=True,
        save_dir=log_dir,
        version=0,
        autosave=False,
        description='-'.join([hparams.task_name, hparams.model_type, hparams.model_weight])
    )

    # set the hparams for the experiment
    exp.argparse(hparams)
    exp.save()

    # build model
    model = Classifier(hparams)

    # callbacks
    early_stop = EarlyStopping(
        monitor=hparams.early_stop_metric,
        patience=hparams.early_stop_patience,
        verbose=True,
        mode=hparams.early_stop_mode
    )

    model_save_path = '{}/{}/{}'.format(hparams.model_save_path, exp.name, exp.version)
    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        save_best_only=False,
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
        nb_gpu_nodes=1,
        gpus=hparams.gpus,
        max_nb_epochs=hparams.max_nb_epochs,
        val_check_interval=hparams.val_check_interval,
    )

    # train model
    trainer.fit(model)


if __name__ == '__main__':
    # use default args given by lightning
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parent_parser = HyperOptArgumentParser(strategy='random_search', add_help=True)
    add_default_args(parent_parser, root_dir)

    # allow model to overwrite or extend args
    parser = Classifier.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    print(hyperparams)

    # train model
    main(hyperparams)
