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

from ai2.model import HuggingFaceClassifier


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main(hparams):
    curr_dir = "output"

    log_dir = os.path.join(curr_dir, f"{hparams.model_type}-{hparams.model_weight}-log")
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    exp = Experiment(
        name=hparams.task_name,
        version=0,
        save_dir=log_dir,
        autosave=True,
    )

    model_save_path = os.path.join(curr_dir,
                                   f"{hparams.model_type}-{hparams.model_weight}-checkpoints", hparams.task_name,
                                   str(exp.version))

    running_config = yaml.safe_load(open(hparams.running_config_file, "r"))
    task_config = yaml.safe_load(open(hparams.task_config_file, 'r'))

    hparams.max_nb_epochs = running_config.get(
        hparams.model_type, {}).get(
        hparams.model_weight, running_config['default']).get('max_epochs', 3)

    hparams.learning_rate = float(running_config.get(
        hparams.model_type, {}).get(
        hparams.model_weight, running_config['default']).get('lr', 2e-5))

    hparams.initializer_range = float(running_config.get(
        hparams.model_type, {}).get(
        hparams.model_weight, running_config['default']).get('initializer_range', 0.02))

    hparams.dropout = float(running_config.get(
        hparams.model_type, {}).get(
        hparams.model_weight, running_config['default']).get('dropout', 0.1))

    hparams.batch_size = running_config.get(
        hparams.model_type, {}).get(
        hparams.model_weight, running_config['default']).get('batch_size', 32)

    hparams.max_seq_len = running_config.get(
        hparams.model_type, {}).get(
        hparams.model_weight, running_config['default']).get('max_seq_len', 128)

    hparams.do_lower_case = task_config[hparams.task_name].get('do_lower_case', False)
    hparams.output_dimension = task_config[hparams.task_name].get('output_dimension', 1)

    hparams.tokenizer_type = hparams.model_type if hparams.tokenizer_type is None else hparams.tokenizer_type
    hparams.tokenizer_weight = hparams.model_weight if hparams.tokenizer_weight is None else hparams.tokenizer_weight

    hparams.seed = running_config.get(
        hparams.model_type, {}).get(
        hparams.model_weight, running_config['default']).get('seed', 42)

    hparams.weight_decay = float(running_config.get(
        hparams.model_type, {}).get(
        hparams.model_weight, running_config['default']).get('weight_decay', 0.0))

    hparams.warmup_steps = running_config.get(
        hparams.model_type, {}).get(
        hparams.model_weight, running_config['default']).get('warmup_steps', 0)

    hparams.adam_epsilon = float(running_config.get(
        hparams.model_type, {}).get(
        hparams.model_weight, running_config['default']).get('adam_epsilon', 1e-8))

    hparams.accumulate_grad_batches = float(running_config.get(
        hparams.model_type, {}).get(
        hparams.model_weight, running_config['default']).get('accumulate_grad_batches', 1))

    hparams.model_save_path = model_save_path

    logger.info(f"{hparams}")

    # set the hparams for the experiment
    exp.argparse(hparams)
    exp.save()

    set_seed(hparams.seed)

    # build model
    model = HuggingFaceClassifier(hparams)

    # callbacks
    early_stop = EarlyStopping(
        monitor=hparams.early_stop_metric,
        patience=hparams.early_stop_patience,
        verbose=True,
        mode=hparams.early_stop_mode
    )

    checkpoint = ModelCheckpoint(
        filepath=hparams.model_save_path,
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

    parser = HuggingFaceClassifier.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    main(hyperparams)
