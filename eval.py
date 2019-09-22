import os
import sys

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.trainer_io import load_hparams_from_tags_csv
from pytorch_lightning.utilities.arg_parse import add_default_args
from test_tube import HyperOptArgumentParser

from ai2.model import HuggingFaceClassifier
from train import set_seed


def load_from_metrics(hparams, model_cls, weights_path, tags_csv, on_gpu, map_location=None):
    prev_hparams = load_hparams_from_tags_csv(tags_csv)
    prev_hparams.__dict__.update(hparams.__dict__)
    hparams.__dict__.update({k: v for k, v in prev_hparams.__dict__.items() if k not in hparams.__dict__})
    hparams.__setattr__('on_gpu', on_gpu)

    if on_gpu:
        if map_location is not None:
            checkpoint = torch.load(weights_path, map_location=map_location)
        else:
            checkpoint = torch.load(weights_path)
    else:
        checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)

    running_config = yaml.safe_load(open(hparams.running_config_file, "r"))
    task_config = yaml.safe_load(open(hparams.task_config_file, 'r'))

    hparams.max_nb_epochs = running_config[hparams.task_name].get(
        hparams.model_type, {}).get(
        hparams.model_weight, running_config[hparams.task_name]['default']).get('max_epochs', 3)

    hparams.learning_rate = float(running_config[hparams.task_name].get(
        hparams.model_type, {}).get(
        hparams.model_weight, running_config[hparams.task_name]['default']).get('lr', 2e-5))

    hparams.initializer_range = float(running_config[hparams.task_name].get(
        hparams.model_type, {}).get(
        hparams.model_weight, running_config[hparams.task_name]['default']).get('initializer_range', 0.02))

    hparams.dropout = float(running_config[hparams.task_name].get(
        hparams.model_type, {}).get(
        hparams.model_weight, running_config[hparams.task_name]['default']).get('dropout', 0.1))

    hparams.batch_size = running_config[hparams.task_name].get(
        hparams.model_type, {}).get(
        hparams.model_weight, running_config[hparams.task_name]['default']).get('batch_size', 32)

    hparams.max_seq_len = running_config[hparams.task_name].get(
        hparams.model_type, {}).get(
        hparams.model_weight, running_config[hparams.task_name]['default']).get('max_seq_len', 128)

    hparams.seed = running_config[hparams.task_name].get(
        hparams.model_type, {}).get(
        hparams.model_weight, running_config[hparams.task_name]['default']).get('seed', 42)

    hparams.weight_decay = float(running_config[hparams.task_name].get(
        hparams.model_type, {}).get(
        hparams.model_weight, running_config[hparams.task_name]['default']).get('weight_decay', 0.0))

    hparams.warmup_steps = running_config[hparams.task_name].get(
        hparams.model_type, {}).get(
        hparams.model_weight, running_config[hparams.task_name]['default']).get('warmup_steps', 0)

    hparams.adam_epsilon = float(running_config[hparams.task_name].get(
        hparams.model_type, {}).get(
        hparams.model_weight, running_config[hparams.task_name]['default']).get('adam_epsilon', 1e-8))
    hparams.accumulate_grad_batches = running_config[hparams.task_name].get(
        hparams.model_type, {}).get(
        hparams.model_weight, running_config[hparams.task_name]['default']).get('accumulate_grad_batches', 1)

    hparams.do_lower_case = task_config[hparams.task_name].get('do_lower_case', False)

    hparams.tokenizer_type = hparams.model_type if hparams.tokenizer_type is None else hparams.tokenizer_type
    hparams.tokenizer_weight = hparams.model_weight if hparams.tokenizer_weight is None else hparams.tokenizer_weight

    set_seed(hparams.seed)

    model = model_cls(hparams)
    model.load_state_dict(checkpoint['state_dict'])

    model.on_load_checkpoint(checkpoint)

    return model


def main(hparams):
    model = load_from_metrics(
        hparams=hparams,
        model_cls=HuggingFaceClassifier,
        weights_path=hparams.weights_path,
        tags_csv=hparams.tags_csv,
        on_gpu=torch.cuda.is_available(),
        map_location=None
    )

    trainer = Trainer()

    trainer.test(model)


if __name__ == '__main__':
    root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
    parent_parser = HyperOptArgumentParser(strategy='random_search', add_help=True)
    add_default_args(parent_parser, root_dir)

    parser = HuggingFaceClassifier.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    main(hyperparams)
