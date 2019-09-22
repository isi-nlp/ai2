import os
import sys

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.trainer_io import load_hparams_from_tags_csv
from pytorch_lightning.utilities.arg_parse import add_default_args
from test_tube import HyperOptArgumentParser

from ai2.model import HuggingFaceClassifier
from train import set_seed, get_default


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

    hparams.max_nb_epochs = get_default(running_config, hparams.task_name, hparams.model_type, hparams.model_weight,
                                        'max_nb_epochs')

    hparams.learning_rate = float(
        get_default(running_config, hparams.task_name, hparams.model_type, hparams.model_weight,
                    'lr'))

    hparams.initializer_range = float(
        get_default(running_config, hparams.task_name, hparams.model_type, hparams.model_weight,
                    'initializer_range'))

    hparams.dropout = float(
        get_default(running_config, hparams.task_name, hparams.model_type, hparams.model_weight,
                    'dropout'))

    hparams.batch_size = get_default(running_config, hparams.task_name, hparams.model_type, hparams.model_weight,
                                     'batch_size')

    hparams.max_seq_len = get_default(running_config, hparams.task_name, hparams.model_type, hparams.model_weight,
                                      'max_seq_len')

    hparams.seed = get_default(running_config, hparams.task_name, hparams.model_type, hparams.model_weight,
                               'seed')

    hparams.weight_decay = float(
        get_default(running_config, hparams.task_name, hparams.model_type, hparams.model_weight,
                    'weight_decay'))

    hparams.warmup_steps = get_default(running_config, hparams.task_name, hparams.model_type, hparams.model_weight,
                                       'warmup_steps')

    hparams.adam_epsilon = float(
        get_default(running_config, hparams.task_name, hparams.model_type, hparams.model_weight,
                    'adam_epsilon'))

    hparams.accumulate_grad_batches = get_default(running_config, hparams.task_name, hparams.model_type,
                                                  hparams.model_weight,
                                                  'accumulate_grad_batches')

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
