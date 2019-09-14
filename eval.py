

import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.arg_parse import add_default_args
from test_tube import HyperOptArgumentParser, Experiment
from ai2.model import HuggingFaceClassifier
from pytorch_lightning.root_module.model_saving import load_hparams_from_tags_csv
import torch
import yaml
import sys
from copy import deepcopy


def load_from_metrics(hparams, model_cls, weights_path, tags_csv, on_gpu, map_location=None):

    prev_hparams = load_hparams_from_tags_csv(tags_csv)
    prev_hparams.__dict__.update(hparams.__dict__)
    hparams.__dict__.update({k: v for k, v in prev_hparams.__dict__.items() if k not in hparams.__dict__})
    hparams.__setattr__('on_gpu', on_gpu)

    hparams.tokenizer_type = hparams.model_type if hparams.tokenizer_type is None else hparams.tokenizer_type
    hparams.tokenizer_weight = hparams.model_weight if hparams.tokenizer_weight is None else hparams.tokenizer_weight

    if on_gpu:
        if map_location is not None:
            checkpoint = torch.load(weights_path, map_location=map_location)
        else:
            checkpoint = torch.load(weights_path)
    else:
        checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)

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
