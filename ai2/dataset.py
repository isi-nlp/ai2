from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
import glob

from collections import defaultdict
from dataclasses import dataclass
from itertools import zip_longest
from typing import *
from tqdm import tqdm
import numpy as np
import requests
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from loguru import logger


def download(urls: Union[str, List[str]], cache_dir: str) -> Union[str, List[str]]:

    if isinstance(urls, str):
        filename = urls.split('/')[-1]
        filename = filename.split('.')[0]
        target_dir = os.path.join(cache_dir, filename)

        if not os.path.exists(target_dir) or not os.listdir(target_dir):
            logger.debug(f"Download dataset from {urls} to {target_dir}")
            r = requests.get(urls)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(target_dir)

        return target_dir

    elif isinstance(urls, list):
        cache_dirs = []
        for url in urls:
            filename = url.split('/')[-1]
            filename = filename.split('.')[0]
            target_dir = os.path.join(cache_dir, filename)

            if not os.path.exists(target_dir) or not os.listdir(target_dir):
                logger.debug(f"Download dataset from {url} to {target_dir}")
                r = requests.get(url)
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(target_dir)

            cache_dirs.append(target_dir)
        return cache_dirs


@dataclass
class AI2Dataset(Dataset):

    tokens: List[List[str]]
    input_ids: List[List[int]]
    token_type_ids: List[List[int]]
    attention_mask: List[List[int]]
    y: List[int] = None

    def __len__(self):
        return len(self.tokens)

    @classmethod
    def load(
            cls, cache_dir: str, file_mapping: Dict, task_formula: str, type_formula: str, preprocessor: TokenizerLoader, pretokenized: bool = False,
            label_formula: str = None, label_offset: int = 0) -> AI2Dataset:
        """Load the dataset from a directory.

        Arguments:
            cache_dir {str} -- directory where the dataset is stored.
            file_mapping {Dict} -- Mapping between the dataset name and the file. {'input_x': 'filex', 'input_y': 'filey'}  
            task_formula {str} -- Task formula from the config. 
            type_formula {str} -- Type formula for token_type_ids.
            preprocessor {TokenizerLoader} -- TokenizerLoader.

        Keyword Arguments:
            pretokenized {bool} -- Whether the input is pretokenized or not. (default: {False})
            label_formula {str} -- Label field if the input y is a json file. (default: {None})
            label_offset {int} -- Label offset. (default: {0})

        Returns:
            AI2Dataset -- [description]
        """

        assert len(file_mapping) <= 2, "At most two files can be specified"

        x = [k for k in file_mapping.keys() if k.endswith('_x')]
        y = [k for k in file_mapping.keys() if k.endswith('_y')]

        assert x != [], "Mapping should have x"

        x = x[0]
        y = y[0] if y != [] else None

        task_formula_mapping = [
            token if '|' not in token else token.split('|') for token in task_formula.split(' ')
        ]

        type_formula_mapping = list(map(int, type_formula.split(' ')))

        if file_mapping[x] is None:
            x = [f for f in glob.glob(f"{cache_dir}/*.jsonl")]
            assert len(x) == 1, f"Multiple input files found in cache_dir {x}"
            x = x[0]
        else:
            x = os.path.join(cache_dir, file_mapping[x])

        with open(x) as input_file:
            tokens = []
            token_type_ids = []

            for line in tqdm(input_file.readlines()):

                example_raw = json.loads(line.strip('\r\n ').replace('\n', ''))

                if pretokenized:
                    for k in example_raw:
                        if isinstance(example_raw[k], list):
                            if all(isinstance(i, list) for i in example_raw[k]):
                                example_raw[k] = [' '.join(map(str, s)) for s in example_raw[k]]
                            else:
                                example_raw[k] = ' '.join(map(str, example_raw[k]))

                example = [[]]
                example_token_type_ids = [[]]

                for i, segment in zip(type_formula_mapping, task_formula_mapping):

                    if isinstance(segment, str):

                        if segment.startswith('[') and segment.endswith(']'):
                            example = [e + [getattr(preprocessor, segment.strip('[]'))] for e in example]
                            example_token_type_ids = [e + [i] for e in example_token_type_ids]

                        elif isinstance(example_raw[segment], str):
                            example_tokens = preprocessor.tokenize(example_raw[segment])
                            example = [e + example_tokens for e in example]
                            example_token_type_ids = [e + [i for _ in example_tokens] for e in example_token_type_ids]
                        elif isinstance(example_raw[segment], list):
                            example_tokens = [preprocessor.tokenize(k) for k in example_raw[segment]]
                            example = [e + t for t in example_tokens for e in example]
                            example_token_type_ids = [e + [i for _ in t]
                                                      for t in example_tokens for e in example_token_type_ids]

                    elif isinstance(segment, list):
                        example_tokens = [preprocessor.tokenize(example_raw[k]) for k in segment]
                        example = [e + t for t in example_tokens for e in example]
                        example_token_type_ids = [e + [i for _ in t]
                                                  for t in example_tokens for e in example_token_type_ids]

                tokens.append(example)
                token_type_ids.append(example_token_type_ids)

        labels = None
        if y:
            labels = []
            with open(os.path.join(cache_dir, file_mapping[y])) as input_file:
                for line in input_file:
                    if label_formula is not None:
                        labels.append(int(json.loads(line.strip('\r\n ').replace('\n', ' '))[label_formula]) - label_offset)
                    else:
                        labels.append(int(line) - label_offset)

        input_ids = [[preprocessor.tokens2ids(ee) for ee in e] for e in tokens]
        attention_mask = [[[1 for _ in ee] for ee in e] for e in tokens]

        logger.info(f"""
            {x}
            Total number of examples: {len(tokens)}
            Average input length: {sum(map(lambda e: sum(map(len, e)), tokens))//sum(map(len, tokens))}
            Maximum input length: {max(map(lambda e: max(map(len, e)), tokens))}
            99 % of input length: {sorted(map(lambda e: max(map(len, e)), tokens))[int(len(tokens)*.99)]}
        """)

        return AI2Dataset(tokens, input_ids, token_type_ids, attention_mask, labels)

    def __getitem__(self, index) -> Dict:

        return {
            'tokens': self.tokens[index],
            'input_ids': self.input_ids[index],
            'token_type_ids': self.token_type_ids[index],
            'attention_mask': self.attention_mask[index],
            'y': self.y[index] if self.y is not None else None
        }


if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser("Dataset download script.")
    parser.add_argument('--task_config', type=str, default='./tasks.yaml')
    parser.add_argument('--task', type=str, choices=['alphanli', 'hellaswag', 'physicaliqa',
                                                     'socialiqa', 'vcrqa', 'vcrqr', 'all'], default='alphanli')
    parser.add_argument('--cache_dir', type=str, required=False, default='./cache')

    args = parser.parse_args()

    with open(args.task_config, 'r') as input_file:
        config = yaml.safe_load(input_file)

    if args.task == 'all':
        for task in ['alphanli', 'hellaswag', 'physicaliqa', 'socialiqa', 'vcrqa', 'vcrqr']:
            cache_dirs = download(config[task]['urls'], args.cache_dir)
    else:
        cache_dirs = download(config[args.task]['urls'], args.cache_dir)
