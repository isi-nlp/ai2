#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-06 09:17:36
# @Author  : Chenghao Mou (chengham@isi.edu)
# @Link    : https://github.com/ChenghaoMou/ai2

# pylint: disable=unused-wildcard-import
# pylint: disable=no-member

from __future__ import annotations

import glob
import io
import json
import os
import io
import random
import zipfile
import requests
from dataclasses import dataclass
from typing import *

from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm

from textbook.interface import TokenizerLoader


def download(urls: Union[str, List[str]], cache_dir: str) -> Union[str, List[str]]:
    """
    Download ai2 datasets from urls.

    :param urls: List of urls or single url
    :param cache_dir: path where to store the dataset.
    :return: single path or list of paths.
    """
    if isinstance(urls, str):
        filename = urls.split('/')[-1]
        filename = filename.replace('.zip', '').replace('.tar.gz', '')
        target_dir = os.path.join(cache_dir, filename)

        if not os.path.exists(target_dir) or not os.listdir(target_dir):
            logger.debug(f"Download dataset from {urls} to {target_dir}")
            r = requests.get(urls)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(target_dir)
        else:
            logger.debug(f"Dataset is already downloaded: {filename}")

        return target_dir

    elif isinstance(urls, list):
        cache_dirs = [cache_dir] if urls == [] else []
        for url in urls:
            filename = url.split('/')[-1]
            filename = filename.replace('.zip', '').replace('.tar.gz', '')
            target_dir = os.path.join(cache_dir, filename)

            if not os.path.exists(target_dir) or not os.listdir(target_dir):
                logger.debug(f"Download dataset from {url} to {target_dir}")
                r = requests.get(url)
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(target_dir)
            else:
                logger.debug(f"Dataset is already downloaded: {filename}")
            cache_dirs.append(target_dir)
        return cache_dirs


@dataclass
class ClassificationDataset(Dataset):
    tokens: List[List[str]]
    input_ids: List[List[int]]
    token_type_ids: List[List[int]]
    attention_mask: List[List[int]]
    y: List[int] = None

    def __len__(self):
        return len(self.tokens)

    @classmethod
    def load(
            cls, cache_dir: str, file_mapping: Dict, task_formula: str, type_formula: str,
            preprocessor: TokenizerLoader, pretokenized: bool = False,
            label_formula: str = None, label_offset: int = 0, label_transform: Dict = None, shuffle: bool = False) -> ClassificationDataset:
        """
        Load the datase into a dataset class wrapper.

        :param cache_dir: Path to a dataset.
        :param file_mapping: Dictionary where train_x/train_y/dev_x/dev_y are mapped to each file in the cache directory.
        :param task_formula: How to formulate a task, configured in the tasks.yaml.
        :param type_formula: Token type id formulation, configured in the tasks.yaml.
        :param preprocessor: TokenizerLoader object.
        :param pretokenized: Whether the field in task formula is pretokenized or not.
        :param label_formula: None if label is an integer else a field in the dataset.
        :param label_offset: Offset for integer labels.
        :param label_transform: Mapping from string lables to integer labels.
        :param shuffle: Shuffle the tokens.
        :return: A ClassificationDataset.
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
                line = line.strip('\r\n ').replace('\n', '')
                if not line:
                    continue
                # print(line)
                example_raw = json.loads(line)
                if not y and label_formula is not None and label_formula not in example_raw:
                    continue
                # print(example_raw)
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
                        # print(segment)
                        if segment.startswith('[') and segment.endswith(']'):
                            special_token = getattr(preprocessor, segment.strip('[]'))
                            if special_token:
                                example = [e + [special_token] for e in example]
                                example_token_type_ids = [e + [i] for e in example_token_type_ids]

                        elif segment in example_raw and isinstance(example_raw[segment], str):
                            example_tokens = preprocessor.tokenize(example_raw[segment])
                            if shuffle:
                                random.shuffle(example_tokens)

                            example = [e + example_tokens for e in example]
                            example_token_type_ids = [e + [i for _ in example_tokens] for e in example_token_type_ids]
                        elif segment in example_raw and isinstance(example_raw[segment], list):

                            example_tokens = [preprocessor.tokenize(k) for k in example_raw[segment]]
                            if shuffle:
                                for i in range(len(example_tokens)):
                                    random.shuffle(example_tokens[i])

                            example = [e + t for t in example_tokens for e in example]
                            example_token_type_ids = [e + [i for _ in t]
                                                      for t in example_tokens for e in example_token_type_ids]
                        else:
                            logger.debug(str(example_raw))
                            continue
                    elif isinstance(segment, list):
                        example_tokens = [preprocessor.tokenize(example_raw[k]) for k in segment]
                        example = [e + t for t in example_tokens for e in example]
                        example_token_type_ids = [e + [i for _ in t]
                                                  for t in example_tokens for e in example_token_type_ids]
                # print(example)
                tokens.append(example)
                token_type_ids.append(example_token_type_ids)

        labels = None
        if y:
            labels = []
            with open(os.path.join(cache_dir, file_mapping[y])) as input_file:
                for line in input_file:
                    jd = json.loads(line.strip('\r\n ').replace('\n', ' '))
                    if label_formula is not None and label_formula in jd:
                        if label_transform is None:
                            labels.append(
                                int(json.loads(line.strip('\r\n ').replace('\n', ' '))[label_formula]) - label_offset)
                        else:
                            labels.append(
                                label_transform[
                                    json.loads(line.strip('\r\n ').replace('\n', ' '))[label_formula]] - label_offset)
                    elif label_formula is None:
                        labels.append(int(line) - label_offset)
                    else:
                        logger.debug(jd)
                        continue
            assert len(labels) == len(tokens), f"{len(labels)}, {len(tokens)}"

        input_ids = [[preprocessor.tokens2ids(ee) for ee in e] for e in tokens]
        attention_mask = [[[1 for _ in ee] for ee in e] for e in tokens]

        logger.info(f"""
            {x}
            Total number of examples: {len(tokens)}
            Average input length: {sum(map(lambda e: sum(map(len, e)), tokens)) // sum(map(len, tokens))}
            Maximum input length: {max(map(lambda e: max(map(len, e)), tokens))}
            99 % of input length: {sorted(map(lambda e: max(map(len, e)), tokens))[int(len(tokens) * .99)]}
        """)

        return ClassificationDataset(tokens, input_ids, token_type_ids, attention_mask, labels)

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

    tasks = ['alphanli', 'hellaswag', 'physicaliqa', 'socialiqa', 'vcrqa', 'vcrqr']

    parser = argparse.ArgumentParser("Dataset download script.")
    parser.add_argument('--task_config', type=str, default='./tasks.yaml')
    parser.add_argument('--task', type=str, choices=tasks + ["all"], default='alphanli')
    parser.add_argument('--cache_dir', type=str, required=False, default='./cache')

    args = parser.parse_args()

    with open(args.task_config, 'r') as input_file:
        config = yaml.safe_load(input_file)

    if args.task == 'all':
        for task in tasks:
            cache_dirs = download(config[task]['urls'], args.cache_dir)
    else:
        cache_dirs = download(config[args.task]['urls'], args.cache_dir)
