# encoding: utf-8
# Created by chenghaomou at 9/7/19
# Contact: mouchenghao at gmail dot com
# Description: Utility functions for data preprocess and loading.
from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
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

from ai2.interface import BaseTokenizer


def download(url: str, cache_dir: str) -> str:
    if not os.path.exists(cache_dir) or not os.listdir(cache_dir):
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(cache_dir)
    return cache_dir


@dataclass
class AI2Dataset(Dataset):
    tokens: List[List[str]]
    input_ids: List[List[int]]
    token_type_ids: List[List[int]]
    attention_mask: List[List[int]]
    y: List[List[int]] = None

    def __len__(self):
        return len(self.tokens)

    @classmethod
    def load(cls, cache_dir: str, mapping: Dict, transform: Callable) -> AI2Dataset:

        content = defaultdict(list)

        for root, _, files in os.walk(cache_dir):
            for file in files:
                _, filename = os.path.split(file)

                if mapping.get(filename, '') == 'input':
                    content['input'].extend(
                        json.loads(line.strip('\r\n ').replace('\r', ' ')) for line in open(os.path.join(root, file))
                    )
                elif mapping.get(filename, '') == 'label':
                    content['label'].extend(
                        map(lambda line: int(line.strip('\r\n ')), open(os.path.join(root, file)).readlines())
                    )

        content = transform(content)
        return AI2Dataset(**content)

    def __getitem__(self, index) -> Dict:

        return {
            'tokens': self.tokens[index],
            'input_ids': self.input_ids[index],
            'token_type_ids': self.token_type_ids[index],
            'attention_mask': self.attention_mask[index],
            'y': self.y[index],
        }


@dataclass
class Preprocessor:
    tokenizer: BaseTokenizer
    x_transform: Callable
    y_transform: Callable

    def __call__(self, content: Dict) -> Dict:
        """
        Transform the content {'input': [...], 'label': [...]} into a full dataset.

        :param content: Dictionary containing {'input': [...], 'label': [...]}

        :return: Dictionary containing {'tokens': [...], 'input_ids': [...],
                'token_type_ids': [...], 'attention_mask': [...], 'y': [...]}
        """
        assert 'input' in content, "Input must be specified in content"

        result = {
            'tokens': [],
            'input_ids': [],
            'token_type_ids': [],
            'attention_mask': [],
            'y': []
        }

        if 'label' not in content:
            content['label'] = [None for _ in range(len(content['input']))]

        for x, y in tqdm(zip_longest(content['input'], content['label']), total=len(content['input'])):
            premise, hypotheses = self.x_transform(x)
            y = self.y_transform(y) if y is not None else None

            example_tokens = []
            example_input_ids = []
            example_token_type_ids = []
            example_attention_mask = []
            example_y = y

            for hypothesis in hypotheses:

                pair_tokens = [self.tokenizer.cls_token]
                pair_input_ids = [self.tokenizer.cls_index]
                pair_token_type_ids = [0]
                pair_attention_mask = [1]

                for token in self.tokenizer.tokenize(premise):
                    pair_tokens.append(token)
                    pair_input_ids.append(self.tokenizer.token2id(token))
                    pair_token_type_ids.append(0)
                    pair_attention_mask.append(1)

                pair_tokens.append(self.tokenizer.sep_token)
                pair_input_ids.append(self.tokenizer.sep_index)
                pair_token_type_ids.append(0)
                pair_attention_mask.append(1)

                for token in self.tokenizer.tokenize(hypothesis):
                    pair_tokens.append(token)
                    pair_input_ids.append(self.tokenizer.token2id(token))
                    pair_token_type_ids.append(1)
                    pair_attention_mask.append(1)

                pair_tokens.append(self.tokenizer.sep_token)
                pair_input_ids.append(self.tokenizer.sep_index)
                pair_token_type_ids.append(1)
                pair_attention_mask.append(1)

                example_tokens.append(pair_tokens)
                example_input_ids.append(np.asarray(pair_input_ids))
                example_token_type_ids.append(np.asarray(pair_token_type_ids))
                example_attention_mask.append(np.asarray(pair_attention_mask))

            result['tokens'].append(example_tokens)
            result['input_ids'].append(example_input_ids)
            result['token_type_ids'].append(example_token_type_ids)
            result['attention_mask'].append(example_attention_mask)
            result['y'].append(example_y)

        return result

    def collate(self, examples: List[Dict]) -> Dict:

        result = {
            'tokens': [],
            'input_ids': [],
            'token_type_ids': [],
            'attention_mask': [],
            'y': []
        }

        for example in examples:

            result['tokens'].append(example['tokens'])
            result['input_ids'].append(
                pad_sequence(list(map(torch.from_numpy, example['input_ids'])), batch_first=True,
                             padding_value=self.tokenizer.pad_index))
            result['token_type_ids'].append(
                pad_sequence(list(map(torch.from_numpy, example['token_type_ids'])), batch_first=True,
                             padding_value=self.tokenizer.pad_index))
            result['attention_mask'].append(
                pad_sequence(list(map(torch.from_numpy, example['attention_mask'])), batch_first=True,
                             padding_value=self.tokenizer.pad_index))
            if example['y'] is not None:
                result['y'].append(example['y'])

        result['input_ids'] = pad_sequence([x.transpose(0, 1) for x in result['input_ids']], batch_first=True,
                                           padding_value=self.tokenizer.pad_index).transpose(1, 2)
        result['token_type_ids'] = pad_sequence([x.transpose(0, 1) for x in result['token_type_ids']], batch_first=True,
                                                padding_value=self.tokenizer.pad_index).transpose(1, 2)
        result['attention_mask'] = pad_sequence([x.transpose(0, 1) for x in result['attention_mask']], batch_first=True,
                                                padding_value=self.tokenizer.pad_index).transpose(1, 2)
        if result['y']:
            result['y'] = torch.from_numpy(np.asarray(result['y']))
        else:
            result['y'] = None

        return result


@dataclass
class Formulator:
    formula: Dict

    def __call__(self, j: Dict) -> Tuple[str, List[str]]:
        assert 'premise' in self.formula and 'hypotheses' in self.formula, \
            'Invalid formula, premise and hypotheses must be specified'
        premise = ' '.join(j[key] for key in self.formula['premise'])
        hypotheses = j[self.formula['hypotheses'][0]] if len(self.formula['hypotheses']) == 1 else \
            [j[key] for key in self.formula['hypotheses']]
        return premise, hypotheses


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Download script for leaderboard datasets')

    parser.add_argument('--tasks', nargs='+',
                        choices=['anli', 'hellaswag', 'physicaliqa', 'socialiqa', 'vcr'])
    parser.add_argument('--all', '-a', action='store_true', default=False)
    parser.add_argument('--cache_dir', type=str, default='cache', required=True)

    args = parser.parse_args()

    tasks = {
        'anli': ['https://storage.googleapis.com/ai2-mosaic/public/alphanli/alphanli-train-dev.zip'],
        'hellaswag': ['https://storage.googleapis.com/ai2-mosaic/public/hellaswag/hellaswag-train-dev.zip'],
        'physicaliqa': ['https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip'],
        'socialiqa': ['https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip'],
        'vcr': [
            'https://storage.googleapis.com/ai2-mosaic/public/vcr/train.tar.gz',
            'https://storage.googleapis.com/ai2-mosaic/public/vcr/val.tar.gz'
        ]
    }

    if args.all:
        for task, urls in tqdm(tasks.items()):
            for i, url in enumerate(urls):
                download(url, cache_dir=os.path.join(args.cache_dir, task, str(i)))
    elif args.tasks is not None:
        for task in tqdm(args.tasks):
            for i, url in enumerate(tasks[task]):
                download(url, cache_dir=os.path.join(args.cache_dir, task, str(i)))
