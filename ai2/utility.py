# encoding: utf-8
# Description: Utility functions for data preprocessing and loading.
# Author: Chenghao Mou
# Last updated: Aug 30, 2019

import numpy as np
import torch
import os
import requests
import yaml
import json
from dataclasses import dataclass
from typing import *
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
from loguru import logger
from pytorch_transformers.tokenization_utils import PreTrainedTokenizer


@dataclass
class Pair:

    premise: str
    hypothesis: str


@dataclass
class Example:

    pairs: List[Pair]
    label: int   # Zero-indexed true label


@dataclass
class AI2DatasetHelper:

    config: Dict
    filename: str = None
    cache_dir: str = './.cache'

    examples: List = None

    def __post_init__(self):

        assert 'url' in self.config, 'Download url is missing'

        self.filename = self.config['url'].rsplit('/', 1)[-1]

        self.cache_dir = Path(self.cache_dir)

    def download(self) -> Tuple[List[Dict], List[str], List[Dict], List[str]]:
        """Download both train and dev datasets."""
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        if not os.path.exists(self.cache_dir / self.filename):
            request = requests.get(self.config['url'])
            with open(self.cache_dir / self.filename, 'wb') as output:
                output.write(request.content)

        with open(self.cache_dir / self.filename, "rb") as input_file:
            zipped_file = ZipFile(BytesIO(input_file.read()))

        # Check expected fields with actual ones.
        expected = set(self.config['format'].values())
        found = set(f for f in zipped_file.namelist() if f.endswith(('jsonl', 'lst')) and not f.startswith(('__', '.')))

        assert expected == found, f'Dismatched files in downloaded file, looking for {exp} in {found}'

        train_x = list(map(json.loads, zipped_file.open(self.config['format'].get('train_x', None)).readlines()))
        train_y = zipped_file.open(self.config['format'].get('train_y', None)).readlines()
        dev_x = list(map(json.loads, zipped_file.open(self.config['format'].get('dev_x', None)).readlines()))
        dev_y = zipped_file.open(self.config['format'].get('dev_y', None)).readlines()

        logger.info(f"""Training examples: {len(train_x)}, Dev examples: {len(dev_x)}""")

        return train_x, train_y, dev_x, dev_y

    def preprocess(self, input_x: List[Dict], input_y: List[str]) -> List[Example]:

        examples = []
        premise_keys, choice_keys = self.parse_formula()

        for raw_example, raw_label in zip(input_x, input_y):

            choices = raw_example[choice_keys[0]] if len(choice_keys) == 1 else [raw_example[choice_key] for choice_key in choice_keys]
            premise = ' '.join(raw_example[premise_key] for premise_key in premise_keys)

            examples.append(
                Example(
                    [
                        Pair(premise, choice) for choice in choices
                    ],
                    int(raw_label) - self.config['start']
                )
            )

        return examples

    def parse_formula(self):

        assert 'formula' in self.config, 'Formula is missing'

        def strip(l: List[str]) -> List[str]:
            return [x.strip() for x in l]

        premise_keys, choice_keys = self.config['formula'].split('=>')
        premise_keys = strip(premise_keys.split('+'))
        choice_keys = strip(choice_keys.split('|'))

        return premise_keys, choice_keys


@dataclass
class AI2Dataset(Dataset):

    tokenizer: PreTrainedTokenizer
    examples: List[Example]
    padding_index: int = None
    max_sequence_length: int = 128

    def __post_init__(self):
        self.padding_index = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        logger.info(f'Padding index: {self.padding_index}')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        example = self.examples[index]

        input_ids = []
        input_token_type_ids = []
        input_mask = []
        input_tokens = []

        for pair in example.pairs:

            # [CLS] TOK TOK ... [SEP] TOK TOK ... [SEP] [PAD] ...
            # 0     0   0   ... 0     1   1   ... 1     PAD   ...
            # 1     1   1   ... 1     1   1   ... 1     0     0

            tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(pair.premise) + [self.tokenizer.sep_token]

            token_type_ids = [0] * len(tokens)

            tokens += self.tokenizer.tokenize(pair.hypothesis) + [self.tokenizer.sep_token]

            token_type_ids += [1] * (len(tokens) - len(token_type_ids))

            tokens = tokens[:self.max_sequence_length]
            token_type_ids = token_type_ids[:self.max_sequence_length]

            input_mask.append([1] * len(tokens))
            input_token_type_ids.append(np.asarray(token_type_ids))
            input_tokens.append(tokens)
            input_ids.append(np.asarray(self.tokenizer.convert_tokens_to_ids(tokens)))

        input_tensor = pad_list(input_ids, self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0])
        input_token_type_ids = pad_list(input_token_type_ids, self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0])
        input_mask = pad_list(input_mask, self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0])

        #if index == 0:
        #    print(input_token_type_ids)
        #    logger.debug(f"Example: {input_tokens[0]}")
        #    logger.debug(f"X: {input_tensor[0].numpy().tolist()}")
        #    logger.debug(f"token_type_ids: {input_token_type_ids[0].numpy().tolist()}")
        #    logger.debug(f"attention_mask: {input_mask[0].numpy().tolist()}")

        return {
            'x': input_tensor.long(),
            'token_type_ids': input_token_type_ids,
            'attention_mask': input_mask,
            'y': torch.tensor(example.label).reshape((1,)).long()
        }


def pad_list(l: Union[List[np.ndarray], List[torch.Tensor]], padding_index: int) -> torch.LongTensor:

    l = list([np.asarray(x) if isinstance(x, list) else x for x in l])
    assert all(x.shape[:-1] == l[0].shape[:-1] for x in l), 'Mismatch dimensions for list elements!'
    *dimensions, _ = l[0].shape
    placeholder = torch.zeros((len(l), *dimensions, max(map(lambda x: x.shape[-1], l))), requires_grad=False).long() + padding_index

    for i, x in enumerate(l):
        placeholder[i, ..., :x.shape[-1]] = torch.tensor(x).long() if isinstance(x, np.ndarray) else x

    return placeholder


def load_config(path: Path, name: str = None) -> Dict:

    with open(path) as f:
        if name:
            return yaml.load(f, Loader=yaml.FullLoader)[name]
        else:
            return yaml.load(f, Loader=yaml.FullLoader)


def collate_fn(exmples, padding_index: int):

    return {
        'x': pad_list([x['x'] for x in exmples], padding_index),
        'token_type_ids': pad_list([x['token_type_ids'] for x in exmples], padding_index),
        'attention_mask': pad_list([x['attention_mask'] for x in exmples], padding_index),
        'y': pad_list([x['y'] for x in exmples], padding_index).reshape(-1)
    }


if __name__ == "__main__":

    print(pad_list([[1, 2], [2, 3, 4]], 0))
    print(pad_list([[[1], [2]], [[2, 1], [3, 4]]], 0))

    config = load_config('/Users/chenghaomou/Code/Code-ProjectsPyCharm/ai2/ai2/tasks.yaml')
    helper = AI2DatasetHelper(config)
    train_x, train_y, dev_x, dev_y = helper.download()
    train_examples = helper.preprocess(train_x, train_y)

    # print(train_examples[0])

    from pytorch_transformers import *

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', cache_dir='./.cache', do_lower_case=False)
    dataset = AI2Dataset(tokenizer, train_examples)

    # print(dataset[0])

    batch_x, batch_y = collate_fn([dataset[0], dataset[1], dataset[4]], tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0])
    print(batch_x.shape, batch_y.shape)
