from dataclasses import dataclass
from typing import *
from io import BytesIO
from zipfile import ZipFile
import requests
import json
import torch
import numpy as np
from loguru import logger
from collections import namedtuple
from pprint import pprint
from pytorch_transformers.tokenization_utils import PreTrainedTokenizer
from torch.utils.data import Dataset
import os

TASKS = {
    'anli': {
        'url': 'https://storage.googleapis.com/ai2-mosaic/public/alphanli/alphanli-train-dev.zip',
        'type': 'classification',
        'classes': 2,  # Boolean value for each of the two hypotheses
        'contains': ['story_id', 'obs1', 'obs2', 'hyp1', 'hyp2'],
        'formula': 'obs1 + obs2 => hyp1|hyp2',
        'start': 1,
        'format': {
            'train_x': 'train.jsonl',
            'train_y': 'train-labels.lst',
            'dev_x': 'dev.jsonl',
            'dev_y': 'dev-labels.lst'
        }
    },
    'hellaswag': {
        'url': 'https://storage.googleapis.com/ai2-mosaic/public/hellaswag/hellaswag-train-dev.zip',
        'type': 'classification',
        'classes': 4,  # Boolean value for each of the four choices
        'contains': ['ind', 'id', 'activity_label', 'ctx', 'ctx_a', 'ctx_b', 'dataset', 'ending_options'],
        'formula': 'ctx => ending_options',
        'start': 0,
        'format': {
            'train_x': 'hellaswag-train-dev/train.jsonl',
            'train_y': 'hellaswag-train-dev/train-labels.lst',
            'dev_x': 'hellaswag-train-dev/valid.jsonl',
            'dev_y': 'hellaswag-train-dev/valid-labels.lst'
        }
    },
    'physicaliqa': {
        'url': 'https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip',
        'type': 'classification',
        'classes': 2,  # Boolean value for each of the two solutions
        'contains': ['goal', 'sol1', 'sol2'],
        'formula': 'goal => sol1|sol2',
        'start': 0,
        'format': {
            'train_x': 'physicaliqa-train-dev/train.jsonl',
            'train_y': 'physicaliqa-train-dev/train-labels.lst',
            'dev_x': 'physicaliqa-train-dev/dev.jsonl',
            'dev_y': 'physicaliqa-train-dev/dev-labels.lst'
        }
    },
    'socialiqa': {
        'url': 'https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip',
        'type': 'classification',
        'classes': 3,  # Boolean value for each of the three answers
        'contains': ['context', 'question', 'answerA', 'answerB', 'answerC'],
        'formula': 'context + question => answerA|answerB|answerC',
        'start': 1,
        'format': {
            'train_x': 'socialiqa-train-dev/train.jsonl',
            'train_y': 'socialiqa-train-dev/train-labels.lst',
            'dev_x': 'socialiqa-train-dev/dev.jsonl',
            'dev_y': 'socialiqa-train-dev/dev-labels.lst'
        }
    }
}

Sample = namedtuple('Sample', ['premise', 'choice'])
Example = namedtuple('Example', ['samples', 'label'])


@dataclass
class AI2Preprocessor:

    config: Dict  # One of the TASKS

    def download(self):

        # Caching datasets
        if not os.path.exists('./.cache/'):
            os.mkdir("./.cache/")

        filename = self.config['url'].rsplit('/', 1)[-1]

        if not os.path.exists(f"./.cache/{filename}"):
            request = requests.get(self.config['url'])
            with open(f"./.cache/{filename}", "wb") as output:
                output.write(request.content)

        with open(f"./.cache/{filename}", "rb") as input_file:
            zipped_file = ZipFile(BytesIO(input_file.read()))

        # Load examples and labels
        exp = set(self.config['format'].values())
        found = set(f for f in zipped_file.namelist() if (f.endswith('jsonl') or f.endswith('lst')) and (not f.startswith('__')))
        assert exp == found, f'Dismatched files in downloaded file, looking for {exp} in {found}'
        train_x = list(map(json.loads, zipped_file.open(self.config['format'].get('train_x', None)).readlines()))
        train_y = zipped_file.open(self.config['format'].get('train_y', None)).readlines()
        dev_x = list(map(json.loads, zipped_file.open(self.config['format'].get('dev_x', None)).readlines()))
        dev_y = zipped_file.open(self.config['format'].get('dev_y', None)).readlines()

        logger.info(f"""Training examples: {len(train_x)}, Dev examples: {len(dev_x)}""")

        return train_x, train_y, dev_x, dev_y

    def preprocess(self, input_x, input_y):

        premise_keys, choice_keys = self.parse_formula(self.config['formula'])

        examples = []

        for input_json, correct_choice in zip(input_x, input_y):

            premise = ' '.join(input_json[x] for x in premise_keys)
            choices = input_json[choice_keys[0]] if len(choice_keys) == 1 else [input_json[x] for x in choice_keys]

            label = int(correct_choice) - self.config['start']
            samples = []

            for choice in choices:
                sample = Sample(premise, choice)
                samples.append(sample)

            examples.append(Example(samples, label))

        return examples

    @staticmethod
    def parse_formula(formula):
        premise_keys, choice_keys = list(map(lambda x: x.strip(), formula.split('=>', 1)))
        return list(map(lambda x: x.strip(), premise_keys.split('+'))), list(map(lambda x: x.strip(), choice_keys.split('|')))


@dataclass
class AI2Dataset(Dataset):

    x: List
    tokenizer: PreTrainedTokenizer
    padding_index: int

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        samples, label = self.x[idx].samples, self.x[idx].label
        samples = list(map(self.sample2tensor, samples))
        x = torch.zeros((len(samples), max(map(len, samples)))) + self.padding_index
        for i in range(len(samples)):
            x[i, :len(samples[i])] = torch.from_numpy(np.asarray(samples[i]))

        sample = {
            'x': x,  # [C, S]
            'y': torch.tensor(label).long(),  # [C]
        }
        return sample

    def sample2tensor(self, sample):
        return self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.cls_token] +
            self.tokenizer.tokenize(sample.premise) +
            [self.tokenizer.sep_token] +
            self.tokenizer.tokenize(sample.choice) +
            [self.tokenizer.sep_token])


# if __name__ == "__main__":
#     from pytorch_transformers import *
#     from pprint import pprint
#     tokenizer = BertTokenizer.from_pretrained('bert-base-cased', cache_dir='./.cache', do_lower_case=False)
#     preprocessor = AI2Preprocessor(TASKS['anli'])
#     train_x, train_y, dev_x, dev_y = preprocessor.download()
#     examples = preprocessor.preprocess(train_x, train_y)
#     pprint(examples[0])
