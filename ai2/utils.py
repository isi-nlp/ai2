from dataclasses import dataclass
from typing import *
from io import BytesIO
from zipfile import ZipFile
import requests
import json
from loguru import logger
from collections import namedtuple
from pprint import pprint
from pytorch_transformers.tokenization_utils import PreTrainedTokenizer
from torch.utils.data import Dataset

TASKS = {
    'anli': {
        'url': 'https://storage.googleapis.com/ai2-mosaic/public/alphanli/alphanli-train-dev.zip',
        'type': 'classification',
        'classes': 1,  # Boolean value for each of the two hypotheses
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
        'classes': 1,  # Boolean value for each of the four choices
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
        'classes': 1,  # Boolean value for each of the two solutions
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
        'classes': 1,  # Boolean value for each of the three answers
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

Example = namedtuple('Example', ['left', 'right', 'label'])


@dataclass
class AI2Preprocessor:

    config: Dict  # One of the TASKS

    def download(self):

        request = requests.get(self.config['url'])
        zipped_file = ZipFile(BytesIO(request.content))

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

        left, right = self.parse_formula(self.config['formula'])
        examples = []

        for input_json, label in zip(input_x, input_y):
            question = ' '.join(input_json[x] for x in left)
            choices = input_json[right[0]] if len(right) == 1 else [input_json[x] for x in right]

            for i, choice in enumerate(choices):
                if i == int(label) - self.config['start']:
                    examples.append(Example(question, choice, 1))
                else:
                    examples.append(Example(question, choice, 0))

        return examples

    @staticmethod
    def parse_formula(formula):
        left, right = list(map(lambda x: x.strip(), formula.split('=>', 1)))
        return list(map(lambda x: x.strip(), left.split('+'))), list(map(lambda x: x.strip(), right.split('|')))


@dataclass
class AI2Dataset(Dataset):

    x: List
    tokenizer: PreTrainedTokenizer

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        left, right, label = self.x[idx].left, self.x[idx].right, self.x[idx].label
        sample = {
            'x': self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(left) + ['[SEP]'] + self.tokenizer.tokenize(right) + ['[SEP]']),
            'y': label
        }
        # print(sample)
        return sample


if __name__ == "__main__":
    dataset = AI2Dataset(TASKS['socialiqa'])
    train_x, train_y, dev_x, dev_y = dataset.download()

    training_examples = dataset.preprocess(train_x, train_y)
    pprint(training_examples[:5])
