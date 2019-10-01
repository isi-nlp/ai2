import unittest
from typing import *
from pathlib import Path

import yaml
from loguru import logger

from textbook.dataset import download, ClassificationDataset


class TestDataset(unittest.TestCase):

    path: Union[str, Path]

    def __post_init__(self):
        self.path = Path().parent
        with open(self.path / "config" / "tasks.yaml", 'r') as input_file:
            self.config = yaml.safe_load(input_file)
        self.cache_dirs = []

    def test_download(self):

        if getattr(self, 'config', None) is None:
            self.__post_init__()

        for task in ['snli', 'alphanli', 'hellaswag', 'physicaliqa', 'socialiqa', 'vcrqa', 'vcrqr']:
            cache_dirs = download(self.config[task]['urls'], self.path / 'cache')
            self.assertTrue(isinstance(cache_dirs, str) or isinstance(cache_dirs, list))
            self.cache_dirs.append((task, cache_dirs))

    def test_preprocess(self):

        if getattr(self, 'config', None) is None:
            self.test_download()

        # TODO Change the default tokenzier to your own tokenzier
        from huggingface import HuggingFaceTokenizerLoader
        tokenizer = HuggingFaceTokenizerLoader.load('roberta', 'roberta-base', do_lower_case=False)

        for task, cache_dirs in self.cache_dirs:
            dataset_name = "dev"
            dataset = ClassificationDataset.load(cache_dir=cache_dirs[-1] if isinstance(cache_dirs, list) else cache_dirs,
                                                 file_mapping=self.config[task]['file_mapping'][dataset_name],
                                                 task_formula=self.config[task]['task_formula'],
                                                 type_formula=self.config[task]['type_formula'],
                                                 preprocessor=tokenizer,
                                                 pretokenized=self.config[task].get('pretokenized', False),
                                                 label_formula=self.config[task].get('label_formula', None),
                                                 label_offset=self.config[task].get('label_offset', 0),
                                                 label_transform=self.config[task].get('label_transform', None),)
            logger.debug(' '.join(dataset[0]['tokens'][0]))
            logger.debug(dataset[0]['attention_mask'][0])
            logger.debug(dataset[0]['token_type_ids'][0])
            logger.debug(dataset[0]['input_ids'][0])
            self.assertTrue(dataset is not None)


if __name__ == "__main__":
    unittest.main()
