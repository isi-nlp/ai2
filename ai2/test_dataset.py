
from ai2.interface import HuggingFaceTokenizerLoader
from ai2.dataset import download, AI2Dataset
from loguru import logger
import yaml
import unittest


class TestDataset(unittest.TestCase):

    def __post_init__(self):
        with open("./tasks.yaml", 'r') as input_file:
            self.config = yaml.safe_load(input_file)
        self.cache_dirs = []

    def test_download(self):

        if getattr(self, 'config', None) is None:
            self.__post_init__()

        for task in ['snli', 'alphanli', 'hellaswag', 'physicaliqa', 'socialiqa', 'vcrqa', 'vcrqr']:
            cache_dirs = download(self.config[task]['urls'], './cache')
            self.assertTrue(isinstance(cache_dirs, str) or isinstance(cache_dirs, list))
            self.cache_dirs.append((task, cache_dirs))
        # print(self.cache_dirs)

    def test_preprocess(self):

        if getattr(self, 'config', None) is None:
            self.test_download()

        tokenizer = HuggingFaceTokenizerLoader.load('bert', 'bert-base-cased', do_lower_case=False)

        for task, cache_dirs in self.cache_dirs:
            dataset_name = "dev"
            dataset = AI2Dataset.load(cache_dir=cache_dirs[-1] if isinstance(cache_dirs, list) else cache_dirs,
                                      file_mapping=self.config[task]['file_mapping'][dataset_name],
                                      task_formula=self.config[task]['task_formula'],
                                      type_formula=self.config[task]['type_formula'],
                                      preprocessor=tokenizer,
                                      pretokenized=self.config[task].get('pretokenized', False),
                                      label_formula=self.config[task].get('label_formula', None),
                                      label_offset=self.config[task].get('label_offset', 0),
                                      label_transform=self.config[task].get('label_transform', None),)
            # logger.info(dataset[0]['tokens'])
            logger.debug(' '.join(dataset[0]['tokens'][0]))
            self.assertTrue(dataset is not None)


if __name__ == "__main__":
    unittest.main()
