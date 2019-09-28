from __future__ import annotations

from dataclasses import dataclass
from typing import *

import torch
from pytorch_lightning.root_module.root_module import LightningModule
from torch.utils.data import Dataset

from ai2.dataset import AI2Dataset


@dataclass
class JointDataset(Dataset):
    x: Union[List[List[float]], torch.Tensor]
    y: Optional[Union[List[int], torch.Tensor]] = None

    @classmethod
    def load(cls, dataset: AI2Dataset, models: List[LightningModule], merge: str = 'concat') -> JointDataset:
        x = []
        y = []
        for example in dataset:
            example_x = []
            for model in models:
                example_x.append(model.intermediate(**example))
            if merge == 'concat':
                example_x = torch.cat(example_x, dim=-1)
            x.append(example_x)
            if example['y'] is not None:
                y.append(example['y'])
        return JointDataset(x, None if not y else y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {
            'x': self.x[index],
            'y': self.y[index]
        }


# TODO: Pickle joint datasets;
# TODO: Train/Eval on joint dataset;

if __name__ == '__main__':
    pass
