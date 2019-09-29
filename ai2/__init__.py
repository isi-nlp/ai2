import random
import numpy as np
import torch
from typing import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)


def get_default(config: Dict, task_name: str, model_type: str, model_weight: str, field: str):
    task_model_config = config[task_name].get(
        model_type, {}).get(
        model_weight,
        config[task_name]['default'])

    return task_model_config.get(field, config[task_name]['default'][field])
