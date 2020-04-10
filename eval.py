import pathlib
from typing import *

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from loguru import logger
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Classifier

# Save root path as hydra will create copies of this code in a folder
ROOT_PATH = pathlib.Path(__file__).parent.absolute()


@hydra.main(config_path="config/eval.yaml")
def evaluate(config):
    logger.info(config)

    # If the evaluation is deterministic for debugging purposes, we set the random seed
    if not isinstance(config['random_seed'], bool):
        logger.info(f"Running deterministic model with seed {config['random_seed']}")
        np.random.seed(config['random_seed'])
        torch.manual_seed(config['random_seed'])
        if torch.cuda.is_available():
            torch.backends.cuda.deterministic = True
            torch.backends.cuda.benchmark = False

    device = 'cpu' if not torch.cuda.is_available() else "cuda"

    # Load in the check pointed model
    checkpoint = torch.load(ROOT_PATH / config['checkpoint_path'], map_location=device)
    model = Classifier(config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    predictions: List[int] = []
    confidence: List[List[float]] = []
    for batch in tqdm(DataLoader(model.dataloader(
            ROOT_PATH / config['val_x'],
            (ROOT_PATH / config['val_y'] if config['with_eval'] else None)),
            batch_size=model.hparams["batch_size"] * 2,
            collate_fn=model.collate,
            shuffle=False)):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        with torch.no_grad():
            logits = model.forward(batch)
        predictions.extend(torch.argmax(logits, dim=1).cpu().detach().numpy().tolist())
        confidence.extend(F.softmax(logits, dim=-1).cpu().detach().numpy().tolist())
    predictions = [p + model.label_offset for p in predictions]

    with open(f"{config['model']}_{config['task_name']}_predictions.lst", "w+") as f:
        f.write("\n".join(map(str, predictions)))
    with open(f"{config['model']}_{config['task_name']}confidence.lst", "w+") as f:
        f.write("\n".join(map(lambda l: '\t'.join(map(str, l)), confidence)))

    if config['with_eval']:
        labels = pd.read_csv(ROOT_PATH / config['val_y'], sep='\t', header=None).values.tolist()
        logger.info(f"F1 score: {accuracy_score(labels, predictions):.3f}")

        stats = []
        for _ in range(100):
            indices = [i for i in np.random.random_integers(0, len(predictions)-1, size=len(predictions))]
            stats.append(accuracy_score([labels[j] for j in indices], [predictions[j] for j in indices]))

        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lower = max(0.0, np.percentile(stats, p))
        p = (alpha+((1.0-alpha)/2.0)) * 100
        upper = min(1.0, np.percentile(stats, p))
        logger.info(f'{alpha*100:.1f} confidence interval {lower*100:.1f} and {upper*100:.1f}, '
                    f'average: {np.mean(stats)*100:.1f}')


if __name__ == "__main__":
    evaluate()
