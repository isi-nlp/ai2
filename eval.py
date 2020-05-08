from pathlib import Path
from typing import List, Union

import hydra
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Classifier

# Save root path as hydra will create copies of this code in a folder
ROOT_PATH = Path(__file__).parent.absolute()


# If script is executed by itself, load in the configuration yaml file and desired checkpoint model
@hydra.main(config_path="config/eval.yaml")
def main(config):
    logger.info(config)

    # If the evaluation is deterministic for debugging purposes, we set the random seed
    if not isinstance(config['random_seed'], bool):
        logger.info(f"Running deterministic model with seed {config['random_seed']}")
        np.random.seed(config['random_seed'])
        torch.manual_seed(config['random_seed'])
        if torch.cuda.is_available():
            torch.backends.cuda.deterministic = True
            torch.backends.cuda.benchmark = False

    # Load in the check pointed model
    model = Classifier(config)
    device = 'cpu' if not torch.cuda.is_available() else "cuda"
    checkpoint = torch.load(ROOT_PATH / config['checkpoint_path'], map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    save_path = Path(f"{config['model']}-{config['task_name']}-s{config['random_seed']}")
    save_path.mkdir(parents=True, exist_ok=True)

    # Call the main function with appropriate parameters
    evaluate(a_classifier=model,
             output_path=save_path,
             compute_device=device,
             val_x=ROOT_PATH / config['val_x'],
             val_y=(ROOT_PATH / config['val_y'] if config['with_true_label'] else None))


# Function to perform the evaluation (This was separated out to be called in train script)
def evaluate(a_classifier: Classifier, output_path: Union[str, Path], compute_device: str,
             val_x: Union[str, Path], val_y: Union[str, Path] = None):
    # Move model to device and set to evaluation mode
    a_classifier.to(compute_device)
    a_classifier.eval()

    # Forward propagate the model to get a list of predictions and their respective confidence
    predictions: List[int] = []
    confidence: List[List[float]] = []
    for batch in tqdm(DataLoader(a_classifier.dataloader(val_x, val_y),
                                 batch_size=a_classifier.hparams["batch_size"] * 2,
                                 collate_fn=a_classifier.collate, shuffle=False)):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(compute_device)
        with torch.no_grad():
            logits = a_classifier.forward(batch)
        predictions.extend(torch.argmax(logits, dim=1).cpu().detach().numpy().tolist())
        confidence.extend(F.softmax(logits, dim=-1).cpu().detach().numpy().tolist())

    # Offset the predictions with the lowest label
    predictions = [p + a_classifier.label_offset for p in predictions]

    # Write out the result lists
    with open(f"{output_path}/predictions.lst", "w+") as f:
        f.write("\n".join(map(str, predictions)))
    with open(f"{output_path}/confidence.lst", "w+") as f:
        f.write("\n".join(map(lambda l: '\t'.join(map(str, l)), confidence)))

    # If desired y value is provided, calculate relevant statistics
    if val_y:
        labels = pd.read_csv(val_y, sep='\t', header=None).values.tolist()
        logger.info(f"F1 score: {accuracy_score(labels, predictions):.3f}")

        stats = []
        for _ in range(100):
            indices = [i for i in np.random.random_integers(0, len(predictions) - 1, size=len(predictions))]
            stats.append(accuracy_score([labels[j] for j in indices], [predictions[j] for j in indices]))

        # Calculate the confidence interval and log it to console
        alpha = 0.95
        p = ((1.0 - alpha) / 2.0) * 100
        lower = max(0.0, np.percentile(stats, p))
        p = (alpha + ((1.0 - alpha) / 2.0)) * 100
        upper = min(1.0, np.percentile(stats, p))
        logger.info(f'{alpha * 100:.1f} confidence interval {lower * 100:.1f} and {upper * 100:.1f}, '
                    f'average: {np.mean(stats) * 100:.1f}')


if __name__ == "__main__":
    main()
