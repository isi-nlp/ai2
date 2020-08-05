from pathlib import Path
from typing import List, Union

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Classifier

# Save root path as hydra will create copies of this code in a folder
ROOT_PATH = Path(__file__).parent.absolute()


# If script is executed by itself, load in the configuration yaml file and desired checkpoint model
@hydra.main(config_path="config/eval.yaml")
def main(config):
    config = OmegaConf.to_container(config)
    logger.info(config)

    # Load in the saved model, as well as configuration saved during training time
    model = Classifier.load_from_checkpoint(str(ROOT_PATH / config['checkpoint_path']))
    model_config = model.hparams

    # If the training is deterministic, we set the random seed for evaluation as well
    if isinstance(model_config['random_seed'], int):
        seed_everything(model_config['random_seed'])

    save_path = Path(f"eval-{model_config['model']}-{model_config['task_name']}-s{model_config['random_seed']}")
    save_path.mkdir(parents=True, exist_ok=True)

    # Call the main function with appropriate parameters
    evaluate(a_classifier=model, output_path=save_path,
             compute_device='cpu' if not torch.cuda.is_available() else "cuda",
             with_progress_bar=config['with_progress_bar'],
             val_x=ROOT_PATH / model_config['val_x'],
             val_y=(ROOT_PATH / model_config['val_y'] if config['with_true_label'] else None))


# Function to perform the evaluation (This was separated out to be called in train script)
def evaluate(a_classifier: Classifier, output_path: Union[str, Path], compute_device: str, with_progress_bar: bool,
             val_x: Union[str, Path], val_y: Union[str, Path] = None):

    # Move model to device and set to evaluation mode
    a_classifier.to(compute_device)
    a_classifier.eval()

    # Forward propagate the model to get a list of all logits for the task
    all_logits: List[torch.Tensor] = []
    for batch in tqdm(DataLoader(a_classifier.dataloader(val_x, val_y), batch_size=a_classifier.hparams["batch_size"],
                                 collate_fn=a_classifier.collate, shuffle=False), disable=not with_progress_bar):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(compute_device)
        with torch.no_grad():
            logits = a_classifier.forward(batch)
        logits = logits.reshape(-1, batch["num_choice"])
        all_logits.append(logits)

    # Stack the logits and
    complete_logit = torch.cat(all_logits, dim=0)
    predictions = torch.argmax(complete_logit, dim=1).cpu().detach().numpy().tolist()
    confidence = F.softmax(complete_logit, dim=-1).cpu().detach().numpy().tolist()

    # Offset the predictions with the lowest label
    predictions = [p + a_classifier.label_offset for p in predictions]

    # Write out the result lists
    with open(f"{output_path}/predictions.lst", "w+") as f:
        f.write("\n".join(map(str, predictions)))
    with open(f"{output_path}/confidence.lst", "w+") as f:
        f.write("\n".join(map(lambda l: '\t'.join(map(str, l)), confidence)))

    # If desired y value is provided, calculate the accuracy, validation loss, and relevant statistics
    if val_y:
        labels = pd.read_csv(val_y, sep='\t', header=None).values.tolist()
        val_loss = a_classifier.loss(complete_logit, [a_label - a_classifier.label_offset for a_label in labels])
        logger.info(f"Accuracy score: {accuracy_score(labels, predictions):.3f}")
        logger.info(f'Validation Loss: {val_loss.mean()}')

        # Calculate the confidence interval and log it to console
        stats = []
        for _ in range(100):
            indices = [i for i in np.random.random_integers(0, len(predictions) - 1, size=len(predictions))]
            stats.append(accuracy_score([labels[j] for j in indices], [predictions[j] for j in indices]))
        alpha = 0.95
        p = ((1.0 - alpha) / 2.0) * 100
        lower = max(0.0, np.percentile(stats, p))
        p = (alpha + ((1.0 - alpha) / 2.0)) * 100
        upper = min(1.0, np.percentile(stats, p))
        logger.info(f'{alpha * 100:.1f} confidence interval {lower * 100:.1f} and {upper * 100:.1f}, '
                    f'average: {np.mean(stats) * 100:.1f}')


if __name__ == "__main__":
    main()
