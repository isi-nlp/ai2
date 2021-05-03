from pathlib import Path
from typing import List, Union, Any

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from ai2.model import Classifier


def main(params: Parameters):
    checkpoint_path = params.existing_file('checkpoint_path')
    results_path = params.creatable_file('results_path')
    val_x_file = params.existing_file('val_x')
    val_y_file = params.optional_existing_file('val_y')
    with_true_label = params.boolean('with_true_label')
    if with_true_label and val_y_file is None:
        raise RuntimeError(
            f'with_true_label set to true but no true labels (val_y) provided! '
        )
    elif not with_true_label and val_y_file is not None:
        raise RuntimeError(
            f'with_true_label set to false but got true labels val_y!'
        )

    model_name = params.string('model.model_name')
    task_name = params.string('task_name')
    maybe_random_seed = params.get('random_seed', object)

    # If the evaluation is deterministic for debugging purposes, we set the random seed
    if not isinstance(maybe_random_seed, bool):
        if not isinstance(maybe_random_seed, int): \
                raise RuntimeError(
                    "Random seed must be either false (i.e. no random seed)"
                    "or an integer seed!"
                )
        logger.info(f"Running deterministic model with seed {maybe_random_seed}")
        np.random.seed(maybe_random_seed)
        torch.manual_seed(maybe_random_seed)
        if torch.cuda.is_available():
            torch.backends.cuda.deterministic = True
            torch.backends.cuda.benchmark = False

    # Load in the check pointed model
    config = params.namespace('model').as_nested_dicts()
    config.update((k, v) for k, v in params.as_nested_dicts().items() if k != 'model')
    model = Classifier(config)
    device = 'cpu' if not torch.cuda.is_available() else "cuda"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    results_path_param: str = "results_path"
    if results_path_param in params:
        save_path = params.creatable_directory(results_path_param)
    else:
        save_path = Path(f"{model_name}-{task_name}-s{maybe_random_seed}")
        save_path.mkdir(parents=True, exist_ok=True)

    # Call the main function with appropriate parameters
    evaluate(a_classifier=model,
             output_path=save_path,
             results_path=results_path,
             compute_device=device,
             val_x=val_x_file,
             val_y=val_y_file)


# Function to perform the evaluation (This was separated out to be called in train script)
def evaluate(a_classifier: Classifier, output_path: Union[str, Path], results_path: Union[str, Path],
             compute_device: str, val_x: Union[str, Path], val_y: Union[str, Path] = None):
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
        num_choice = batch["num_choice"][0].item()
        logits = logits.reshape(-1, num_choice)
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
        logger.info(f"Accuracy score: {accuracy_score(labels, predictions):.3f}")

        stats = []
        for _ in range(10000):
            indices = [i for i in np.random.randint(0, len(predictions) - 1, size=len(predictions))]
            stats.append(accuracy_score([labels[j] for j in indices], [predictions[j] for j in indices]))

        # Calculate the confidence interval and log it to console
        alpha = 0.95
        p = ((1.0 - alpha) / 2.0) * 100
        lower = max(0.0, np.percentile(stats, p))
        p = (alpha + ((1.0 - alpha) / 2.0)) * 100
        upper = min(1.0, np.percentile(stats, p))
        logger.info(f'{alpha * 100:.1f} confidence interval {lower * 100:.1f} and {upper * 100:.1f}, '
                    f'average: {np.mean(stats) * 100:.1f}')

        # Log eval result
        with open(results_path, "a+") as resultf:
            resultf.write(f'{output_path},Accuracy-lower-upper-average,{accuracy_score(labels, predictions):.3f},'
                    f'{lower * 100:.1f},{upper * 100:.1f},{np.mean(stats) * 100:.1f}\n')


if __name__ == "__main__":
    parameters_only_entry_point(main)
