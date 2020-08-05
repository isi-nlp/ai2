"""
Script used to generate the bundle embeddings and its meta data. The result of this script is used in closest.py.

The script is configured by the embed.yaml and the paths to checkpoints are stored in the config/checkpoint_list/
folder.
"""

import pickle
from pathlib import Path

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Classifier

# Save root path as hydra will create copies of this code in date specific folder
ROOT_PATH = Path(__file__).parent.absolute()


@hydra.main(config_path="config/embed.yaml")
def embedding(config):
    logger.info(config)
    bool_pb = config['with_progress_bar']

    # Initiate the Distance Evaluation Dictionary to store all information needed to evaluate
    distance_eval_dict = {'task_name': config['task_name'], 'task_formula': config['formula'],
                          'train_path': config['train_x'], 'dev_path': config['val_x'],
                          'embeddings': []}
    device = 'cpu' if not torch.cuda.is_available() else "cuda"

    # If an out of box model requested, we first embed the files using an out of box model
    if config['with_out_of_box_model']:
        logger.info(f"Parsing embeddings using out of box model: {config['model']}")

        # Loading in an out of box model
        config = OmegaConf.to_container(config)
        model = Classifier(config)
        model.to(device)
        model.eval()
        distance_eval_dict['embeddings'].append(
            ('out_of_box',
             calculate_embeddings(a_classifier=model, compute_device=device, feature=config['feature'],
                                  text_path=ROOT_PATH / config['train_x'],
                                  label_path=ROOT_PATH / config['train_y'], with_progress_bar=bool_pb),
             calculate_embeddings(a_classifier=model, compute_device=device, feature=config['feature'],
                                  text_path=ROOT_PATH / config['val_x'],
                                  label_path=ROOT_PATH / config['val_y'], with_progress_bar=bool_pb))
        )

    # If a list of checkpoints is provided, we use each of them to parse the embeddings for the tasks
    if config['checkpoint_list']:
        with open(ROOT_PATH / config['checkpoint_list'], 'r') as checkpoint_list:
            for a_checkpoint_file_location in checkpoint_list:

                # Extract the model name and calculate embeddings
                model_name = a_checkpoint_file_location.strip().split('/')[-1].split('.')[0]
                logger.info(f'Parsing embeddings using {model_name}')

                # Load in the saved model, as well as weights saved during training time
                model = Classifier.load_from_checkpoint(str(ROOT_PATH / a_checkpoint_file_location.strip()))
                model.to('cpu' if not torch.cuda.is_available() else "cuda")
                model.eval()

                distance_eval_dict['embeddings'].append(
                    (model_name,
                     calculate_embeddings(a_classifier=model, compute_device=device, feature=config['feature'],
                                          text_path=ROOT_PATH / config['train_x'],
                                          label_path=ROOT_PATH / config['train_y'], with_progress_bar=bool_pb),
                     calculate_embeddings(a_classifier=model, compute_device=device, feature=config['feature'],
                                          text_path=ROOT_PATH / config['val_x'],
                                          label_path=ROOT_PATH / config['val_y'], with_progress_bar=bool_pb))
                )

    # Pickle dump the dictionary for embedding distance calculation
    with open(f"{config['model']}-{config['task_name']}-{config['feature']}.embed", 'wb') as output_file:
        pickle.dump(distance_eval_dict, output_file)


# Helper function for parsing embeddings and return a 2D numpy array
def calculate_embeddings(a_classifier: Classifier, text_path: str, label_path: str,
                         compute_device: str, feature: str, with_progress_bar: bool):
    # Initialize the embedding list
    embedding_list = []

    # Forward propagate the model to get a list of predictions and their respective confidence
    for batch in tqdm(DataLoader(a_classifier.dataloader(text_path, label_path),
                                 batch_size=a_classifier.hparams["batch_size"],
                                 collate_fn=a_classifier.collate, shuffle=False), disable=not with_progress_bar):

        # Move component to computing device if possible
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(compute_device)

        batch["token_type_ids"] = None if "roberta" in a_classifier.hparams["model"] else batch["token_type_ids"]

        with torch.no_grad():
            # Embed the given batch
            results = a_classifier.embedder(input_ids=batch["input_ids"],
                                            attention_mask=batch["attention_mask"],
                                            token_type_ids=batch["token_type_ids"])
            token_embeddings, *_ = results

            if feature == 'AVG_MEAN':
                per_story_avg_embed = \
                    token_embeddings.mean(dim=1).reshape(len(batch['labels']), batch['num_choice'], -1).mean(dim=1)
            else:
                # TODO: Implement CLS_*, AVG_CORRECT, AVG_NULL
                raise NotImplementedError(f"Feature for embedding calculation {feature} is not yet implemented")

        # Extend the embedding list so far with newly embeded batch
        embedding_list.extend(per_story_avg_embed.cpu().detach().numpy())

    return np.stack(embedding_list)


if __name__ == "__main__":
    embedding()
