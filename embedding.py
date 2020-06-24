import pathlib
import pickle

import hydra
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Classifier

# Save root path as hydra will create copies of this code in date specific folder
ROOT_PATH = pathlib.Path(__file__).parent.absolute()


@hydra.main(config_path="config/embed.yaml")
def embedding(config):
    logger.info(config)
    device = 'cpu' if not torch.cuda.is_available() else "cuda"

    # Initiate the Distance Evaluation Dictionary to store all information needed to evaluate
    distance_eval_dict = {'task_name': config['task_name'], 'task_formula': config['formula'],
                          'train_path': ROOT_PATH / config['train_x'], 'dev_path': ROOT_PATH / config['val_x'],
                          'checkpoint_names': [], 'train_embed': [], 'dev_embed': []}

    # First we use an out of box model to encode the train and dev files as baseline
    model = Classifier(config)
    model.to(device)
    model.eval()
    logger.info('Parsing embeddings using an out of box model')
    distance_eval_dict['checkpoint_names'].append('out_of_box')
    distance_eval_dict['train_embed'].append(
        calculate_embeddings(a_classifier=model, compute_device=device, feature=config['feature'],
                             text_path=ROOT_PATH / config['train_x'], label_path=ROOT_PATH / config['train_y']))
    distance_eval_dict['dev_embed'].append(
        calculate_embeddings(a_classifier=model, compute_device=device, feature=config['feature'],
                             text_path=ROOT_PATH / config['val_x'], label_path=ROOT_PATH / config['val_y']))

    if config['checkpoint_list']:
        with open(ROOT_PATH / config['checkpoint_list'], 'r') as checkpoint_list:
            for a_checkpoint_file_location in checkpoint_list:
                # Load in the checkpoint file for the mdoel
                model = Classifier(config)
                checkpoint = torch.load(ROOT_PATH / a_checkpoint_file_location.strip(), map_location=device)
                model.load_state_dict(checkpoint['state_dict'])
                model.to(device)
                model.eval()

                # Extract the model name and calculate embeddings
                model_name = a_checkpoint_file_location.strip().split('/')[-1].split('.')[0]
                logger.info(f'Parsing embeddings using {model_name}')

                distance_eval_dict['checkpoint_names'].append(model_name)
                distance_eval_dict['dev_embed'].append(
                    calculate_embeddings(a_classifier=model, compute_device=device, feature=config['feature'],
                                         text_path=ROOT_PATH/config['val_x'], label_path=ROOT_PATH/config['val_y']))
                distance_eval_dict['train_embed'].append(
                    calculate_embeddings(a_classifier=model, compute_device=device, feature=config['feature'],
                                         text_path=ROOT_PATH/config['train_x'], label_path=ROOT_PATH/config['train_y']))

    # Pickle dump the dictionary for embedding distance calculation
    with open(f"{config['model']}-{config['task_name']}-{config['feature']}.pickle", 'wb') as output_file:
        pickle.dump(distance_eval_dict, output_file)


# Helper function for parsing embeddings and return a 2D numpy array
def calculate_embeddings(a_classifier: Classifier, text_path: str, label_path: str, compute_device: str, feature: str):
    embedding_list = []

    # Forward propagate the model to get a list of predictions and their respective confidence
    for batch in tqdm(DataLoader(a_classifier.dataloader(text_path, label_path),
                                 batch_size=a_classifier.hparams["batch_size"] * 2,
                                 collate_fn=a_classifier.collate, shuffle=False)):

        # Move component to computing device if possible
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(compute_device)

        # Extend the embedding list so far with newly embeded batch
        embedding_list.extend(a_classifier.retrieve_embedding(batch, feature))

    return np.stack(embedding_list)


if __name__ == "__main__":
    embedding()
