import json
import pathlib

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

    # Load in the check pointed model from check point file
    device = 'cpu' if not torch.cuda.is_available() else "cuda"
    checkpoint = torch.load(ROOT_PATH / config['checkpoints_dir'] / f"{config['checkpoint_name']}.ckpt",
                            map_location=device)
    model = Classifier(config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # Store output lists
    np.save(f"{config['model']}-{config['task_name']}_train-{config['checkpoint_name']}-{config['feature']}.npy",
            calculate_embeddings(a_classifier=model,
                                 text_path=ROOT_PATH / config['train_x'],
                                 label_path=ROOT_PATH / config['train_y'],
                                 compute_device=device, feature=config['feature']))
    np.save(f"{config['model']}-{config['task_name']}_dev-{config['checkpoint_name']}-{config['feature']}.npy",
            calculate_embeddings(a_classifier=model,
                                 text_path=ROOT_PATH / config['val_x'],
                                 label_path=ROOT_PATH / config['val_y'],
                                 compute_device=device, feature=config['feature']))


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
