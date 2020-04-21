import json
import pathlib

import hydra
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
    checkpoint = torch.load(ROOT_PATH / config['checkpoint_path'], map_location=device)
    model = Classifier(config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # Store output lists
    train_embeddings = calculate_embeddings(a_classifier=model,
                                            text_path=ROOT_PATH / config['train_x'],
                                            label_path=ROOT_PATH / config['train_y'],
                                            compute_device=device, feature=config['feature'])
    dev_embeddings = calculate_embeddings(a_classifier=model,
                                          text_path=ROOT_PATH / config['val_x'],
                                          label_path=ROOT_PATH / config['val_y'],
                                          compute_device=device, feature=config['feature'])

    # Write out the output list to the file
    with open(f"{config['model']}-{config['task_name']}_train-{config['formula']}.jsonl", 'w+') as train_file:
        for an_entry in train_embeddings:
            train_file.write(','.join(an_entry))
            train_file.write('\n')
    with open(f"{config['model']}-{config['task_name']}_dev-{config['formula']}.jsonl", 'w+') as dev_file:
        for an_entry in dev_embeddings:
            dev_file.write(','.join([str(float(a_float) for a_float in an_entry)]))
            dev_file.write('\n')


# Helper function for loading embeddings
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

        with torch.no_grad():
            batch["token_type_ids"] = None if "roberta" in a_classifier.hparams["model"] else batch["token_type_ids"]

            # Embed the given batch
            results = a_classifier.embedder(input_ids=batch["input_ids"],
                                            attention_mask=batch["attention_mask"],
                                            token_type_ids=batch["token_type_ids"])
            token_embeddings, *_ = results
            token_embeddings = token_embeddings.mean(dim=1).reshape(a_classifier.hparams["batch_size"] * 2, 2, -1)
            for i, a_label in enumerate(batch['labels']):
                embedding_list.append(token_embeddings[i][a_label])
    return embedding_list


if __name__ == "__main__":
    embedding()
