import json
import pathlib

import hydra
import torch
from loguru import logger

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

    # Store output lists
    output_list = calculate_embeddings(a_classifier=model,
                                       text_path=ROOT_PATH / config['train_x'],
                                       label_path=ROOT_PATH / config['train_y'])
    output_list.extend(calculate_embeddings(a_classifier=model,
                                            text_path=ROOT_PATH / config['val_x'],
                                            label_path=ROOT_PATH / config['val_y']))

    # Write out the output list to the file as a jsonl file
    save_path = pathlib.Path(f"{config['model']}-{config['task_name']}-{config['formula']}.jsonl")
    with open(save_path, 'w+') as output_file:
        for an_entry in output_list:
            json.dump(an_entry, output_file)
            output_file.write('\n')


# Helper function for loading embeddings
def calculate_embeddings(a_classifier: Classifier, text_path: str, label_path: str):
    # TODO
    return []


if __name__ == "__main__":
    embedding()
