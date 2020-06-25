"""
After parsing embeddings, the script compare train embedding to the dev embedding and write out top correlation
entries
"""

import json
import pathlib
import pickle

import hydra
import numpy as np
import torch
import tqdm

from utilities.DistanceMeasurer import DistanceMeasurer

# Save root path as hydra will create copies of this code in date specific folder
ROOT_PATH = pathlib.Path(__file__).parent.absolute()


@hydra.main(config_path="config/getClosest.yaml")
def get_closest(config):

    # Initiate the distance measurer based on the type in the config file
    dm = DistanceMeasurer(config['distance_type'])

    # Load in the embeddings and all necessary information
    with open(ROOT_PATH/config['embedding_loc'], 'rb') as embedding_dict_file:
        embedding_dict = pickle.load(embedding_dict_file)

    # Load in the training file and the dev file
    train_text = {}
    with open(embedding_dict['train_path']) as train_file:
        for train_line_number, train_line in enumerate(train_file):
            train_text[train_line_number] = train_line.strip()
    dev_text = {}
    with open(embedding_dict['dev_path']) as dev_file:
        for dev_line_number, dev_line in enumerate(dev_file):
            dev_text[dev_line_number] = dev_line.strip()

    # Extract the important fields from the formula
    context, choices = embedding_dict['task_formula'].split("->")
    important_fields = [a_context.strip() for a_context in context.strip().split("+")] + \
                       [a_choice.strip() for a_choice in choices.strip().split("|")]

    # TODO: Better file name
    # Create the output file and start writing to it
    output_file = open(f"output.tsv", 'w')
    output_file.write(f"Task Name:\t{embedding_dict['task_name']}\n")
    output_file.write(f"Distance Function:\t{config['distance_type']}\n\n\n")

    # Loop through each dev story and print out top close stories
    for dev_story_id in range(len(embedding_dict['dev_embed'][0])):
        output_file.write(f'Referencing Dev Story: {dev_text[dev_story_id]}\n\n')

        for checkpoint_index, checkpoint_name in enumerate(embedding_dict['checkpoint_names']):
            output_file.write(f"Seed Name: {checkpoint_name}\n")

            # Retrieve embedding associated to this checkpoint
            dev_embed = embedding_dict['dev_embed'][checkpoint_index][dev_story_id]
            train_embed = embedding_dict['train_embed'][checkpoint_index]

            # Calculate distances from each train embedding to dev embedding and order it from by increasing distance
            train_distances = []
            for idx_train, a_train_embed in tqdm.tqdm(enumerate(train_embed), total=len(train_embed)):
                distances = dm.get_distance(a_train_embed, dev_embed)
                train_distances.append((idx_train, distances))
            train_distances.sort(key=lambda tup: tup[1])

            # Print out train stories in the table format that is useful
            output_file.write('StoryID\tCosine Distance\t' + '\t'.join(important_fields) + '\n')
            for idx, cosine_distance in train_distances[:config['top_N']]:
                train_story = json.loads(train_text[idx])
                print_line = f'{idx}\t{cosine_distance}'
                for an_important_field in important_fields:
                    print_line += f'\t{train_story[an_important_field]}'
                output_file.write(f'{print_line}\n')

            output_file.write('\n')

    output_file.close()


if __name__ == "__main__":
    get_closest()

