"""
After embed.py parsed embeddings, this script unpickle the result, and using the given parameter in closest.yaml and
out put a tsv file that calculates the closest distance to each desired dev story.
"""

import json
import pathlib
import pickle

import hydra
import torch
import tqdm
from loguru import logger

from utilities.DistanceMeasurer import DistanceMeasurer

# Save root path as hydra will create copies of this code in date specific folder
ROOT_PATH = pathlib.Path(__file__).parent.absolute()


# Helper function that turns a list of [x, y-z] indices to a set
def list_to_set(list_of_index):
    index_set = set()
    for an_index in list_of_index:
        if isinstance(an_index, int):
            index_set.add(an_index)
        elif isinstance(an_index, str):
            [start, end] = an_index.strip().split('-')
            index_set.update(range(int(start.strip()), int(end.strip())))
        else:
            logger.error(f'Unrecognized entry in list to set {an_index}')
    return index_set


@hydra.main(config_path="config/closest.yaml")
def closest(config):
    # Initiate the distance measurer based on the type in the config file
    dm = DistanceMeasurer(config['distance_type'])

    # Load in the embeddings and all necessary information
    with open(ROOT_PATH / config['embedding_loc'], 'rb') as embedding_dict_file:
        embedding_dict = pickle.load(embedding_dict_file)

    # Load in the training text file and the dev text file
    train_text = {}
    with open(embedding_dict['train_path']) as train_file:
        for train_line_number, train_line in enumerate(train_file):
            train_text[train_line_number] = train_line.strip()
    dev_text = {}
    with open(embedding_dict['dev_path']) as dev_file:
        for dev_line_number, dev_line in enumerate(dev_file):
            dev_text[dev_line_number] = dev_line.strip()

    # Load in train stories of interest, dev stories of interest, and influential sets
    train_stories_of_interest = set()
    if config['train_subset']:
        train_stories_of_interest = list_to_set(config['train_subset'])
    dev_stories_of_interest = set()
    if config['dev_subset']:
        dev_stories_of_interest = list_to_set(config['dev_subset'])
    influential_set = set()
    if config['influential_range']:
        influential_set = list_to_set(config['influential_range'])

    # If these subset functions are not used, update the set of interest to be all of train or all of dev, and
    # set influential header to be an empty string
    if not bool(train_stories_of_interest):
        train_stories_of_interest = set(range(len(train_text)))
    if not bool(dev_stories_of_interest):
        dev_stories_of_interest = set(range(len(dev_text)))
    if bool(influential_set):
        influential_header = '\tIn Influential Range'
    else:
        influential_header = ''

    # Extract the important fields from the formula
    context, choices = embedding_dict['task_formula'].split("->")
    important_fields = [a_context.strip() for a_context in context.strip().split("+")] + \
                       [a_choice.strip() for a_choice in choices.strip().split("|")]

    # Create the output file and start writing the headers
    output_file = open(f"{embedding_dict['task_name']}-{config['distance_type']}_dist-top_{config['top_N']}.tsv", 'w')
    output_file.write(f"Task Name:\t{embedding_dict['task_name']}\n")
    output_file.write(f"Distance Function:\t{config['distance_type']}\n\n\n")

    # Get the number of checkpoints
    num_checkpoints = len(embedding_dict['embeddings'])
    num_train_stories = len(train_stories_of_interest)
    num_dev_stories = len(dev_stories_of_interest)
    logger.info(f"Calculating closest {config['top_N']} out of {num_train_stories} train stories "
                f"to {num_dev_stories} dev stories with {num_checkpoints} different checkpoint files")

    # Loop through each dev story and print out top close stories
    for dev_story_id in tqdm.tqdm(dev_stories_of_interest, total=num_dev_stories):
        logger.info(f'Start Calculating Embedding for Dev Story {dev_story_id}')
        a_dev_story = json.loads(dev_text[dev_story_id])
        output_file.write(f'Referencing Dev Story ID\t' + '\t'.join(important_fields) + '\n')
        dev_print_line = f'{dev_story_id}\t'
        for an_important_field in important_fields:
            dev_print_line += f'\t{a_dev_story[an_important_field]}'
        output_file.write(f'{dev_print_line}\n')
        accuracies = torch.zeros(num_checkpoints, dtype=torch.float)

        for ckpt_index, embed_tuple in tqdm.tqdm(enumerate(embedding_dict['checkpoint_names']), total=num_checkpoints):
            # Break out the embedding tuple and write header information
            ckpt_name, train_embed, dev_embed = embed_tuple
            a_dev_embed = dev_embed[dev_story_id]
            output_file.write(f"Seed Name: {ckpt_name}\n")

            # Calculate distances from each train embedding to dev embedding and order it from by increasing distance
            train_distances = []
            for idx_train in tqdm.tqdm(train_stories_of_interest, total=num_train_stories):
                distances = dm.get_distance(train_embed[idx_train], a_dev_embed)
                train_distances.append((idx_train, distances))
            train_distances.sort(key=lambda tup: tup[1])

            # Print out train stories in the table format
            output_file.write('StoryID\tCosine Distance\t' + '\t'.join(important_fields) + influential_header + '\n')
            num_in_influential_set = 0
            for idx, cosine_distance in train_distances[:config['top_N']]:
                train_story = json.loads(train_text[idx])
                print_line = f'{idx}\t{cosine_distance:.3f}'
                for an_important_field in important_fields:
                    print_line += f'\t{train_story[an_important_field]}'

                # If we are considering influential set, we also add this additional column of True or False
                if influential_set:
                    print_line += f'\t{idx in influential_set}'
                    num_in_influential_set += int(idx in influential_set)

                output_file.write(f'{print_line}\n')

            # Add the accuracy to the accumulating torch tensor
            accuracies[ckpt_index] = num_in_influential_set/config['top_N']

        # If we are considering influential set, also out put the mean accuracy of this dev story along with its std
        if influential_set:
            output_file.write(f"\nAccuracies:\t{accuracies.mean():.3f} +/- {2 * accuracies.std():.3f}\n")
        output_file.write('\n')
    output_file.close()


if __name__ == "__main__":
    closest()
