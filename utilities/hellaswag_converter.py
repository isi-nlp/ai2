"""
Original HellaSwag data comes in a different format - using a list of ending options rather than 4 discrete ending
option. This script converts the original config to fit with the rest of the ai2 tasks
"""
import json
import yaml

HELLASWAG_PATH = '../task_data/hellaswag-train-dev/'

# Load in the train and dev stories
with open(HELLASWAG_PATH + 'train.jsonl', 'r') as train_story_file:
    train_stories = list(map(lambda json_str: json.loads(json_str.strip()), train_story_file.readlines()))
with open(HELLASWAG_PATH + 'dev.jsonl', 'r') as dev_story_file:
    dev_stories = list(map(lambda json_str: json.loads(json_str.strip()), dev_story_file.readlines()))

# Loop through the two lists to extract the ending options into their respective fields
for a_story in train_stories + dev_stories:
    a_story['opt0'] = a_story['ending_options'][0]
    a_story['opt1'] = a_story['ending_options'][1]
    a_story['opt2'] = a_story['ending_options'][2]
    a_story['opt3'] = a_story['ending_options'][3]
    del a_story['ending_options']

# Write out the converted dataset, we don't need to rewrite the labels because the original one still works
with open(HELLASWAG_PATH + 'c_train.jsonl', 'w') as converted_train_story_file:
    for a_train_story in train_stories:
        converted_train_story_file.write(f'{json.dumps(a_train_story)}\n')
with open(HELLASWAG_PATH + 'c_dev.jsonl', 'w') as converted_dev_story_file:
    for a_dev_story in dev_stories:
        converted_dev_story_file.write(f'{json.dumps(a_dev_story)}\n')

# Write out the modified config file for converted hella swag
with open('../config/task/hellaswag.yaml', 'w') as config_file:
    config_dict = {'task_name': 'hellaswag',
                   'train_x': "task_data/hellaswag-train-dev/c_train.jsonl",
                   'train_y': "task_data/hellaswag-train-dev/train-labels.lst",
                   'val_x': "task_data/hellaswag-train-dev/c_dev.jsonl",
                   'val_y': "task_data/hellaswag-train-dev/dev-labels.lst",
                   'formula': "ctx_a + ctx_b -> opt0|opt1|opt2|opt3"}
    documents = yaml.dump(config_dict, config_file)
