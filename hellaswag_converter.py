import json

# Load in the train and dev stories and their labels
with open('task_data/hellaswag-train-dev/train.jsonl', 'r') as train_story_file:
    train_stories = list(map(lambda json_str: json.loads(json_str.strip()), train_story_file.readlines()))
with open('task_data/hellaswag-train-dev/dev.jsonl', 'r') as dev_story_file:
    dev_stories = list(map(lambda json_str: json.loads(json_str.strip()), dev_story_file.readlines()))

# Loop through the two lists to extract the ending options into their respective fields
for a_story in train_stories + dev_stories:
    a_story['opt0'] = a_story['ending_options'][0]
    a_story['opt1'] = a_story['ending_options'][1]
    a_story['opt2'] = a_story['ending_options'][2]
    a_story['opt3'] = a_story['ending_options'][3]
    del a_story['ending_options']

# Write out the
with open('task_data/hellaswag-train-dev/c_train.jsonl', 'w') as converted_train_story_file:
    for a_train_story in train_stories:
        converted_train_story_file.write(f'{json.dumps(a_train_story)}\n')
with open('task_data/hellaswag-train-dev/c_dev.jsonl', 'w') as converted_dev_story_file:
    for a_dev_story in dev_stories:
        converted_dev_story_file.write(f'{json.dumps(a_dev_story)}\n')
