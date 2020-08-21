"""
NOTE: This task generated at the moment is NOT an easy task for transformers to train on - need to come up with a better
task to serve as the dummy task for transformers

Dummy Anli is used to generate a dumbed down version of the alphanli task. This is used as a sanity test on if a model
is training properly or not. If it is a normal model it should be relatively easy to achieve very good performance on
this task. This task is to randomly insert word 'oisadf' into either of the two observations of anli, and adjust the
hypothesis to say which of the observations has this word. It also generates the correct label.

At the moment it is using the 2% of Full Alphanli training set.
"""

import json
import random

dev_count = 300

# Load in Train
train = []
with open('../task_data/AlphaNLI/increment_training/2/train.jsonl', 'r') as train_file:
    for a_line in train_file:
        a_story = json.loads(a_line.strip())
        a_story['hyp1'] = '1st'
        a_story['hyp2'] = '2nd'
        train.append(a_story)

# Modify train and generate the correct labels
label = []
for a_train in train:
    k = random.randint(0, 1)
    if k == 0:
        sentence = a_train['obs1'].split(' ')
        sentence.insert(random.randrange(len(sentence)+1), 'oisadf')
        a_train['obs1'] = ' '.join(sentence)
        label.append(1)
    else:
        sentence = a_train['obs2'].split(' ')
        sentence.insert(random.randrange(len(sentence)+1), 'oisadf')
        a_train['obs2'] = ' '.join(sentence)
        label.append(2)

# Write out the train and the dev for the dumbed down task
with open('../task_data/AlphaNLI/dummy/train-labels.lst', 'w') as output:
    for a_label in label[:-dev_count]:
        output.write(f'{a_label}\n')
with open('../task_data/AlphaNLI/dummy/train.jsonl', 'w') as output:
    for a_train in train[:-dev_count]:
        output.write(f'{json.dumps(a_train)}\n')
with open('../task_data/AlphaNLI/dummy/dev-labels.lst', 'w') as output:
    for a_label in label[-dev_count:]:
        output.write(f'{a_label}\n')
with open('../task_data/AlphaNLI/dummy/dev.jsonl', 'w') as output:
    for a_train in train[-dev_count:]:
        output.write(f'{json.dumps(a_train)}\n')
