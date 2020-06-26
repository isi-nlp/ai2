import random as r

task = 'alphanli'
train = open(f"{task}-train-dev/train.jsonl", "r")
labels = open(f"{task}-train-dev/train-labels.lst", "r")

all_lines = []
all_labs = []

for line in train:
    all_lines.append(line)

for lab in labels:
    all_labs.append(lab)

assert len(all_lines) == len(all_labs)

data = list(zip(all_lines, all_labs))
r.shuffle(data)
all_lines, all_labs = zip(*data)

index = int(len(all_lines)*9/10)
internal_train_lines = all_lines
internal_train_labs = all_labs

with open(f"{task}-train-dev/train-2.jsonl", "w") as d:
    for line in internal_train_lines:
        d.write(line)

with open(f"{task}-train-dev/train-2-labels.lst", "w") as l:
    for line in internal_train_labs:
        l.write(line)

internal_dev_lines = all_lines[index:]
internal_dev_labs = all_labs[index:]

with open(f"{task}-train-dev/internal-dev-2.jsonl", "w") as d:
    for line in internal_dev_lines:
        d.write(line)

with open(f"{task}-train-dev/internal-dev-2-labels.lst", "w") as l:
    for line in internal_dev_labs:
        l.write(line)

