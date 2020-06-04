import random as r

for task in ['alphanli', 'hellaswag', 'physicaliqa', 'socialiqa']:
    train = open(f"{task}-train-dev/train.jsonl", "r")
    labels = open(f"{task}-train-dev/train-labels.lst", "r")

    all_lines = []
    all_labs = []

    for line in train:
        all_lines.append(line)

    for lab in labels:
        all_labs.append(lab)

    assert len(all_lines) == len(all_labs)

    index = int(len(all_lines)*9/10)
    internal_dev_lines = all_lines[index:]
    internal_dev_labs = all_labs[index:]

    with open(f"{task}-train-dev/internal-dev.jsonl", "w") as d:
        for line in internal_dev_lines:
            d.write(line)

    with open(f"{task}-train-dev/internal-dev-labels.lst", "w") as l:
        for line in internal_dev_labs:
            l.write(line)

