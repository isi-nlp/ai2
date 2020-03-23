import json

f = open("train.jsonl", "r")
l = open("train-labels.lst", "r")

all_lines = []
all_labs = []

for line in f:
    all_lines.append(line)

for lab in l:
    all_labs.append(lab)

assert len(all_lines) == len(all_labs)

train_num = int(float(len(all_lines)) * 0.25)

train_lines = all_lines[:train_num]
train_labs = all_labs[:train_num]

assert len(train_lines) == len(train_labs)

print ("Number of data in train: {}".format(len(train_lines)))

ft = open("train.jsonl", "w")
lt = open("train-labels.lst", "w")

for line in train_lines:
    ft.write(line)
for lab in train_labs:
    lt.write(lab)

ft.close()
lt.close()
f.close()
l.close()
