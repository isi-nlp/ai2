import json
import random as r

f = open("cn-all-cs-multiple-choice-data.jsonl", "r")
l = open("cn-all-cs-multiple-choice-labels.lst", "r")

all_lines = []
all_labs = []

for line in f:
    all_lines.append(line)

for lab in l:
    all_labs.append(lab)

assert len(all_lines) == len(all_labs)

r.seed(1)
c = list(zip(all_lines, all_labs))
r.shuffle(c)
all_lines, all_labs = zip(*c)

eval_num = 2000
train_num = 10000

train_lines = all_lines[eval_num:eval_num+train_num]
train_labs = all_labs[eval_num:eval_num+train_num]
eval_lines = all_lines[:eval_num]
eval_labs = all_labs[:eval_num]

assert len(train_lines) == len(train_labs)
assert len(eval_lines) == len(eval_labs)

print ("Number of data in train: {}".format(len(train_lines)))
print ("Number of data in eval:  {}".format(len(eval_lines)))

ft = open("train.jsonl", "w")
lt = open("train-labels.lst", "w")
fe = open("dev.jsonl", "w")
le = open("dev-labels.lst", "w")

for line in train_lines:
    ft.write(line)
for lab in train_labs:
    lt.write(lab)

for line in eval_lines:
    fe.write(line)
for lab in eval_labs:
    le.write(lab)

ft.close()
lt.close()
fe.close()
le.close()
f.close()
l.close()
