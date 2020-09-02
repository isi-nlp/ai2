from collections import Counter
import itertools
import json
from statistics import mean, median

with open('Cycic-train-dev/CycIC_training_questions.jsonl', encoding='utf-8-sig') as file:
    train_q = [json.loads(q) for q in file]
with open('Cycic-train-dev/CycIC_dev_questions.jsonl', encoding='utf-8-sig') as file:
    dev_q = [json.loads(q) for q in file]

train_cats = Counter(itertools.chain.from_iterable(q['categories'] for q in train_q))
dev_cats = Counter(itertools.chain.from_iterable(q['categories'] for q in dev_q))
print(train_cats)
print(dev_cats)
print([(label, train_count, dev_cats[label]) for label, train_count in train_cats.most_common()])

print(max(len(q['categories']) for q in train_q + dev_q))
print(mean(len(q['categories']) for q in train_q + dev_q))
print(median(len(q['categories']) for q in train_q + dev_q))
