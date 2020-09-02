# CycIC Analysis

This document includes analysis of the categories of questions in CycIC and the overlap between the train and dev sets.

## Categories

The following data are counts of the categories of questions in each split. Some questions belong to multiple categories, but those questions are not extremely common. Questions belong to at most 3 categories, the mean number of categories per question is ~1.28, and the median number is 1.

- Train set:
  - `theory of mind`: 2330
  - `logical reasoning`: 2292
  - `temporal sequences`: 1484
  - `object properties`: 1309
  - `temporal reasoning`: 1251
  - `classification`: 1017
  - `disjointness`: 683
  - `culture`: 550
  - `arithmetic`: 516
  - `events`: 491
  - `conceptual`: 388
  - `causal reasoning`: 348
  - `norms`: 302
  - `quotation`: 225
  - `composition`: 198
  - `social relations`: 120
  - `animals`: 99
  - `nature`: 81
  - `capabilities`: 60
  - `counterfactual`: 36
- Dev set:
  - `logical reasoning`: 321
  - `theory of mind`: 312
  - `temporal sequences`: 230
  - `object properties`: 201
  - `temporal reasoning`: 189
  - `classification`: 147
  - `disjointness`: 84
  - `events`: 78
  - `arithmetic`: 73
  - `culture`: 69
  - `conceptual`: 50
  - `causal reasoning`: 46
  - `norms`: 41
  - `quotation`: 28
  - `composition`: 26
  - `animals`: 17
  - `social relations`: 14
  - `nature`: 8
  - `capabilities`: 7
  - `counterfactual`: 6

### Categories Code

```python
from collections import Counter
import itertools
import json
from statistics import mean, median

train_q = [json.loads(q) for q in open('Cycic-train-dev/CycIC_training_questions.jsonl')]
dev_q = [json.loads(q) for q in open('Cycic-train-dev/CycIC_dev_questions.jsonl')]

train_cats = Counter(itertools.chain.from_iterable(q['categories'] for q in train_q))
dev_cats = Counter(itertools.chain.from_iterable(q['categories'] for q in dev_q))
print(train_cats)
print(dev_cats)

print(max(len(q['categories']) for q in train_q + dev_q))
print(mean(len(q['categories']) for q in train_q + dev_q))
print(median(len(q['categories']) for q in train_q + dev_q))
```

## Overlap

There is some overlap between the train and dev sets, which is concern.

There is also some repetition within each split, but that is ignored for these analysis. The overlapping questions are therefore likely higher in number than reported here.

- There are 14 true/false questions that appear in both sets. The choices are obviously the same between both instances.
- There are 183 multiple choice questions that appear in both sets.
  - There are 0 questions that appear in both sets with the same question and answers (regardless of answer order).
  - There are 77 correct question-answer pairs that appear in both sets.
  - There are 322 question-answer pairs (ignoring correctness) that appear in both sets.

### Overlap Code

```python
import itertools
import json

train_q = [json.loads(q) for q in open('Cycic-train-dev/CycIC_training_questions.jsonl')]
dev_q = [json.loads(q) for q in open('Cycic-train-dev/CycIC_dev_questions.jsonl')]
train_l = [json.loads(q) for q in open('Cycic-train-dev/CycIC_training_labels.jsonl')]
dev_l = [json.loads(q) for q in open('Cycic-train-dev/CycIC_dev_labels.jsonl')]

train_tf_q = set(q['question'] for q in train_q if q['questionType'] == 'true/false')
dev_tf_q = set(q['question'] for q in dev_q if q['questionType'] == 'true/false')
print(len(train_tf_q & dev_tf_q))

train_mc_q = set(q['question'] for q in train_q if q['questionType'] == 'multiple choice')
dev_mc_q = set(q['question'] for q in dev_q if q['questionType'] == 'multiple choice')
print(len(train_mc_q & dev_mc_q))

train_mc_q_as = set((q['question'], tuple(sorted([q['answer_option0'], q['answer_option1'], q['answer_option2'], q['answer_option3'], q['answer_option4']]))) for q in train_q if q['questionType'] == 'multiple choice')
dev_mc_q_as = set((q['question'], tuple(sorted([q['answer_option0'], q['answer_option1'], q['answer_option2'], q['answer_option3'], q['answer_option4']]))) for q in dev_q if q['questionType'] == 'multiple choice')
print(len(train_mc_q_as & dev_mc_q_as))

train_mc_q_correct_a = set((q['question'], q[f'answer_option{l["correct_answer"]}']) for q, l in zip(train_q, train_l) if q['questionType'] == 'multiple choice')
dev_mc_q_correct_a = set((q['question'], q[f'answer_option{l["correct_answer"]}']) for q, l in zip(dev_q, dev_l) if q['questionType'] == 'multiple choice')
print(len(train_mc_q_correct_a & dev_mc_q_correct_a))

train_mc_q_a = set(itertools.chain.from_iterable([[(q[0], a) for a in q[1:]] for q in [(c['question'], c['answer_option0'], c['answer_option1'], c['answer_option2'], c['answer_option3'], c['answer_option4']) for c in train_q if c['questionType'] == 'multiple choice']]))
dev_mc_q_a = set(itertools.chain.from_iterable([[(q[0], a) for a in q[1:]] for q in [(c['question'], c['answer_option0'], c['answer_option1'], c['answer_option2'], c['answer_option3'], c['answer_option4']) for c in dev_q if c['questionType'] == 'multiple choice']]))
print(len(train_mc_q_a & dev_mc_q_a))
```
