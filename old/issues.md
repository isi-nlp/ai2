# CycIC Issues

As I have been working with the CycIC data set, I have come across multiple issues with it that should either be fixed or clarified.

## Distribution

The 2020-06-24 version of the data set (available at [CycIC-train-dev.zip](https://storage.googleapis.com/ai2-mosaic/public/cycic/CycIC-train-dev.zip)) has some annoyances with distribution.

The Zip contains a `__MACOSX`, which can safely be discarded. This appears to be an artifact of zipping with macOS's builtin GUI utility instead of `zip` through the terminal.

Python could not handle `cycic_dev_questions.jsonl` and `cycic_training_questions.jsonl` by default because both have the [byte order mark](https://en.wikipedia.org/wiki/Byte_order_mark) (BOM) and therefore do not register to Python as UTF-8. This can be seen by running `ls | xargs -I % bash -c "echo % && head -c 10 % | hexdump -C"` inside the unzipped `CycIC-train-dev` directory. The RoBERTa model's [`CycicLeaderboardProcessor`](https://github.com/cycorp/cycic-transformers/blob/c2d6dafa639b0839abae25b09de1c6dada67ce44/multiple-choice/utils_multiple_choice.py#L345) handles this by using `encoding='utf-8-sig'`. Removing the BOMs allows Python to automatically detect the files as UTF-8 without special code.

## Leaderboard

I am curious as to why the listed human performance is the same as before. Is it by chance, or was the value not updated for the new data set?

## Documentation

The question files have a `blanks` field that is undocumented in with the rest of the fields. What does it mean? Either the documentation should be updated to include it, or it should be removed from the files.

## Overlap

After some initial findings from Jon May, I did a deeper analysis into the overlap between training and dev sets. There are 324 (7.21% of dev) multiple choice question-answer pairs and 22 (3.57% of dev) true/false questions that appear in both splits. This is a small amount, but it still could be causing unintentional problems. In addition, there is repetition within each split that is ignored here. The code (run with Python 3.7.6) obtained the numbers given above.

```python
import itertools
import json


def get_q_a_pairs(data):
    mc_questions = [q for q in data if q['questionType'] == 'multiple choice']
    questions = [q['question'] for q in mc_questions]
    answers = [[q[f'answer_option{i}'] for i in range(5)] for q in mc_questions]
    pairs = [[(q, a) for a in ans] for q, ans in zip(questions, answers)]
    return set(itertools.chain.from_iterable(pairs))


with open('Cycic-train-dev/CycIC_training_questions.jsonl', encoding='utf-8-sig') as file:
    train = [json.loads(q) for q in file]
with open('Cycic-train-dev/CycIC_dev_questions.jsonl', encoding='utf-8-sig') as file:
    dev = [json.loads(q) for q in file]

train_mc_q_a = get_q_a_pairs(train)
dev_mc_q_a = get_q_a_pairs(dev)
num_mc_overlap = len(train_mc_q_a & dev_mc_q_a)
print('Overlapping multiple choice question-answer pairs: '
      f'{num_mc_overlap} ({num_mc_overlap / len(dev_mc_q_a):.2%} of dev)')

train_tf_q = set(q['question'] for q in train if q['questionType'] == 'true/false')
dev_tf_q = set(q['question'] for q in dev if q['questionType'] == 'true/false')
num_tf_overlap = len(train_tf_q & dev_tf_q)
print('Overlapping true/false questions: '
      f'{num_tf_overlap} ({num_tf_overlap / len(dev_tf_q):.2%} of dev)')
```
