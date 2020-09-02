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
