import argparse
from collections import Counter
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=Path('CycIC-train-dev'))
    p.add_argument('--mc-lst', type=Path, default=Path('mc_predictions.lst'))
    p.add_argument('--tf-lst', type=Path, default=Path('tf_predictions.lst'))
    p.add_argument('--output-file', type=Path, default=Path('predictions.lst'))
    args = p.parse_args()

    question_path = args.input_dir / f'CycIC_dev_questions.jsonl'

    with question_path.open() as file:
        question_types = [json.loads(line)['questionType'] for line in file]

    all_predictions = []
    with args.mc_lst.open() as mc_file:
        mc_predictions = mc_file.read().split()
    with args.tf_lst.open() as tf_file:
        tf_predictions = tf_file.read().split()
    assert len(mc_predictions) + len(tf_predictions) == len(question_types)
    assert len(mc_predictions) == question_types.count('multiple choice')
    assert len(tf_predictions) == question_types.count('true/false')

    for question_type in question_types:
        if question_type == 'multiple choice':
            all_predictions.append(mc_predictions.pop())
        elif question_type == 'true/false':
            all_predictions.append(tf_predictions.pop())
        else:
            raise ValueError(f'Invalid "questionType" {question_type}')
    assert len(all_predictions) == len(question_types)

    print(Counter(all_predictions))

    with args.output_file.open('w') as file:
        for pred in all_predictions:
            file.write(f'{pred}\n')


if __name__ == '__main__':
    main()
