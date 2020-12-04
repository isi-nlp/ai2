import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import random


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=Path('cycic3_sample'))
    p.add_argument('--output-dir', type=Path, default=Path('task_data', 'cycic3'))
    args = p.parse_args()

    for partition in ('a', 'b'):
        question_path = args.input_dir / f'cycic3{partition}_sample_questions.jsonl'
        label_path = args.input_dir / f'cycic3{partition}_sample_labels.jsonl'

        with question_path.open() as file:
            questions = [json.loads(line) for line in file]
        with label_path.open() as file:
            labels = [json.loads(line) for line in file]

        assert len(questions) == len(labels)

        new_questions = []
        new_labels = []

        for question, label in zip(questions, labels):
            assert question['run_id'] == label['run_id']
            assert question['guid'] == label['guid']
            new_question = {k: v for k, v in question.items() if not k.startswith('answer_option')}
            new_question['answer_options'] = [question[k] for k in question if k.startswith('answer_option')]
            if new_question['questionType'] == 'true/false':
                assert len(new_question['answer_options']) == 2
            else:
                raise ValueError(f'Invalid "questionType" {new_question["questionType"]}')
            new_questions.append(new_question)
            new_labels.append(label['correct_answer'])

        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / f'cycic3{partition}.jsonl').open('w') as file:
            for question in new_questions:
                file.write(f"{json.dumps(question, ensure_ascii=False)}\n")
        with (args.output_dir / f'cycic3{partition}-labels.lst').open('w') as file:
            for label in new_labels:
                file.write(f"{label}\n")


if __name__ == '__main__':
    main()
