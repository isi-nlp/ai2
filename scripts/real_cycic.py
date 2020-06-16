import argparse
from collections import Counter
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=Path('CycIC-train-dev'))
    p.add_argument('--output-dir', type=Path, default=Path('task_data', 'cycic-real-train-dev'))
    args = p.parse_args()

    for in_split, out_split in (('training', 'train'), ('dev', 'dev')):
        question_path = args.input_dir / f'cycic_{in_split}_questions.jsonl'
        label_path = args.input_dir / f'cycic_{in_split}_labels.jsonl'

        with question_path.open() as file:
            questions = [json.loads(line) for line in file]
        with label_path.open() as file:
            labels = [json.loads(line) for line in file]

        assert len(questions) == len(labels)

        print(f'{in_split}: {Counter(q["questionType"] for q in questions).most_common()}')

        new_questions = []
        new_labels = []

        for question, label in zip(questions, labels):
            assert question['run_id'] == label['run_id']
            assert question['guid'] == label['guid']
            new_question = {k: v for k, v in question.items() if not k.startswith('answer_option')}
            new_question['answer_options'] = [question[k] for k in question if k.startswith('answer_option')]
            if new_question['questionType'] == 'multiple choice':
                assert len(new_question['answer_options']) == 5
            elif new_question['questionType'] == 'true/false':
                assert len(new_question['answer_options']) == 2
            else:
                raise ValueError(f'Invalid "questionType" {new_question["questionType"]}')
            new_questions.append(new_question)
            new_labels.append(label['correct_answer'])

        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / f'{out_split}.jsonl').open('w') as file:
            for question in new_questions:
                file.write(f"{json.dumps(question, ensure_ascii=False)}\n")
        with (args.output_dir / f'{out_split}-labels.lst').open('w') as file:
            for label in new_labels:
                file.write(f"{label}\n")


if __name__ == '__main__':
    main()
