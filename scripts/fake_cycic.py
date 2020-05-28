import argparse
from collections import Counter
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=Path('CycIC-train-dev'))
    p.add_argument('--output-dir', type=Path, default=Path('task_data', 'cycic-fake-train-dev'))
    args = p.parse_args()

    for in_split, out_split in (('training', 'train'), ('dev', 'dev')):
        question_path = args.input_dir / f'CycIC_{in_split}_questions.jsonl'
        label_path = args.input_dir / f'CycIC_{in_split}_labels.jsonl'

        with question_path.open() as file:
            questions = [json.loads(line) for line in file]
        with label_path.open() as file:
            labels = [json.loads(line) for line in file]

        assert len(questions) == len(labels)

        print(f'{in_split}: {Counter(q["questionType"] for q in questions).most_common()}')

        fake_questions = []
        fake_labels = []

        for question, label in zip(questions, labels):
            assert question['run_id'] == label['run_id']
            assert question['guid'] == label['guid']
            if question['questionType'] == 'multiple choice':
                assert len([k for k in question if k.startswith('answer_option')]) == 5
                fake_questions.append(question)
                fake_labels.append(label['correct_answer'])
            elif question['questionType'] == 'true/false':
                assert len([k for k in question if k.startswith('answer_option')]) == 2
                new_question = question.copy()
                new_question['answer_option2'] = 'INVALID ANSWER'
                new_question['answer_option3'] = 'INVALID ANSWER'
                new_question['answer_option4'] = 'INVALID ANSWER'
                assert len([k for k in new_question if k.startswith('answer_option')]) == 5
                fake_questions.append(new_question)
                fake_labels.append(label['correct_answer'])
            else:
                raise ValueError(f'Invalid "questionType" {question["questionType"]}')

        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / f'{out_split}.jsonl').open('w') as file:
            for question in fake_questions:
                file.write(f"{json.dumps(question, ensure_ascii=False)}\n")
        with (args.output_dir / f'{out_split}-labels.lst').open('w') as file:
            for label in fake_labels:
                file.write(f"{label}\n")


if __name__ == '__main__':
    main()
