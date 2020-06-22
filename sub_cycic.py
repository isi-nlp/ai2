import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-file', type=Path, default=Path('data', 'cycic.jsonl'))
    p.add_argument('--output-file', type=Path, default=Path('temp.jsonl'))
    args = p.parse_args()

    with args.input_file.open(encoding='utf-8-sig') as file:
        questions = [json.loads(line) for line in file]

    fake_questions = []

    for question in questions:
        new_question = {k: v for k, v in question.items() if not k.startswith('answer_option')}
        new_question['answer_options'] = [question[k] for k in question if k.startswith('answer_option')]
        if new_question['questionType'] == 'multiple choice':
            assert len(new_question['answer_options']) == 5
        elif new_question['questionType'] == 'true/false':
            assert len(new_question['answer_options']) == 2
        else:
            raise ValueError(f'Invalid "questionType" {question["questionType"]}')
        fake_questions.append(new_question)

    with args.output_file.open('w') as file:
        for question in fake_questions:
            file.write(f"{json.dumps(question, ensure_ascii=False)}\n")


if __name__ == '__main__':
    main()
