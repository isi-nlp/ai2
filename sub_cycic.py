import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-file', type=Path, default=Path('data' 'cycic.jsonl'))
    p.add_argument('--output-file', type=Path, default=Path('temp.jsonl'))
    args = p.parse_args()

    with args.input_file.open() as file:
        questions = [json.loads(line) for line in file]

    fake_questions = []

    for question in questions:
        if question['questionType'] == 'multiple choice':
            assert len([k for k in question if k.startswith('answer_option')]) == 5
            fake_questions.append(question)
        elif question['questionType'] == 'true/false':
            assert len([k for k in question if k.startswith('answer_option')]) == 2
            new_question = question.copy()
            new_question['answer_option2'] = 'INVALID ANSWER'
            new_question['answer_option3'] = 'INVALID ANSWER'
            new_question['answer_option4'] = 'INVALID ANSWER'
            assert len([k for k in new_question if k.startswith('answer_option')]) == 5
            fake_questions.append(new_question)
        else:
            raise ValueError(f'Invalid "questionType" {question["questionType"]}')

    with args.output_file.open('w') as file:
        for question in fake_questions:
            file.write(f"{json.dumps(question, ensure_ascii=False)}\n")


if __name__ == '__main__':
    main()
