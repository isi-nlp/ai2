"""Converts raw RACE data set into usable format.

Adapted from https://github.com/pytorch/fairseq/blob/master/examples/roberta/preprocess_RACE.py.
"""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Sequence, Tuple


@dataclass
class InputExample:
    """Storage for IDs, context, choices, and answer for a single question.

    Attributes:
        index: Example ID.
        context_id: Context ID.
        paragraph: Context text.
        qa_list: Question-answer choices.
        label: Index of correct answer.
    """
    index: int
    context_id: str
    paragraph: str
    qa_list: Sequence[str]
    label: int


def get_examples(data_dir: Path, example_index: int = 0) -> Tuple[Sequence[InputExample], int]:
    """Extract paragraph and question-answer list from each JSON file.

    Args:
        data_dir: Directory containing split data.
        example_index: Starting index for example indices.

    Returns:
        A sequence of examples, and the next example index to use.
    """
    examples = []

    for cur_path in sorted(data_dir.rglob("*.txt"), key=lambda x: int(x.stem)):
        with cur_path.open("r") as file:
            cur_data = json.load(file)
            context_id = cur_data["id"]
            answers = cur_data["answers"]
            options = cur_data["options"]
            questions = cur_data["questions"]
            context = cur_data["article"].replace("\n", " ")
            context = re.sub(r"\s+", " ", context)
            for i, _ in enumerate(answers):
                label = ord(answers[i]) - ord("A")
                qa_list = []
                question = questions[i]
                for j in range(4):
                    option = options[i][j]
                    if "_" in question:
                        qa_cat = question.replace("_", option)
                    else:
                        qa_cat = " ".join([question, option])
                    qa_cat = re.sub(r"\s+", " ", qa_cat)
                    qa_list.append(qa_cat)
                examples.append(InputExample(example_index, context_id, context, qa_list, label))
                example_index += 1
                if example_index % 10_000 == 0:
                    print(f"Processed {example_index:,} examples")

    return examples, example_index


def main() -> None:
    """Helper script to extract paragraphs questions and answers from RACE datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help='Input directory for downloaded RACE dataset.')
    parser.add_argument("--output-dir", type=Path, required=True,
                        help='Output directory for extracted data.')
    args = parser.parse_args()

    if not args.output_dir.exists():
        os.makedirs(args.output_dir, exist_ok=True)

    start_index = 0
    for set_type in ["train", "dev", "test"]:
        examples, start_index = get_examples(args.input_dir / set_type, start_index)

        samples = []
        labels = []
        for example in examples:
            output = {
                "index": example.index,
                "id": example.context_id,
                "paragraph": example.paragraph,
                "qa_list": example.qa_list
            }
            samples.append(json.dumps(output, ensure_ascii=False))
            labels.append(example.label)

        with (args.output_dir / f"{set_type}.jsonl").open("w") as file:
            file.write("".join(f"{sample}\n" for sample in samples))

        with (args.output_dir / f"{set_type}-labels.lst").open("w") as file:
            file.write("".join(f"{label}\n" for label in labels))

        print(f"Finished processing {set_type}")


if __name__ == "__main__":
    main()
