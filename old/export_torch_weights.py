import argparse
import json

import torch


def serialize_difficult_types(foo):
    if isinstance(foo, torch.Tensor):
        return foo.tolist()
    raise TypeError(type(foo))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('input_file', help='Path to JSON file of processed Wikipedia articles')
    p.add_argument('output_file', help='Path to output directory of Wikipedia article files')
    args = p.parse_args()

    with open(args.input_file, 'rb') as in_file:
        data = torch.load(in_file)

    with open(args.output_file, 'w') as out_file:
        json.dump(data, out_file, indent=2, default=serialize_difficult_types)


if __name__ == '__main__':
    main()
