import pandas as pd
import json
from collections import Counter
from visual import read_csv


def source(path):

    dataset = map(lambda l: json.loads(l).get('dataset', 'unknown'), open(path, 'r').readlines())
    return list(dataset)


def accuracy(source, eval_path):
    df = read_csv(eval_path)
    counter = Counter()
    denominator = Counter()

    # print(df.iloc[-3:])

    def correct(slice):

        assert all(slice['Premise'] == slice['Premise'].values.tolist()[0]), slice
        assert len(slice) == 4, 'Wrong number of choices'

        res = slice[slice['Truth'] == 'True'].values.tolist()[0].count('True') - 1

        return res == 1
    
    assert len(df) % 4 == 0, "Wrong number of rows!"
    assert len(source) == len(df)//4, f"Wrong number of source {len(source)} {len(df)//4}"

    for i in range(len(df)//4):
        slice = df.iloc[i*4:(i+1)*4]
        provenance = source[i]
        denominator[provenance] += 1
        if correct(slice):
            counter[provenance] += 1

    return {k: (v, denominator[k]) for k, v in counter.items()}


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser('Wikihow VS. ActivityNet')
    parser.add_argument('--eval_jsonl', type=str, required=True)
    parser.add_argument('--eval_tsv', type=str, required=True)
    args = parser.parse_args()

    from loguru import logger

    for k, (c, t) in accuracy(source(args.eval_jsonl), args.eval_tsv).items():
        logger.info(f"Accuracy: {k}: {100 * c / t:.2f} {c}/{t}")
