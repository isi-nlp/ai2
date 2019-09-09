import pandas as pd
import json
from collections import Counter
from visual import read_csv


def source(path):

    dataset = map(json.loads, open(path, 'r').readlines())
    return [d['dataset']]


def accuracy(source, eval_path):
    df = read_csv(source)
    counter = Counter()

    def correct(slice):

        assert all(slice['Premise'] == slice['Premise'].values.tolist()[0]), slice
        assert len(slice) == num_choice, 'Wrong number of choices'

        res = slice[slice['Truth'] == 'True'].values.tolist()[0].count('True') - 1

        return res == 1

    for i in range(len(df)/4):
        slice = df.loc[i*4:(i+1)*4]
        provenance = source[i]
        if correct(slice):
            counter[provenance] += 1

    return {k: v/len(df) for k, v in counter.items()}


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser('Wikihow VS. ActivityNet')
    parser.add_argument('--eval_jsonl', type=str, required=True)
    parser.add_argument('--eval_tsv', type=str, required=True)
    args = parser.parse_args()

    from loguru import logger

    for k, v in accuracy(args.eval_jsonl, args.eval_tsv):
        logger.info(f"Accuracy: {k}: {100 * v:.2f}")
