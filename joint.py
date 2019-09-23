# encoding: utf-8
# Created by chenghaomou at 9/22/19
# Contact: mouchenghao at gmail dot com
# Description: #ENTER

# eval.tsv
# eval-train.tsv
# eval-train.truth
# eval.truth
import glob
from pathlib import Path
from typing import *

from loguru import logger
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def load_dataset(task: str, path: str) -> Tuple:
    train_x = []
    train_y = []
    dev_x = []
    dev_y = []
    logger.info(task)
    logger.info(path)

    path = Path(path)
    train_truth_files = list(glob.glob(f'{path}/{task}-*eval-train.truth'))
    train_files = list(glob.glob(f'{path}/{task}-*eval-train.tsv'))

    dev_truth_files = list(glob.glob(f'{path}/{task}-*eval.truth'))
    dev_files = list(glob.glob(f'{path}/{task}-*-eval.tsv'))

    # logger.info("\n".join(train_files))
    # logger.info("\n".join(dev_files))

    assert len(train_files) == len(dev_files), "Mismatch feature files"

    with open(train_truth_files[0], "r") as input_file:
        train_y = list(map(int, input_file.readlines()))

    with open(dev_truth_files[0], "r") as input_file:
        dev_y = list(map(int, input_file.readlines()))

    for f in sorted(train_files):
        with open(f, "r") as input_file:
            train_x.extend(zip(*map(lambda l: list(map(float, l.strip('\r\n ').split('\t'))), input_file.readlines())))

    for f in sorted(dev_files):
        with open(f, "r") as input_file:
            try:
                dev_x.extend(
                    zip(*map(lambda l: list(map(float, l.strip('\r\n ').split('\t'))), input_file.readlines())))
            except Exception as e:
                logger.debug(f)
                exit(0)

    return list(zip(*train_x)), train_y, list(zip(*dev_x)), dev_y


def main(args):
    for task in args.tasks:
        train_x, train_y, dev_x, dev_y = load_dataset(task, args.path)

        model = LinearDiscriminantAnalysis(store_covariance=True)
        model.fit(train_x, train_y)
        logger.info(f"{task}: {model.score(dev_x, dev_y)}")
        # logger.info(model.covariance_)
        # logger.info(model.coef_)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Joint prediction using logistic regression')
    parser.add_argument('--tasks', nargs='+',
                        default=['anli', 'hellaswag', 'physicaliqa', 'socialiqa', 'vcrqa', 'vcrqar'])
    parser.add_argument('--path', type=str, default='outputs')
    args = parser.parse_args()

    main(args)
