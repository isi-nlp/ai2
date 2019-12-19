#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-24 13:50:10
# @Author  : Chenghao Mou (chengham@isi.edu)
# @Link    : https://github.com/ChenghaoMou/ai2

# pylint: disable=unused-wildcard-import
# pylint: disable=no-member

import os
import math
import glob
from typing import *
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def heatmap(dirs: List[str], prefix: str = "dev-", output_file: str = None, offset: int = 0) -> None:
    """Generating a heatmap from a list of directories; Each directory contains a list of files:

        ${prefix}probabilities.lst;
        ${prefix}predictions.lst;
        ${prefix}labels.lst;

    Arguments:
        dirs {List[str]} -- Prediction directories;

    Keyword Arguments:
        prefix {str} -- Prediction File Prefix (default: {"dev-"})
        output_file {str} -- Output image file name (default: {None})

    Returns:
        None -- [description]
    """
    dirs = map(Path, dirs)

    labels = []
    result = {}

    for d in dirs:
        name, *weight, task, _ = [x for x in os.path.split(d)[-1].split("-") if x]
        weight = "-".join(weight)

        with open(d / (prefix + "labels.lst"), "r") as model_labels, \
                open(d / (prefix + "probabilities.lst"), "r") as model_probabilities, \
                open(d / (prefix + "predictions.lst"), "r") as model_predictions:

            model_predictions_values = list(map(int, model_predictions.readlines()))
            model_labels_value = list(map(int, model_labels.readlines()))
            model_probabilities_value = list(map(lambda l: list(map(float, l.split("\t"))), model_probabilities.readlines()))
            if labels == [] or labels is None:
                labels = model_labels_value
            else:
                if [(x, y) for x, y in zip(labels, model_labels_value) if x != y] != []:
                    logger.warning("Inconsistent labels " + str(d))
                    continue

            differences = []

            for i, (pred, truth) in enumerate(zip(labels, model_predictions_values)):

                if pred != truth:
                    differences.append(abs(model_probabilities_value[i][pred - offset] - model_probabilities_value[i][truth - offset]))

                else:
                    differences.append(np.NaN)
            accuracy = 1 - np.count_nonzero(~np.isnan(np.asarray(differences))) / len(differences)
            # print(len(differences))
            # print(weight.split('-', 1))
            result['\n'.join(weight.split('-', 1)) + f"\n({accuracy * 100:.2f})"] = np.asarray(differences)
    dataframe = pd.DataFrame(result)
    dataframe = dataframe.iloc[dataframe.isnull().sum(1).sort_values(ascending=False).index]
    dataframe = dataframe.transpose()

    ax = sns.heatmap(dataframe, cmap="autumn", mask=dataframe.isnull(), xticklabels=False)
    ax.hlines([i for i in range(dataframe.shape[0] + 1)], linewidth=1, *ax.get_xlim())
    ax.vlines([0, dataframe.shape[1]], linewidth=1, *ax.get_ylim())

    ax.figure.tight_layout()

    plt.savefig(output_file)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dirs', nargs='+', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--offset', type=int, required=True)

    args = parser.parse_args()
    heatmap(args.pred_dirs, output_file=args.output_file, offset=args.offset)


# python visualize/heatmap.py --pred_dir output/roberta-roberta-large-alphanli--pred output/roberta-roberta-base-alphanli--pred output/xlnet-xlnet-base-cased-alphanli--pred output/xlnet-xlnet-large-cased-alphanli--pred output/bert-bert-large-cased-alphanli--pred output/bert-bert-base-cased-alphanli--pred output/distilbert-distilbert-base-uncased-alphanli--pred --output_file anli.svg --offset 1
# python visualize/heatmap.py --pred_dir output/roberta-roberta-large-hellaswag--pred output/roberta-roberta-base-hellaswag--pred output/xlnet-xlnet-base-cased-hellaswag--pred output/xlnet-xlnet-large-cased-hellaswag--pred output/bert-bert-large-cased-hellaswag--pred output/bert-bert-base-cased-hellaswag--pred output/distilbert-distilbert-base-uncased-hellaswag--pred --output_file hellaswag.svg --offset 0
# python visualize/heatmap.py --pred_dir output/roberta-roberta-large-physicaliqa--pred output/roberta-roberta-base-physicaliqa--pred output/xlnet-xlnet-base-cased-physicaliqa--pred output/xlnet-xlnet-large-cased-physicaliqa--pred output/bert-bert-large-cased-physicaliqa--pred output/bert-bert-base-cased-physicaliqa--pred output/distilbert-distilbert-base-uncased-physicaliqa--pred --output_file piqa.svg --offset 0
# python visualize/heatmap.py --pred_dir output/roberta-roberta-large-socialiqa--pred output/roberta-roberta-base-socialiqa--pred output/xlnet-xlnet-base-cased-socialiqa--pred output/xlnet-xlnet-large-cased-socialiqa--pred output/bert-bert-large-cased-socialiqa--pred output/bert-bert-base-cased-socialiqa--pred output/distilbert-distilbert-base-uncased-socialiqa--pred --output_file siqa.svg --offset 1
