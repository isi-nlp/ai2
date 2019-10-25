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

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
# TODO: Generate heatmap graphs


def heatmap(dirs: List[str], prefix: str = "dev-", output_file: str = None) -> None:
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
        name, *weight, task, _ = os.path.split(d)[-1].split("-")
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
                assert labels == model_labels_value, "Inconsistent labels"

            differences = []

            for i, (pred, truth) in enumerate(zip(labels, model_predictions_values)):

                if pred != truth:

                    differences.append(abs(model_probabilities_value[i][pred] - model_probabilities_value[i][truth]))

                else:
                    differences.append(np.NaN)
            result[weight] = np.asarray(differences)
            print(f"{weight}: {np.count_nonzero(~np.isnan(result[weight])) / len(result[weight]):.2f}")
    dataframe = pd.DataFrame(result)
    dataframe = dataframe.iloc[dataframe.isnull().sum(1).sort_values(ascending=False).index]
    dataframe = dataframe.transpose()

    print(dataframe.head(5))

    ax = sns.heatmap(dataframe, cmap="YlGnBu", mask=dataframe.isnull(), center=0.5, xticklabels=False)
    plt.show()


if __name__ == "__main__":
    heatmap(glob.glob("./output/*physicaliqa-pred"), output_file="physicaliqa.svg")
