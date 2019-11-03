#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-25 10:33:57
# @Author  : Chenghao Mou (chengham@isi.edu)
# @Link    : https://github.com/ChenghaoMou/ai2

# pylint: disable=unused-wildcard-import
import json
import numpy as np
import pandas as pd
from typing import *
from pathlib import Path
from loguru import logger


def open_image_relationships(
        classes: Union[str, Path],
        relationships: Union[str, Path],
        input_files: List[Union[str, Path]],
        output_files: List[Union[str, Path]]) -> None:

    classes = Path(classes)
    input_files = [Path(file) for file in input_files]
    output_files = [Path(file) for file in output_files]
    class_lookup = {code: name for code, name in pd.read_csv(classes, header=None).values}

    relationship_lookup = {code: name for code, name in pd.read_csv(relationships, header=None).values}
    logger.info(f"{len(class_lookup)} Objects")
    for input_file, output_file in zip(input_files, output_files):

        df = pd.read_csv(input_file)
        df['LabelName1'] = df['LabelName1'].apply(lambda x: class_lookup[x])
        df['LabelName2'] = df['LabelName2'].apply(lambda x: class_lookup[x])

        df = df.drop_duplicates(subset=["LabelName1", "LabelName2", "RelationshipLabel"], keep='first', inplace=False)
        logger.info(f"{df.shape[0]} triples")
        with open(output_file, "w") as output:

            for (entity1, entity2, relationship) in df[["LabelName1", "LabelName2", "RelationshipLabel"]].values:
                output.write(f"{entity1} {relationship_lookup[relationship]} {entity2}.\n")


def visualgenome_relationships(
        relationships: Union[str, Path],
        alias: Union[str, Path],
        output_dir: Union[str, Path]) -> None:

    relationships, alias, output_dir = map(Path, [relationships, alias, output_dir])
    Path.mkdir(output_dir, exist_ok=True)
    with open(relationships, 'r') as f, open(output_dir / "train.txt", "w") as train, open(output_dir / "valid.txt", "w") as valid:
        data = json.loads(f.read())
        cnt = 0
        for subject in data:
            for rel in data[subject]:
                for obj in data[subject][rel]:
                    if cnt % 100 == 0:
                        valid.write(f"{subject} {rel} {obj}\n")
                    else:
                        train.write(f"{subject} {rel} {obj}\n")
                    cnt += 1


if __name__ == "__main__":
    # open_image_relationships("visual_data/class-descriptions.csv",
    #                          "visual_data/challenge-2018-relationships-description.csv",
    #                          ["visual_data/challenge-2018-train-vrd.csv",
    #                           "visual_data/validation-annotations-vrd.csv"],
    #                          ["visual_data/open_image_train.txt", "visual_data/open_image_dev.txt"])

    visualgenome_relationships("visual_data/subjects_all.json",
                               "visual_data/relationship_alias.txt",
                               "visual_data/genome")
