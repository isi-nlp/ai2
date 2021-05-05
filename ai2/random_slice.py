"""
A script for slicing a dataset using a given seed and fraction.
"""
import logging

import numpy as np
import pandas as pd

from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point
from vistautils.range import Range


_log = logging.getLogger(__name__)


def random_slice_entry_point(params: Parameters) -> None:
    """
    Parameters:
        input (existing file):
            The input file. We infer the format from the suffix (e.g.

        output (creatable file):
            The output file. We infer the desire format from the suffix.

        random_seed (integer):
            The random seed to use when slicing.

        fraction (decimal):
            How much of the data to sample. Must be greater than 0 and at most 1. A value of 0.5 for example means
            "sample 50% of the available data."
    """
    input_ = params.existing_file("input")
    output = params.creatable_file("output")
    random_seed = params.integer("random_seed", valid_range=Range.at_least(0))
    fraction = params.floating_point("fraction", valid_range=Range.open_closed(0., 1.))
    np.random.seed(random_seed)

    df: pd.DataFrame
    if input_.suffix == ".jsonl":
        df = pd.read_json(str(input_), lines=True)
    elif input_.suffix in {".csv", ".lst"}:
        df = pd.read_csv(str(input_))
    else:
        _log.warning("Don't know how to handle input with suffix %s; attempting to load as CSV...", input_.suffix)
        df = pd.read_csv(str(input_))

    n_rows = len(df)
    _log.info("Got input of size %d", n_rows)

    permuted_indices = np.random.permutation(df.index)
    sampled_indices = permuted_indices[:int(n_rows * fraction)]
    sampled_df = df[sampled_indices]
    _log.info("Producing slice of size %d", len(sampled_df))

    if output.suffix == ".jsonl":
        sampled_df.to_json(str(output), lines=True, index=False)
    elif output.suffix in {".lst"}:
        sampled_df.to_csv(str(output), header=False, index=False)
    else:
        _log.warning("Don't know how to save output with suffix %s; saving as CSV with header...", input_.suffix)
        sampled_df.to_csv(str(output), header=True, index=False)


if __name__ == '__main__':
    parameters_only_entry_point(random_slice_entry_point)
