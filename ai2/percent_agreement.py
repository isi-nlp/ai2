from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd

from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point


def percent_agreement_entrypoint(params: Parameters):
    comparisons_to_make_path = params.existing_file("comparisons_to_make")
    save_agreement_seqs_to = params.creatable_file("save_agreement_seqs_to")
    save_comparison_results_to = params.creatable_file("save_comparison_results_to")

    comparisons_to_make = pd.read_json(comparisons_to_make_path, orient="records", lines=True)
    comparisons = []
    agreement_seqs = []
    for comparison_to_make in comparisons_to_make:
        model1_name = name_model(comparison_to_make["model1_combination"])
        model2_name = name_model(comparison_to_make["model2_combination"])
        model1_accuracy = float(Path(comparison_to_make["model1_accuracy"]).read_text())
        model2_accuracy = float(Path(comparison_to_make["model2_accuracy"]).read_text())

        # Read in raw predictions
        model1_predicted_labels: pd.Series = pd.read_csv(
            comparisons_to_make["model1_predicted_labels"], names=["label"]
        )["label"]
        model2_predicted_labels: pd.Series = pd.read_csv(
            comparisons_to_make["model1_predicted_labels"], names=["label"]
        )["label"]
        gold_labels: pd.Series = pd.read_csv(
            comparison_to_make["gold_labels"], names=["label"]
        )["label"]

        # Calculate agreement
        agreement_seq: pd.Series = (model1_predicted_labels == gold_labels) == (model2_predicted_labels == gold_labels)
        agreement_seqs.append({
            f"{model1_name} with {model2_name}": agreement_seq,
        })
        percent_agreement = agreement_seq.mean()

        comparison = {
            "model1_name": model1_name,
            "model2_name": model1_name,
            "model1_accuracy": model1_accuracy,
            "model2_accuracy": model2_accuracy,
            "percent_agreement": percent_agreement,
        }
        comparison.update({
            f"model1 {name}": value for name, value in comparisons_to_make["model1_combination"]
        })
        comparison.update({
            f"model2 {name}": value for name, value in comparisons_to_make["model2_combination"]
        })
        comparisons.append(comparison)

    pd.DataFrame(comparisons).to_csv(save_comparison_results_to)
    pd.DataFrame(agreement_seqs).to_csv(save_agreement_seqs_to)


def name_model(combination: List[Tuple[str, Any]]) -> str:
    return ','.join(
        '='.join(str(x) for x in option_pair)
        for option_pair in combination
        if "task" not in option_pair[0]
    )


if __name__ == "__main__":
    parameters_only_entry_point(percent_agreement_entrypoint)
