from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd

from immutablecollections import immutabledict
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from ai2.stats_tests import fishers_test, binomial_difference_of_proportions_test, mcnemar, mcnemar_worst_case, mcnemar_best_case


UNPAIRED_TESTS = immutabledict({
    "fisher": fishers_test,
    "prop": binomial_difference_of_proportions_test,
    "mc-worst": mcnemar_worst_case,
    "mc-best": mcnemar_best_case,
})


def stat_analysis_entrypoint(params: Parameters):
    comparisons_to_make_path = params.existing_file("comparisons_to_make")
    save_accuracies_to = params.creatable_file("save_accuracies_to")
    save_agreement_seqs_to = params.creatable_file("save_agreement_seqs_to")
    save_comparison_results_to = params.creatable_file("save_comparison_results_to")

    comparisons_to_make = pd.read_json(comparisons_to_make_path, orient="records", lines=True)
    comparisons = []
    agreement_seqs = []
    model_accuracies = {}
    for _, comparison_to_make in comparisons_to_make.iterrows():
        model1_name = name_model(comparison_to_make["model1_combination"])
        model2_name = name_model(comparison_to_make["model2_combination"])
        model1_accuracy = get_accuracy_from_results(Path(comparison_to_make["model1_accuracy"]))
        model2_accuracy = get_accuracy_from_results(Path(comparison_to_make["model2_accuracy"]))

        # Collect model accuracies for summary
        if model_accuracies.get(model1_name, model1_accuracy) != model1_accuracy:
            raise RuntimeError(f"Model {model1_name} has two listed accuracies: {model1_accuracy} and (prior) {model_accuracies[model1_name]}.")
        model_accuracies[model1_name] = model1_accuracy

        if model_accuracies.get(model2_name, model2_accuracy) != model2_accuracy:
            raise RuntimeError(f"Model {model2_name} has two listed accuracies: {model2_accuracy} and (prior) {model_accuracies[model2_name]}.")
        model_accuracies[model2_name] = model2_accuracy

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

        # Calculate tests
        test_set_size = len(gold_labels)
        stats = {
            "mcnemar": mcnemar(
                test_set_size=test_set_size,
                model1_accuracy=model1_accuracy,
                model2_accuracy=model2_accuracy,
                fractional_agreement=percent_agreement,
            )
        }
        stats.update({
            test_name: unpaired_test(
                test_set_size=test_set_size,
                model1_accuracy=model1_accuracy,
                model2_accuracy=model2_accuracy,
            )
            for test_name, unpaired_test in UNPAIRED_TESTS.items()
        })

        # Start with the basics, then add stats, then extra model details.
        comparison = {
            "model1_name": model1_name,
            "model2_name": model1_name,
            "model1_accuracy": model1_accuracy,
            "model2_accuracy": model2_accuracy,
            "percent_agreement": percent_agreement,
        }
        # add tests
        for test_name, (test_statistic, p_value) in stats.items():
            comparisons.update({
                f"{test_name} stat": test_statistic,
                f"{test_name} p": p_value,
            })
        comparison.update({
            f"model1 {name}": value for name, value in comparisons_to_make["model1_combination"]
        })
        comparison.update({
            f"model2 {name}": value for name, value in comparisons_to_make["model2_combination"]
        })
        comparisons.append(comparison)

    pd.DataFrame(
        [
            {
                "model_name": model_name,
                "accuracy": accuracy,
            }
            for model_name, accuracy in model_accuracies.items()
        ]
    ).to_csv(save_accuracies_to)
    pd.DataFrame(comparisons).to_csv(save_comparison_results_to)
    pd.DataFrame(agreement_seqs).to_csv(save_agreement_seqs_to)


def get_accuracy_from_results(results_file: Path) -> float:
    """
    Read the accuracy from a CSV file called results.txt, as produced by the ai2.evaluate code.
    """
    results = pd.read_csv(results_file, header=["name", "_explainer", "accuracy", "_average", "_lower", "_upper"])
    return float(results.loc[0, "accuracy"])


def name_model(combination: List[Tuple[str, Any]]) -> str:
    return ','.join(
        '='.join(str(x) for x in option_pair)
        for option_pair in combination
        if "task" not in option_pair[0]
    )


if __name__ == "__main__":
    parameters_only_entry_point(stat_analysis_entrypoint)
