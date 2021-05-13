import logging
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd

from immutablecollections import immutabledict
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from ai2.stats_tests import fishers_test, binomial_difference_of_proportions_test, mcnemar, mcnemar_worst_case, mcnemar_best_case


_logger = logging.getLogger(__name__)

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
    log_every_n_steps = params.optional_positive_integer("log_every_n_steps", default=10)

    comparisons_to_make = pd.read_json(comparisons_to_make_path, orient="records", lines=True)
    n_comparisons = len(comparisons_to_make)
    comparisons = []
    agreement_seqs = {}
    task_to_model_to_accuracy = {}
    _logger.info("Doing %d comparisons...", n_comparisons)
    for idx, comparison_to_make in comparisons_to_make.iterrows():
        task_name = comparison_to_make["task"]
        model1_name = name_model(comparison_to_make["model1_combination"])
        model2_name = name_model(comparison_to_make["model2_combination"])
        model1_accuracy = get_accuracy_from_results(Path(comparison_to_make["model1_results"]))
        model2_accuracy = get_accuracy_from_results(Path(comparison_to_make["model2_results"]))

        # Collect model accuracies for summary
        model_to_accuracy = task_to_model_to_accuracy.setdefault(task_name, {})
        if model_to_accuracy.get(model1_name, model1_accuracy) != model1_accuracy:
            raise RuntimeError(f"Model {model1_name} has two listed accuracies: {model1_accuracy} and (prior) {model_to_accuracy[model1_name]}.")
        model_to_accuracy[model1_name] = model1_accuracy

        if model_to_accuracy.get(model2_name, model2_accuracy) != model2_accuracy:
            raise RuntimeError(f"Model {model2_name} has two listed accuracies: {model2_accuracy} and (prior) {model_to_accuracy[model2_name]}.")
        model_to_accuracy[model2_name] = model2_accuracy

        # Read in raw predictions
        model1_predicted_labels: pd.Series = pd.read_csv(
            comparison_to_make["model1_predicted_labels"], names=["label"]
        )["label"]
        model2_predicted_labels: pd.Series = pd.read_csv(
            comparison_to_make["model2_predicted_labels"], names=["label"]
        )["label"]
        gold_labels: pd.Series = pd.read_csv(
            comparison_to_make["gold_labels"], names=["label"]
        )["label"]

        # Calculate agreement
        agreement_seq: pd.Series = (model1_predicted_labels == gold_labels) == (model2_predicted_labels == gold_labels)
        agreement_seqs[f"{model1_name} with {model2_name} ({task_name})"] = agreement_seq
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
            "task": task_name,
            "num test": task_name,
            "Model A": model1_name,
            "Model B": model2_name,
            "Model A Accuracy": model1_accuracy,
            "Model B Accuracy": model2_accuracy,
            "% agree": percent_agreement,
        }
        # add tests
        for test_name, (test_statistic, p_value) in stats.items():
            comparison.update({
                f"{test_name} stat": test_statistic,
                f"{test_name} p": p_value,
            })
        comparison.update({
            f"Model A {name}": value for name, value in comparison_to_make["model1_combination"]
        })
        comparison.update({
            f"Model B {name}": value for name, value in comparison_to_make["model2_combination"]
        })
        comparisons.append(comparison)

        if (idx + 1) % log_every_n_steps == 0:
            _logger.info("Ran %d / %d comparisons.", idx + 1, n_comparisons)

    _logger.info("Done comparing.")

    _logger.info("Collecting and saving results...")
    accuracy_df = pd.DataFrame(
        [
            {
                "task": task_name,
                "model": model_name,
                "accuracy": accuracy,
            }
            for task_name, model_to_accuracy in task_to_model_to_accuracy.items()
            for model_name, accuracy in model_to_accuracy.items()
        ]
    )
    # Add a rank column where 1 is the highest rank and 2 is the lowest
    accuracy_df["Rank"] = accuracy_df.groupby("task", as_index=False).transform(
        lambda leaderboard_results: leaderboard_results.sort_values(
            ascending=False
        ).reset_index().index + 1
    )
    accuracy_df.sort_values(by=["task", "accuracy"])
    accuracy_df.to_csv(save_accuracies_to, index=False)
    comparisons_df = pd.DataFrame(comparisons)
    comparisons_df = pd.merge(
        comparisons_df,
        accuracy_df[["task", "model", "rank"]].rename(columns={"model": "Model A", "rank": "Model A Rank"}),
        on=["task", "Model A"],
        how="left",
    )
    comparisons_df = pd.merge(
        comparisons_df,
        accuracy_df[["task", "model", "rank"]].rename(columns={"model": "Model B", "rank": "Model B Rank"}),
        on=["task", "Model B"],
        how="left",
    )
    comparisons_df.to_csv(save_comparison_results_to, index=False)
    pd.DataFrame(agreement_seqs).to_csv(save_agreement_seqs_to, index=False)
    _logger.info("Saved collected results.")


def get_accuracy_from_results(results_file: Path) -> float:
    """
    Read the accuracy from a CSV file called results.txt, as produced by the ai2.evaluate code.
    """
    results = pd.read_csv(results_file, names=["name", "_explainer", "accuracy", "_average", "_lower", "_upper"])
    return float(results.loc[0, "accuracy"])


def name_model(combination: List[Tuple[str, Any]]) -> str:
    return ','.join(
        '='.join(str(x) for x in option_pair)
        for option_pair in combination
        if "task" not in option_pair[0]
    )


if __name__ == "__main__":
    parameters_only_entry_point(stat_analysis_entrypoint)
