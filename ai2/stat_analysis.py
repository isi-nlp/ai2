import logging
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd

from immutablecollections import immutabledict
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from ai2.stats_tests import (
    fishers_test,
    binomial_difference_of_proportions_test,
    mcnemar,
    mcnemar_exact_ct,
    mcnemar_exact_upper_p,
    mcnemar_exact_lower_p,
    mcnemar_min_overlap,
    mcnemar_max_overlap,
)

_logger = logging.getLogger(__name__)

UNPAIRED_TESTS = immutabledict(
    {
        "fisher": fishers_test,
        "prop": binomial_difference_of_proportions_test,
        "mc-min-overlap-upper-p": mcnemar_min_overlap,
        "mc-max-overlap-lower-p": mcnemar_max_overlap,
        "mc-b-upper-p": mcnemar_exact_upper_p,
        "mc-b-lower-p": mcnemar_exact_lower_p,
    }
)
EPSILON = 0.01
ACCURACY_MISMATCH_ERROR = (
    f"Computed accuracy for model %d (%s) is %3f but accuracy from results.txt is %3f "
    f"which differs by more than %3f.",
)


def stat_analysis_entrypoint(params: Parameters):
    comparisons_to_make_path = params.existing_file("comparisons_to_make")
    save_accuracies_to = params.creatable_file("save_accuracies_to")
    save_overlap_seqs_to = params.creatable_file("save_overlap_seqs_to")
    save_comparison_results_to = params.creatable_file("save_comparison_results_to")
    log_every_n_steps = params.positive_integer("log_every_n_steps", default=10)

    comparisons_to_make = pd.read_json(
        comparisons_to_make_path, orient="records", lines=True
    )
    n_comparisons = len(comparisons_to_make)
    comparisons = []
    agreement_seqs = {}
    task_to_model_to_accuracy = {}
    _logger.info("Doing %d comparisons...", n_comparisons)
    for idx, comparison_to_make in comparisons_to_make.iterrows():
        task_name = comparison_to_make["task"]
        model1_name = name_model(comparison_to_make["model1_combination"])
        model2_name = name_model(comparison_to_make["model2_combination"])
        model1_accuracy = get_accuracy_from_results(
            Path(comparison_to_make["model1_results"])
        )
        model2_accuracy = get_accuracy_from_results(
            Path(comparison_to_make["model2_results"])
        )

        # Collect model accuracies for summary
        model_to_accuracy = task_to_model_to_accuracy.setdefault(task_name, {})
        if model_to_accuracy.get(model1_name, model1_accuracy) != model1_accuracy:
            raise RuntimeError(
                f"Model {model1_name} has two listed accuracies: {model1_accuracy} and (prior) {model_to_accuracy[model1_name]}."
            )
        model_to_accuracy[model1_name] = model1_accuracy

        if model_to_accuracy.get(model2_name, model2_accuracy) != model2_accuracy:
            raise RuntimeError(
                f"Model {model2_name} has two listed accuracies: {model2_accuracy} and (prior) {model_to_accuracy[model2_name]}."
            )
        model_to_accuracy[model2_name] = model2_accuracy

        # To keep things organized, reorder the comparison if model1 is worse than model2
        # This makes it so that when we sort our comparisons at the end, we end up sorting (for each task)
        # such that all the (rank 1 vs. rank j), comparisons come first (j > 1),
        # then the (rank 2 vs. rank j), comparisons (j > 2),
        # etc.
        if model1_accuracy < model2_accuracy:
            logging.info("Model 1, %s, worse than model 2, %s; swapping...")
            model1_name, model2_name = model2_name, model1_name
            model1_accuracy, model2_accuracy = model2_accuracy, model1_accuracy
            for key1 in comparisons_to_make:
                if "model1" in key1:
                    key2 = key1.replace("model1", "model2")
                    comparison_to_make[key1], comparison_to_make[key2] = (
                        comparison_to_make[key2],
                        comparison_to_make[key1],
                    )
            logging.info(
                "New model 1 is %s and new model 2 is %s.", model1_name, model2_name
            )

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
        model1_correct = model1_predicted_labels == gold_labels
        model2_correct = model2_predicted_labels == gold_labels
        computed_model1_accuracy = model1_correct.mean()
        computed_model2_accuracy = model2_correct.mean()
        if abs(model1_correct.mean() - model1_accuracy) > EPSILON:
            _logger.error(
                ACCURACY_MISMATCH_ERROR,
                1,
                model1_name,
                model1_accuracy,
                computed_model1_accuracy,
                EPSILON,
            )
        if abs(model2_correct.mean() - model2_accuracy) > EPSILON:
            _logger.error(
                ACCURACY_MISMATCH_ERROR,
                2,
                model2_name,
                model2_accuracy,
                computed_model2_accuracy,
                EPSILON,
            )
        contingency_table = pd.crosstab(
            model1_correct.rename("model1"), model2_correct.rename("model2")
        )
        _logger.debug("model1 = %s, model2 = %s", model1_name, model2_name)
        _logger.debug(
            "first entries of model1_predicted_labels = \n%s",
            model1_predicted_labels.head(),
        )
        _logger.debug(
            "first entries of model2_predicted_labels = \n%s",
            model2_predicted_labels.head(),
        )
        _logger.debug("-----")
        _logger.debug("first entries of model1_correct = \n%s", model1_correct.head())
        _logger.debug("first entries of model2_correct = \n%s", model2_correct.head())
        _logger.debug("contingency table = \n%s", contingency_table)
        agreement_seq: pd.Series = model1_correct == model2_correct
        agreement_seqs[
            f"{model1_name} with {model2_name} ({task_name})"
        ] = agreement_seq
        percent_overlap = agreement_seq.mean()

        # Calculate tests
        test_set_size = len(gold_labels)
        stats = {
            "mcnemar": mcnemar(
                test_set_size=test_set_size,
                model1_accuracy=model1_accuracy,
                model2_accuracy=model2_accuracy,
                percent_overlap=percent_overlap,
            ),
            "mcnemar-exact": mcnemar_exact_ct(
                n_disagreements=int(test_set_size - agreement_seq.sum()),
                n_only_model1_correct=int((model1_correct & ~model2_correct).sum()),
                n_only_model2_correct=int((model2_correct & ~model1_correct).sum()),
            ),
        }
        stats.update(
            {
                test_name: unpaired_test(
                    test_set_size=test_set_size,
                    model1_accuracy=model1_accuracy,
                    model2_accuracy=model2_accuracy,
                )
                for test_name, unpaired_test in UNPAIRED_TESTS.items()
            }
        )

        # Start with the basics, then add stats, then extra model details.
        comparison = {
            "task": task_name,
            "num test": len(gold_labels),
            "Model A": model1_name,
            "Model B": model2_name,
            "Model A Accuracy": model1_accuracy,
            "Model B Accuracy": model2_accuracy,
            "% overlap": percent_overlap,
        }
        # add tests
        for test_name, (test_statistic, p_value) in stats.items():
            comparison.update(
                {f"{test_name} stat": test_statistic, f"{test_name} p": p_value}
            )
        comparison.update(
            {
                "both right": contingency_table.loc[True, True],
                "A right/B wrong": contingency_table.loc[True, False],
                "B right/A wrong": contingency_table.loc[False, True],
                "both wrong": contingency_table.loc[False, False],
            }
        )
        comparison.update(
            {
                f"Model A {name}": value
                for name, value in comparison_to_make["model1_combination"]
                if name != "task"
            }
        )
        comparison.update(
            {
                f"Model B {name}": value
                for name, value in comparison_to_make["model2_combination"]
                if name != "task"
            }
        )
        model1_slice_seed, model1_slice_size = comparison["Model A slice"].split("_")
        model2_slice_seed, model2_slice_size = comparison["Model B slice"].split("_")
        comparison.update(
            {
                f"Model A slice size (%)": model1_slice_size,
                f"Model A slice seed": model1_slice_seed,
                f"Model B slice size (%)": model2_slice_size,
                f"Model B slice seed": model2_slice_seed,
            }
        )
        comparisons.append(comparison)

        if (idx + 1) % log_every_n_steps == 0:
            _logger.info("Ran %d / %d comparisons.", idx + 1, n_comparisons)

    _logger.info("Done comparing.")

    _logger.info("Collecting and saving results...")
    accuracy_df = pd.DataFrame(
        [
            {"task": task_name, "model": model_name, "accuracy": accuracy}
            for task_name, model_to_accuracy in task_to_model_to_accuracy.items()
            for model_name, accuracy in model_to_accuracy.items()
        ]
    )
    # Add a rank column where 1 is the highest rank and 2 is the lowest
    # Thanks to https://stackoverflow.com/a/33899937 for the .rank() solution.
    accuracy_df["rank"] = (
        accuracy_df.groupby("task", as_index=False)["accuracy"]
        .rank("first", ascending=False)
        .astype(int)
    )
    accuracy_df = accuracy_df.sort_values(
        by=["task", "accuracy"], ascending=[True, False]
    )
    accuracy_df.to_csv(save_accuracies_to, index=False)

    pd.DataFrame(agreement_seqs).to_csv(save_overlap_seqs_to, index=False)

    # Create the comparisons DF.
    # Make sure to merge in the model ranks and reorder them so they're together with the model names.
    comparisons_df = pd.DataFrame(comparisons)
    comparisons_df = pd.merge(
        comparisons_df,
        accuracy_df[["task", "model", "rank"]].rename(
            columns={"model": "Model A", "rank": "Model A Rank"}
        ),
        on=["task", "Model A"],
        how="left",
    )
    comparisons_df = pd.merge(
        comparisons_df,
        accuracy_df[["task", "model", "rank"]].rename(
            columns={"model": "Model B", "rank": "Model B Rank"}
        ),
        on=["task", "Model B"],
        how="left",
    )
    model_a_rank = comparisons_df.pop("Model A Rank")
    model_b_rank = comparisons_df.pop("Model B Rank")
    comparisons_df.insert(3, "Model A Rank", model_a_rank)
    comparisons_df.insert(5, "Model B Rank", model_b_rank)
    comparisons_df = comparisons_df.sort_values(
        by=["task", "Model A Rank", "Model B Rank"], ascending=True
    )
    comparisons_df.to_csv(save_comparison_results_to, index=False)

    _logger.info("Saved collected results.")


def get_accuracy_from_results(results_file: Path) -> float:
    """
    Read the accuracy from a CSV file called results.txt, as produced by the ai2.evaluate code.
    """
    results = pd.read_csv(
        results_file,
        names=["name", "_explainer", "accuracy", "_average", "_lower", "_upper"],
    )
    return float(results.iloc[-1]["accuracy"])


def name_model(combination: List[Tuple[str, Any]]) -> str:
    return ",".join(
        "=".join(str(x) for x in option_pair)
        for option_pair in combination
        if "task" not in option_pair[0]
    )


if __name__ == "__main__":
    parameters_only_entry_point(stat_analysis_entrypoint)
