"""
Train two slightly different RoBERTa models and compare them on
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from more_itertools import only
import pandas as pd

from immutablecollections import immutableset
from vistautils.parameters import Parameters, YAMLParametersLoader
from vistautils.parameters_only_entrypoint import parameters_only_entry_point
from pegasus_wrapper import (
    directory_for,
    initialize_vista_pegasus_wrapper,
    run_python_on_parameters,
    limit_jobs_for_category,
    write_workflow_description,
)
from pegasus_wrapper.resource_request import ResourceRequest, SlurmResourceRequest
from pegasus_wrapper.locator import Locator
from pegasus_wrapper.artifact import ValueArtifact

import ai2.random_slice as random_slice_script
import ai2.eval as eval_script
import ai2.stat_analysis as stat_analysis_script
from ai2.pegasus import override_generality, override_matches


TIME_LIMIT_HOURS_NOT_ALPHANLI = 12  # Time limit in hours for tasks other than AlphaNLI
MINUTES_PER_HOUR = 60

# Default limit on the number of jobs that will run on MICS at once
DEFAULT_MAX_JOBS_ON_MICS = 2

# Represents a parameter combination as a list of (parameter_name, value) tuples.
ParameterCombination = List[Tuple[str, Any]]

logger = logging.getLogger(__name__)


def run_random_slice(
    slice_locator: Locator,
    *,
    input: ValueArtifact,
    output: Path,
    random_seed: int,
    fraction: float,
) -> ValueArtifact:
    slice_job = run_python_on_parameters(
        slice_locator,
        random_slice_script,
        Parameters.from_mapping(
            {
                "input": input.value,
                "output": output,
                "random_seed": random_seed,
                "fraction": fraction,
            }
        ),
        depends_on=[input],
        resource_request=SlurmResourceRequest(job_time_in_minutes=120),
    )
    return ValueArtifact(value=output, depends_on=immutableset([slice_job]))


def run_random_slice_for_x_y_pair(
    slice_locator: Locator,
    *,
    input_x: ValueArtifact,
    input_y: ValueArtifact,
    output_x: Path,
    output_y: Path,
    random_seed: int,
    fraction: float,
) -> Tuple[ValueArtifact, ValueArtifact]:
    sliced_x_artifact = run_random_slice(
        slice_locator / "x",
        input=input_x,
        output=output_x,
        random_seed=random_seed,
        fraction=fraction,
    )
    sliced_y_artifact = run_random_slice(
        slice_locator / "y",
        input=input_y,
        output=output_y,
        random_seed=random_seed,
        fraction=fraction,
    )
    return sliced_x_artifact, sliced_y_artifact


def compare_models_entrypoint(params: Parameters):
    initialize_vista_pegasus_wrapper(params)

    experiment_root = params.creatable_directory("experiment_root")
    project_root = params.existing_directory("project_root")
    params_root = project_root / "parameters"
    slice_options = params.arbitrary_list("slice_options")
    parameter_options = params.namespace("parameter_options").as_nested_dicts()
    all_tasks = immutableset(parameter_options["task"])

    max_jobs_on_mics = params.integer(
        "max_jobs_on_mics", default=DEFAULT_MAX_JOBS_ON_MICS
    )

    # Set up jobs to slice the datasets appropriately
    task_to_parameters = {
        task: YAMLParametersLoader().load(params_root / "task" / f"{task}.params")
        for task in iter(all_tasks)
    }

    def name_slice(task: str, slice_option: Dict[str, Any]) -> Tuple[str, ...]:
        return "slice", task, f"{slice_option['seed']}_{slice_option['percent']}"

    slices = {
        task: [
            (
                name_slice(task, slice_option),
                run_random_slice_for_x_y_pair(
                    Locator(name_slice(task, slice_option)),
                    input_x=ValueArtifact.preexisting(
                        task_to_parameters[task].existing_file("train_x")
                    ),
                    input_y=ValueArtifact.preexisting(
                        task_to_parameters[task].existing_file("train_y")
                    ),
                    output_x=(
                        experiment_root
                        / "slices"
                        / task
                        / f"seed{slice_option['seed']}_pct{slice_option['percent']}"
                        / "train.jsonl"
                    ),
                    output_y=(
                        experiment_root
                        / "slices"
                        / task
                        / f"seed{slice_option['seed']}_pct{slice_option['percent']}"
                        / "train-labels.lst"
                    ),
                    random_seed=slice_option["seed"],
                    fraction=slice_option["percent"] / 100.0,
                ),
            )
            for slice_option in slice_options
        ]
        for task in iter(all_tasks)
    }

    # Compute all possible combinations of the parameters
    parameter_combinations: List[ParameterCombination] = [[]]
    for parameter_name, options in parameter_options.items():
        new_combinations = []
        for combination in parameter_combinations:
            for option in options:
                new_combination = combination + [(parameter_name, option)]
                new_combinations.append(new_combination)
        parameter_combinations = new_combinations

    # Process combination-specific overrides
    training_overrides = sorted(
        list(
            params.namespace_or_empty("training_overrides").as_nested_dicts().values()
        ),
        key=lambda override_: override_generality(override_, parameter_options),
    )

    # Schedule jobs for each parameter combination:
    # both a train job (output under 'models')
    # and an eval job (output under 'eval')
    model_outputs_locator = Locator(("models",))
    evaluation_artifacts = []
    for idx, combination in enumerate(parameter_combinations):
        task: str = only(
            option for parameter, option in combination if parameter == "task"
        )
        options: Tuple[str, ...] = tuple(
            str(option) if option != "" else "_default" for _, option in combination
        )
        for tuple_slice_name, (slice_x_artifact, slice_y_artifact) in slices[task]:
            string_slice_name = "_".join(tuple_slice_name)
            train_locator = model_outputs_locator / Locator(options) / string_slice_name

            # Set up common job parameters
            eval_job_params = Parameters.from_key_value_pairs(
                [("model", params.namespace("model"))]
            ).unify(params.namespace("train"))

            # Read in combination-specific parameters
            eval_job_params = eval_job_params.unify(
                Parameters.from_key_value_pairs(combination, namespace_separator=None)
            )
            for parameter, option in combination:
                if option != "":
                    parameter_directory = params_root / parameter
                    if parameter_directory.exists():
                        option_params: Parameters = YAMLParametersLoader().load(
                            parameter_directory / f"{option}.params"
                        )
                        eval_job_params = eval_job_params.unify(option_params)

            # Because the job parameters tend to indirectly include root.params, which includes a
            # default partition, we need to override the partition setting to reflect our input
            # parameters.
            eval_job_params = eval_job_params.unify(
                {"partition": params.string("partition")}
            )

            # Process overrides
            for override in training_overrides:
                if override_matches(override, dict(combination)):
                    eval_job_params = eval_job_params.unify(
                        {
                            parameter_option: value
                            for parameter_option, value in override.items()
                            if parameter_option != "parameter_options"
                        }
                    )

            # Messy parameters input. This shouldn't matter to ResourceRequest, though. Maybe clean up
            # later.
            resource_request_params = params.unify(eval_job_params)
            resource_request = ResourceRequest.from_parameters(
                resource_request_params
                # Run training on MICS.
                # jac: temporary, while supplies lost
            ).unify(
                SlurmResourceRequest(
                    partition="mics",
                    job_time_in_minutes=resource_request_params.positive_integer(
                        "job_time_in_minutes"
                    ),
                )
            )

            # Set common parameters and schedule the job.
            options_name = "_".join(
                "=".join(str(x) for x in option_pair) for option_pair in combination
            )
            # Grab the latest checkpoint for this configuration and evaluate that.
            save_path = experiment_root / f"{options_name}_{string_slice_name}"
            checkpoints_root = save_path / "checkpoints"
            # assert checkpoints_root.exists()
            checkpoint_paths_from_latest_to_earliest = sorted(
                checkpoints_root.glob("*.ckpt"), reverse=True, key=lambda path: path.stat().st_mtime,
            )
            latest_checkpoint_path = checkpoint_paths_from_latest_to_earliest[0] if checkpoint_paths_from_latest_to_earliest else checkpoints_root / "dummy.ckpt"
            logger.info(
                "Using latest checkpoint for model %s. Latest checkpoint is: `%s`",
                options_name,
                latest_checkpoint_path,
            )
            logger.debug(
                "For model %s, available checkpoints were: %s",
                options_name,
                checkpoint_paths_from_latest_to_earliest,
            )
            eval_job_params = eval_job_params.unify(
                {
                    "checkpoint_path": latest_checkpoint_path,
                    "results_path": save_path,
                    "with_true_label": True,
                    # use a fixed random seed for evaluation
                    "random_seed": 0,
                }
            )
            eval_job = run_python_on_parameters(
                train_locator / "evaluate",
                eval_script,
                eval_job_params,
                depends_on=[slice_x_artifact, slice_y_artifact],
                resource_request=resource_request,
            )

            evaluation_artifacts.append(
                (
                    combination + [("slice", "_".join(tuple_slice_name[2:]))],
                    task,
                    ValueArtifact(
                        value=save_path / "results.txt",
                        depends_on=immutableset([eval_job]),
                    ),
                    ValueArtifact(
                        value=save_path / "predictions.lst",
                        depends_on=immutableset([eval_job]),
                    ),
                )
            )

    # Calculate summary statistics for comparing the models
    stat_analysis_locator = Locator(("stat_analysis",))
    comparisons_to_make = pd.DataFrame(
        [
            {
                "task": task1,
                "model1_combination": model1_combination,
                "model2_combination": model2_combination,
                "model1_results": str(model1_results_artifact.value),
                "model2_results": str(model2_results_artifact.value),
                "model1_predicted_labels": str(model1_predictions_artifact.value),
                "model2_predicted_labels": str(model2_predictions_artifact.value),
                "gold_labels": str(task_to_parameters[task1].existing_file("val_y")),
            }
            for idx, (
                model1_combination,
                task1,
                model1_results_artifact,
                model1_predictions_artifact,
            ) in enumerate(evaluation_artifacts)
            for model2_combination, task2, model2_results_artifact, model2_predictions_artifact in evaluation_artifacts[
                idx + 1 :
            ]
            if task1 == task2
        ]
    )
    file_of_comparisons_to_make = (
        directory_for(stat_analysis_locator) / "comparisons.jsonl"
    )
    comparisons_to_make.to_json(
        file_of_comparisons_to_make, orient="records", lines=True
    )

    stat_analysis_parameters = params.unify(
        {
            "comparisons_to_make": file_of_comparisons_to_make,
            "save_accuracies_to": experiment_root / "accuracies.csv",
            "save_overlap_seqs_to": experiment_root / "overlap_data.csv",
            "save_comparison_results_to": experiment_root / "summary.csv",
        }
    )

    accuracy_artifacts = tuple(
        accuracy_artifact for _, _, accuracy_artifact, _ in evaluation_artifacts
    )
    prediction_artifacts = tuple(
        prediction_artifact for _, _, _, prediction_artifact in evaluation_artifacts
    )
    run_python_on_parameters(
        stat_analysis_locator,
        stat_analysis_script,
        stat_analysis_parameters,
        depends_on=accuracy_artifacts + prediction_artifacts,
        resource_request=SlurmResourceRequest(job_time_in_minutes=120),
    )

    # Limit number of jobs that will run at once on MICS account/partition
    limit_jobs_for_category(category="mics", max_jobs=max_jobs_on_mics)

    write_workflow_description()


if __name__ == "__main__":
    parameters_only_entry_point(compare_models_entrypoint)
