"""
Train two slightly different RoBERTa models and compare them on
"""
from typing import List, Tuple, Any

from more_itertools import only

from immutablecollections import immutableset
from vistautils.parameters import Parameters, YAMLParametersLoader
from vistautils.parameters_only_entrypoint import parameters_only_entry_point
from pegasus_wrapper import (
    initialize_vista_pegasus_wrapper,
    run_python_on_parameters,
    limit_jobs_for_category,
    write_workflow_description,
)
from pegasus_wrapper.resource_request import ResourceRequest
from pegasus_wrapper.locator import Locator
from pegasus_wrapper.artifact import ValueArtifact

import ai2.train as train_script
import ai2.percent_agreement as percent_agreement_script
from ai2.pegasus import override_generality, override_matches


TIME_LIMIT_HOURS_NOT_ALPHANLI = 12  # Time limit in hours for tasks other than AlphaNLI
MINUTES_PER_HOUR = 60

# Default limit on the number of jobs that will run on MICS at once
DEFAULT_MAX_JOBS_ON_MICS = 2

# Represents a parameter combination as a list of (parameter_name, value) tuples.
ParameterCombination = List[Tuple[str, Any]]


def main(params: Parameters):
    initialize_vista_pegasus_wrapper(params)

    experiment_root = params.creatable_directory('experiment_root')
    project_root = params.existing_directory('project_root')
    params_root = project_root / 'parameters'
    parameter_options = params.namespace('parameter_options').as_nested_dicts()

    max_jobs_on_mics = params.integer('max_jobs_on_mics', default=DEFAULT_MAX_JOBS_ON_MICS)

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
        list(params.namespace_or_empty('training_overrides')
             .as_nested_dicts()
             .values()),
        key=lambda override_: override_generality(override_, parameter_options),
    )

    # Schedule jobs for each parameter combination:
    # both a train job (output under 'models')
    # and an eval job (output under 'eval')
    model_outputs_locator = Locator(('models',))
    prediction_artifacts = []
    for idx, combination in enumerate(parameter_combinations):
        task: str = only(option for parameter, option in combination if parameter == 'task')
        options: Tuple[str] = tuple(str(option) if option != '' else '_default' for _, option in combination)
        train_locator = model_outputs_locator / Locator(options)

        # Set up common job parameters
        train_job_params = Parameters.from_key_value_pairs([
            ('model', params.namespace('model'))
        ]).unify(
            params.namespace("train")
        )

        # Read in combination-specific parameters
        train_job_params = train_job_params.unify(Parameters.from_key_value_pairs(combination, namespace_separator=None))
        for parameter, option in combination:
            if option != '':
                parameter_directory = params_root / parameter
                if parameter_directory.exists():
                    option_params: Parameters = YAMLParametersLoader().load(
                        parameter_directory / f'{option}.params'
                    )
                    train_job_params = train_job_params.unify(option_params)

        # Because the job parameters tend to indirectly include root.params, which includes a
        # default partition, we need to override the partition setting to reflect our input
        # parameters.
        train_job_params = train_job_params.unify({'partition': params.string('partition')})

        # Process overrides
        for override in training_overrides:
            if override_matches(override, dict(combination)):
                train_job_params = train_job_params.unify({
                    parameter_option: value for parameter_option, value in override.items()
                    if parameter_option != 'parameter_options'
                })

        # Messy parameters input. This shouldn't matter to ResourceRequest, though. Maybe clean up
        # later.
        resource_request = ResourceRequest.from_parameters(
            params.unify(train_job_params)
        )

        # Set common parameters and schedule the job.
        save_path = experiment_root / "_".join(
            "=".join(str(x) for x in option_pair)
            for option_pair in combination
        )
        train_job_params = train_job_params.unify({
            'save_path': save_path,
            'save_best_only': False,
            'save_by_date_and_parameters': False,
            'eval_after_training': True,
        })
        train_job = run_python_on_parameters(
            train_locator,
            train_script,
            train_job_params,
            depends_on=[],
            resource_request=resource_request,
        )

        prediction_artifacts.append(
            (
                combination,
                task,
                ValueArtifact(
                    value=save_path / "predictions.lst",
                    depends_on=immutableset([train_job]),
                )
            )
        )

    # Calculate the percent agreement for all same-task model pairs
    base_percent_agreement_locator = Locator(("percent_agreement",))
    for idx, (combination1, task1, model1_predictions_artifact) in enumerate(prediction_artifacts):
        model1_name = '__'.join(
            '='.join(str(x) for x in option_pair)
            for option_pair in combination1
            if "task" not in option_pair[0]
        )
        for combination2, task2, model2_predictions_artifact in prediction_artifacts[idx + 1:]:
            if task1 != task2:
                continue

            task_parameters: Parameters = YAMLParametersLoader().load(
                params_root / 'task' / f'{task1}.params'
            )
            percent_agreement_parameters = params.unify(task_parameters)

            model2_name = '__'.join(
                '='.join(str(x) for x in option_pair)
                for option_pair in combination2
                if "task" not in option_pair[0]
            )

            percent_agreement_locator = base_percent_agreement_locator / model1_name / model2_name
            run_python_on_parameters(
                percent_agreement_locator,
                percent_agreement_script,
                percent_agreement_parameters.unify({
                    'model1_predicted_labels': model1_predictions_artifact.value,
                    'model2_predicted_labels': model2_predictions_artifact.value,
                    'gold_labels': task_parameters.existing_file("val_y"),
                    'save_agreement_series_to': experiment_root / task1 / model1_name / model2_name / "agreement_series.csv",
                    'save_percent_agreement_to': experiment_root / task1 / model1_name / model2_name / "agreement.txt",
                }),
                depends_on=[model1_predictions_artifact, model2_predictions_artifact],
            )

    # Limit number of jobs that will run at once on MICS account/partition
    limit_jobs_for_category(category='mics', max_jobs=max_jobs_on_mics)

    write_workflow_description()


if __name__ == '__main__':
    parameters_only_entry_point(main)
