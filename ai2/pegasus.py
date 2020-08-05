from typing import Mapping, List, Tuple, Any, cast
from pathlib import Path

from vistautils.iter_utils import only
from vistautils.parameters import Parameters, YAMLParametersLoader
from vistautils.parameters_only_entrypoint import parameters_only_entry_point
from pegasus_wrapper import (
    initialize_vista_pegasus_wrapper,
    directory_for,
    # experiment_directory,
    run_python_on_parameters,
    limit_jobs_for_category,
    write_workflow_description,
)
from pegasus_wrapper.resource_request import ResourceRequest
from pegasus_wrapper.locator import Locator
from pegasus_wrapper.artifact import ValueArtifact

TIME_LIMIT_HOURS_NOT_ALPHANLI = 12  # Time limit in hours for tasks other than AlphaNLI
MINUTES_PER_HOUR = 60

# Default limit on the number of jobs that will run on MICS at once
DEFAULT_MAX_JOBS_ON_MICS = 2

# Represents a parameter combination as a list of (parameter_name, value) tuples.
ParameterCombination = List[Tuple[str, Any]]


def main(params: Parameters):
    initialize_vista_pegasus_wrapper(params)

    params_root = params.existing_directory('project_root') / 'parameters'
    parameter_options = params.namespace('parameter_options').as_nested_dicts()

    max_jobs_on_mics = params.integer('max_jobs_on_mics', default=DEFAULT_MAX_JOBS_ON_MICS)

    ensemble_params = params.namespace('ensemble')
    data_sizes = params.arbitrary_list('parameter_options.train_data_slice')
    ensemble_output_file_name = ensemble_params.string('output_file_name')

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
        key=lambda override: override_generality(override, parameter_options),
    )

    # Training phase.
    # Schedule training jobs for each parameter combination. Their outputs will be under "{experiment_root}/models".
    model_outputs_locator = Locator(('models',))
    task_to_jobs_info = {}
    for i, combination in enumerate(parameter_combinations):
        task: str = only(option for parameter, option in combination if parameter == 'task')
        train_data_slice: int = only(option for parameter, option in combination if parameter == 'train_data_slice')
        options: Tuple[str] = tuple(str(option) if option != '' else '_default' for _, option in combination)
        locator = model_outputs_locator / Locator(options)

        # Read in combination-specific parameters
        job_params = params.from_key_value_pairs([('model', params.namespace('model'))])
        job_params = job_params.unify(Parameters.from_key_value_pairs(combination, namespace_separator=None))
        params_root = params.existing_directory('project_root') / 'parameters'
        for parameter, option in combination:
            if option != '':
                parameter_directory = params_root / parameter
                if parameter_directory.exists():
                    option_params: Parameters = YAMLParametersLoader().load(
                        parameter_directory / f'{option}.params'
                    )
                    job_params = job_params.unify(option_params)

        # Because the job parameters tend to indirectly include root.params, which includes a
        # default partition, we need to override the partition setting to reflect our input
        # parameters.
        job_params = job_params.unify({'partition': params.string('partition')})

        # Process overrides
        for override in training_overrides:
            if override_matches(override, dict(combination)):
                job_params = job_params.unify({
                    parameter_option: value for parameter_option, value in override.items()
                    if parameter_option != 'parameter_options'
                })

        # Messy parameters input this shouldn't matter to ResourceRequest.
        # Maybe clean up later.
        resource_request = ResourceRequest.from_parameters(
            params.unify(job_params)
        )

        # Set common parameters and schedule the job.
        save_path = directory_for(locator)
        job_params = job_params.unify({
            'save_path': save_path,
            'save_best_only': False,
            'save_by_date_and_parameters': False,
            'eval_after_training': True,
        })
        job = run_python_on_parameters(
            locator,
            "ai2.train",
            job_params,
            depends_on=[],
            resource_request=resource_request
        )

        # We track job info so that it can be fed to the ensembling script.
        jobs_info = task_to_jobs_info.get(task, [])
        jobs_info.append({
            'job': job,
            'train_data_slice': train_data_slice,
            'parameters': combination,
            'predictions': ValueArtifact(locator=locator, value=Path('predictions.lst')),
            'confidence': ValueArtifact(locator=locator, value=Path('confidence.lst')),
        })
        task_to_jobs_info[task] = jobs_info

    # Ensembling phase.
    ensemble_locator = Locator(('ensembled',))

    # Load the gold labels for every task covered by ensembling.
    for task in ensemble_params.namespace('task_to_threshold').as_nested_dicts().keys():
        task_params = YAMLParametersLoader().load(
            params_root / 'task' / f'{task}.params'
        )
        ensemble_params = ensemble_params.unify({'task_to_gold': {
            task: {'val_y': task_params.existing_file('val_y')}
        }})

    ensemble_params = ensemble_params.unify({
        'data_sizes': data_sizes,
        'output_file': directory_for(ensemble_locator) / ensemble_output_file_name,
    })

    # Make a list of models and the relevant job info for the ensembling script to use. It needs to
    # know, for example, where to find their predictions and their confidences.
    for task, jobs_info in task_to_jobs_info.items():
        models_list = []
        for job_info in jobs_info:
            predictions_path = directory_for(job_info['predictions'].locator) / job_info['predictions'].value
            confidence_path = directory_for(job_info['confidence'].locator) / job_info['confidence'].value

            models_list.append({
                'train_data_slice': job_info['train_data_slice'],
                'parameters': job_info['parameters'],
                'predictions': predictions_path,
                'confidence': confidence_path,
            })

        ensemble_params = ensemble_params.unify({
            'models': {task: models_list}
        })

    run_python_on_parameters(
        ensemble_locator,
        'ai2.ensemble',
        ensemble_params,
        depends_on=[
            job_info['job']
            for jobs_info in task_to_jobs_info.values()
            for job_info in jobs_info
        ]
    )

    # Limit number of jobs that will run at once on MICS account/partition
    limit_jobs_for_category(category='mics', max_jobs=max_jobs_on_mics)

    write_workflow_description()


OVERRIDE_FILTER_KEY = 'parameter_options'


def override_generality(override: Mapping[str, Any], parameter_combinations: Mapping[str, List[Any]]) -> int:
    """
    Returns the generality of an override with respect to the given mapping of possible parameter
    combinations.

    The generality of an override is the number of configurations it applies to. A bigger number
    indicates a more general override. The generality is computed as the total number of parameter
    combinations the options dict applies to.
    """
    override_options = cast(Mapping[str, List[Any]], override[OVERRIDE_FILTER_KEY])
    complexity = 1
    for parameter_name, all_possible_values in parameter_combinations.items():
        allowed_values = override_options.get(parameter_name, all_possible_values)
        complexity *= len(allowed_values)
    return complexity


def override_matches(override: Mapping[str, Any], parameter_combination: Mapping[str, Any]) -> bool:
    override_options = cast(Mapping[str, List[Any]], override[OVERRIDE_FILTER_KEY])
    return all(parameter_combination.get(parameter_name) in allowed_values
               for parameter_name, allowed_values in override_options.items())


if __name__ == '__main__':
    parameters_only_entry_point(main)
