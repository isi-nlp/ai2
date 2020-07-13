from typing import List, Tuple, Any
from pathlib import Path

from more_itertools import flatten

from vistautils.iter_utils import only
from vistautils.parameters import Parameters, YAMLParametersLoader
from vistautils.parameters_only_entrypoint import parameters_only_entry_point
from pegasus_wrapper import (
    initialize_vista_pegasus_wrapper,
    directory_for,
    # experiment_directory,
    run_python_on_parameters,
    write_workflow_description,
)
from pegasus_wrapper.resource_request import ResourceRequest
from pegasus_wrapper.locator import Locator
from pegasus_wrapper.artifact import ValueArtifact


def main(params: Parameters):
    initialize_vista_pegasus_wrapper(params)

    parameter_options = params.namespace('parameter_options').as_nested_dicts()
    print(parameter_options)

    # Compute all possible combinations of the parameters
    parameter_combinations: List[List[Tuple[str, Any]]] = [[]]
    for parameter_name, options in parameter_options.items():
        new_combinations = []
        for combination in parameter_combinations:
            for option in options:
                new_combination = combination + [(parameter_name, option)]
                new_combinations.append(new_combination)
        parameter_combinations = new_combinations

    model_outputs_locator = Locator(('models',))
    task_to_jobs_info = {}
    for i, combination in enumerate(parameter_combinations):
        task: str = only(option for parameter, option in combination if parameter == 'task')
        train_data_slice: str = only(option for parameter, option in combination if parameter == 'train_data_slice')
        options: Tuple[str] = tuple(str(option) if option != '' else '_default' for _, option in combination)
        locator = model_outputs_locator / Locator(options)

        # Special logic for AlphaNLI
        # Tepmorarily disabled for testing purposes since I (Joe) don't have ephemeral access.
        # if task != 'alphanli':
        if False:
            resource_request = ResourceRequest.from_parameters(params.unify({
                'partition': 'ephemeral',
                # TODO: Set time limit of 12 hours.
            }))
        else:
            resource_request = ResourceRequest.from_parameters(params)

        # Set up combination-specific parameters
        job_params = Parameters.from_key_value_pairs(combination, namespace_separator=None)
        project_root = params.existing_directory('project_root')
        for parameter, option in combination:
            parameter_directory = project_root / parameter
            if parameter_directory.exists():
                option_params: Parameters = YAMLParametersLoader().load(
                    parameter_directory / f'{option}.params'
                )
                job_params = job_params.unify(option_params)

        # Special logic for Hellaswag
        if task == 'hellaswag':
            job_params = job_params.unify({
                'batch_size': 2
            })

        # Set general parameters
        save_path = directory_for(locator)
        job_params = job_params.unify({
            'save_path': save_path,
            'save_best_only': False,
        })
        job = run_python_on_parameters(
            locator,
            'ai2.train.train',
            job_params,
            depends_on=[],
            resource_request=resource_request
        )
        jobs_info = task_to_jobs_info.get(task, [])
        jobs_info.append({
            'job': job,
            'train_data_slice': train_data_slice,
            'parameter_combination': combination,
            'predictions': ValueArtifact(locator=locator, value=Path('predictions.lst')),
            'confidence': ValueArtifact(locator=locator, value=Path('confidence.lst')),
        })
        task_to_jobs_info[task] = jobs_info

    ensemble_params = params.namespace('ensemble')
    ensemble_params.unify({
        'data_sizes': params.arbitrary_list('parameter_options.train_data_slice'),
    })
    for task, jobs_info in task_to_jobs_info.items():
        models_list = []
        for job_info in jobs_info:
            predictions_path = directory_for(job_info['predictions'].locator) / job_info['predictions'].value
            confidence_path = directory_for(job_info['confidence'].locator) / job_info['confidence'].value

            models_list.append({
                'train_data_slice': job_info['train_data_slice'],
                'parameters': job_info['parameter_combination'],
                'predictions': predictions_path,
                'confidence': confidence_path,
            })

        ensemble_params = ensemble_params.unify({
            'models': {task: models_list}
        })

    run_python_on_parameters(
        Locator(('ensembled',)),
        'ai2.ensemble.main',
        ensemble_params,
        depends_on=[
            job_info['job']
            for jobs_info in task_to_jobs_info.values()
            for job_info in jobs_info
        ]
    )

    write_workflow_description()


if __name__ == '__main__':
    parameters_only_entry_point(main)
