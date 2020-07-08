from pathlib import Path
from typing import Tuple

from vistautils.iter_utils import only
from vistautils import parameters_only_entrypoint
from vistautils.parameters import Parameters
from pegasus_wrapper import (
    initialize_vista_pegasus_wrapper,
    directory_for,
    # experiment_directory,
    run_python_on_parameters,
    write_workflow_description,
)
from pegasus_wrapper.resource_request import ResourceRequest
from pegasus_wrapper.locator import Locator
from pegasus_wrapper.artifact import DependencyNode, ValueArtifact


def main(params: Parameters):
    initialize_vista_pegasus_wrapper(params)

    parameter_options = params.namespace('parameter_options').as_mapping()
    print(parameter_options)

    # Compute all possible combinations of the parameters
    parameter_combinations = [[]]
    for parameter_name, options in parameter_options.items():
        new_combinations = []
        for combination in parameter_combinations:
            for option in options:
                new_combination = combination + [(parameter_name, option)]
                new_combinations.append(new_combination)
        parameter_combinations = new_combinations

    model_outputs_locator = Locator(('models',))
    jobs_info = []
    for i, combination in enumerate(parameter_combinations):
        task = only(option for parameter, option in combination if parameter == 'task')
        train_data_slice = only(option for parameter, option in combination if parameter == 'train_data_slice')
        options: Tuple[str] = tuple(option for _, option in combination if option != '')
        locator = model_outputs_locator / Locator(options)
        experiment_id = '_'.join(options)

        # os.system(f"sbatch "
        #           # Additional SLURM specifications
        #           f"-J {experiment_id} "
        #           f"-o outputs/slurm/{experiment_id}.out "
        #           # Ephemeral specifications - sudo sacctmgr modify user beser set MaxJobs=25
        #           f"{'' if 'alphanli' in experiment_id else '--partition=ephemeral --qos=ephemeral --time=12:00:00 '}"
        #           f"slurm/run_saga.sh "
        #           # Python script commands
        #           f"\""
        #           f"{' '.join([f'{name}={option}' for name,option in combination  if option != ''])}"
        #           f" save_path={experiment_id}"
        #           f" save_best_only=False"
        #           f"{' batch_size=2' if 'hellaswag' in experiment_id else ''}"
        #           f"\"")
        # TODO: Special logic for alphanli vs. not alphanli
        # TODO: Make sure I do this right with Slurm
        resource_request = ResourceRequest.from_parameters(params)
        slurm_output_path = directory_for(locator) / 'slurm.out'
        save_path = directory_for(locator)
        # TODO: Create parameters from the combination, that is the (parameter, option) list
        # Set save_best_only to false
        job_params = params
        job = run_python_on_parameters(
            locator,
            "TODO.train",
            # Pass some kind of parameters here to tell train.py where to put our stuff.
            job_params,
            # TODO maybe depend on the input file? however I specify that... I don't think I need to
            #  though.
            depends_on=[],
            # TODO set this up right, make sure it's using Slurm with the right parameters
            resource_request=ResourceRequest.from_parameters(params)
        )
        jobs_info.append({
            'job': job,
            'task': task,
            'train_data_slice': train_data_slice,
            'predictions': ValueArtifact(locator=locator, value=Path('predictions.lst')),
            'confidence': ValueArtifact(locator=locator, value=Path('confidence.lst')),
        })
    # TODO: Run ensembling
    write_workflow_description()


if __name__ == '__main__':
    parameters_only_entrypoint(main)
