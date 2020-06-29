import os

parameter_options = {
                # 'task': ['alphanli', 'hellaswag', 'physicaliqa', 'socialiqa'],
                'task': ['alphanli'],
                # 'train_data_slice': ['25', '90'],
                'train_data_slice': ['100'],
                # 'task2': ['cn_10k', 'cn_20k', 'cn_40k', 'cn_physical_10k'],
                # 'task2': ['','cn_10k'],
                'task2': ['cn_10k'],
                # 'architecture': ['standard', 'include_answers_in_context', 'embed_all_sep_mean'],
                'architecture': ['include_answers_in_context'],
                'random_seed': ['0', '42', '10061880'],
              }

# Create all possible combinations of parameters
parameter_combinations = [[]]
for parameter_name, options in parameter_options.items():
    new_combinations = []
    for combination in parameter_combinations:
        for option in options:
            new_combination = combination + [(parameter_name,option)]
            new_combinations.append(new_combination)
    parameter_combinations = new_combinations

for i, combination in enumerate(parameter_combinations):
    experiment_id = '_'.join(option for _, option in combination if option != '')
    # experiment_id = 'newsplit_'+experiment_id

    os.system(f"sbatch "
          # Additional SLURM specifications
          f"-J {experiment_id} "
          f"-o outputs/slurm/{experiment_id}.out "
          # Ephemeral specifications - sudo sacctmgr modify user beser set MaxJobs=25
          # f"--partition=ephemeral "
          # f"--qos=ephemeral "
          # f"--time=12:00:00 "
          # f"{'--gpus-per-task=2 ' if 'hellaswag' in experiment_id else ''}"
          f"slurm/run_saga.sh "
          # Python script commands
          f"\""
              f"{' '.join([f'{name}={option}' for name,option in combination  if option != ''])}"
              f" save_path={experiment_id}" 
              f"{' batch_size=2' if 'hellaswag' in experiment_id else ''}"
              f"\"")
