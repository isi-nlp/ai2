import os

parameter_options = {
                'task': ['alphanli', 'hellaswag', 'physicaliqa', 'socialiqa'],
                # 'train_data_slice': ['10', '25', '100'],
                # 'task2': ['cn_10k', 'cn_20k', 'cn_40k', 'cn_physical_10k'],
                # 'architecture': ['standard', 'include_answers_in_context', 'embed_all_sep_mean'],
                'architecture': ['include_answers_in_context', 'embed_all_sep_mean'],
                # 'random_seed': ['0', '42', '10061880'],
                'random_seed': ['10061880'],
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
    experiment_id = '_'.join(option for _, option in combination)
    os.system(f"sbatch run_saga "
          # Python script commands
          f"\""
              f"{' '.join([f'{name}={option}' for name,option in combination])}"
              f" save_path={experiment_id}"
              f"\" "
          # Additional sbatch specifications
          f"--job-name={experiment_id} "
          f"--output=outputs/slurm/{experiment_id}.out")
