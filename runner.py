import os

parameter_options = {
                'task': ['alphanli', 'hellaswag', 'physicaliqa', 'socialiqa'],
                # 'task': ['socialiqa'],
                'train_data_slice': ['10','25', '90'],
                # 'train_data_slice': ['10'],
                # 'task2': ['cn_10k', 'cn_20k', 'cn_40k', 'cn_physical_10k'],
                # 'task2': ['','cn_10k'],
                # 'task2': [''],
                # 'architecture': ['standard', 'include_answers_in_context', 'embed_all_sep_mean'],
                'architecture': ['standard'],
                # 'random_seed': ['0', '42', '10061880'],
                # 'random_seed': ['541401', '283219', '566944', '605430', '47299', '115719', '169760', '112068', '789504', '926273'],
                # 'random_seed': ['541401', '283219', '566944'],
                'random_seed': ['541401', '566944'],
                'learning_rate': ['5e-7','5e-6','5e-5'],
                'batch_size': ['2','3','6'],
                # 'batch_size': ['2'],
                'dropout': ['0','0.2','0.3'],
                # 'dropout': ['0.2'],
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
    os.system(f"sbatch "
          # Additional SLURM specifications
          f"-J {experiment_id} "
          f"-o outputs/slurm/{experiment_id}.out "
          # Ephemeral specifications - sudo sacctmgr modify user beser set MaxJobs=25
          # f"{'' if 'alphanli' in experiment_id else '--partition=ephemeral --qos=ephemeral --time=12:00:00 '}"
          f"slurm/run_saga.sh "
          # Python script commands
          f"\""
              f"{' '.join([f'{name}={option}' for name,option in combination  if option != ''])}"
              f" save_path={experiment_id}" 
              # f" save_best_only=False" 
              f"{' batch_size=2' if 'hellaswag' in experiment_id else ''}"
              f"\"")
