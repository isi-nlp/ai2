import os

# parameter_options = {
#                 'task': ['alphanli', 'hellaswag', 'physicaliqa', 'socialiqa'],
#                 # 'task': ['physicaliqa'],
#                 # 'train_data_slice': ['25', '90'],
#                 'train_data_slice': ['100'],
#                 # 'task2': ['cn_10k', 'cn_20k', 'cn_40k', 'cn_physical_10k'],
#                 'task2': ['','cn_10k'],
#                 'task2': [''],
#                 'architecture': ['standard', 'include_answers_in_context', 'embed_all_sep_mean'],
#                 # 'architecture': ['include_answers_in_context'],
#                 'random_seed': ['0', '42', '10061880'],
#                 # 'random_seed': ['10061880'],
#               }
#
# # Create all possible combinations of parameters
# parameter_combinations = [[]]
# for parameter_name, options in parameter_options.items():
#     new_combinations = []
#     for combination in parameter_combinations:
#         for option in options:
#             new_combination = combination + [(parameter_name,option)]
#             new_combinations.append(new_combination)
#     parameter_combinations = new_combinations
#
# for i, combination in enumerate(parameter_combinations):
#     # experiment_id = '_'.join(option for _, option in combination if option != '')
#     break


# 'alphanli_100_include_answers_in_context_0',
# 'alphanli_100_cn_10k_standard_0',
# 'alphanli_100_cn_10k_include_answers_in_context_0',
# 'alphanli_100_standard_42'
models = f"""
'physicaliqa_100_cn_10k_standard_42
'physicaliqa_100_include_answers_in_context_10061880
'physicaliqa_100_cn_10k_include_answers_in_context_0
'physicaliqa_100_standard_42'
'hellaswag_100_standard_0',
'hellaswag_100_include_answers_in_context_42', 
'hellaswag_100_cn_10k_standard_0', 
'hellaswag_100_cn_10k_include_answers_in_context_0',
'hellaswag_100_cn_10k_embed_all_sep_mean_0', 
'hellaswag_100_embed_all_sep_mean_42'
'socialiqa_100_cn_10k_standard_42', 
'socialiqa_100_cn_10k_include_answers_in_context_0', 
'socialiqa_100_cn_10k_embed_all_sep_mean_10061880',
'socialiqa_100_include_answers_in_context_0', 
'socialiqa_100_embed_all_sep_mean_42',
'socialiqa_100_standard_42'
"""

for model in models.split():
    experiment_id = model.replace('\'', '').replace(',', '').replace(' ', '')
    combination = [('train_data_slice','100')]
    combination.append(('random_seed', experiment_id.split('_')[-1]))
    combination.append(('task', experiment_id.split('_')[0]))
    if 'cn_10k' in experiment_id:
        combination.append(('task2','cn_10k'))
    if 'standard' in experiment_id:
        combination.append(('architecture','standard'))
    elif 'include_answers_in_context' in experiment_id:
        combination.append(('architecture','include_answers_in_context'))
    elif 'embed_all_sep_mean' in experiment_id:
        combination.append(('architecture','embed_all_sep_mean'))


    os.system(f"sbatch "
          # Additional SLURM specifications
          f"-J {experiment_id} "
          f"-o outputs/slurm/{experiment_id}.out "
          # Ephemeral specifications - sudo sacctmgr modify user beser set MaxJobs=25
          f"--partition=ephemeral "
          f"--qos=ephemeral "
          f"--time=12:00:00 "
          f"{'--gpus-per-task=2 ' if 'hellaswag' in experiment_id else ''}"
          f"slurm/run_saga.sh "
          # Python script commands
          f"\""
              f"{' '.join([f'{name}={option}' for name,option in combination  if option != ''])}"
              f" save_path={experiment_id}" 
              f" save_best_only=False" 
              f"{' batch_size=2' if 'hellaswag' in experiment_id else ''}"
              f"\"")
