MODEL_TYPE="roberta"
MODEL_WEIGHTS="roberta-large"
SEED=10061880

double_experiments=[("physicaliqa-10pc","cn_all_cs_20k"), ("physicaliqa","cn_all_cs_20k")]
single_experiments=["physicaliqa-10pc", "physicaliqa"]

all_args = []

# Sweep parameters
for BS in [3, 4]:
    for ACB in [2, 8, 16]:
        for WS in [150, 300]:
            for DR in ['0', '0.3', '0.5']:

                # For each parameter setup, prepate arguments file
                SWEEP_PARAMS=f"--accumulate_grad_batches {ACB} --batch_size {BS} --warmup_steps {WS} --dropout {DR} --random_seed {SEED}"
                SETUP_NAME=f"rs{SEED}_bs{BS}_acb{ACB}_ws{WS}_dr{DR}"

                # Train double task experiments
                for TASK1, TASK2 in double_experiments:
                    # Setup all parameters
                    DIF_PARAMS=f"--config_file configs/{TASK1}-{TASK2}.yaml --save_path outputs/{SETUP_NAME}/{MODEL_WEIGHTS}-{TASK1}-{TASK2}"
                    PARAMS=f"{DIF_PARAMS} {SWEEP_PARAMS}"
                    all_args.append(PARAMS)

                # Train single task experiments
                for TASK1 in single_experiments:
                    # Setup all parameters
                    DIF_PARAMS=f"--config_file configs/{TASK1}.yaml  --save_path outputs/{SETUP_NAME}/{MODEL_WEIGHTS}-{TASK1}"
                    PARAMS=f"{DIF_PARAMS} {SWEEP_PARAMS}"
                    all_args.append(PARAMS)

for i, args in enumerate(all_args):
    with open(f'sweep_args/args_{i}.txt', 'w') as args_file:
        args_file.write(args)