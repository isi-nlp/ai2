MODEL_TYPE="t5"
MODEL_WEIGHTS="t5-small"
SEED=10061880

# double_experiments=[("physicaliqa-10pc","cn_all_cs_10k"), ("physicaliqa","cn_all_cs_10k")]
single_experiments=["t5-small-physicaliqa-25pc"]# "physicaliqa"]
double_experiments=[]

all_args = []

# Sweep parameters
# for BS in [1, 2, 3]:
for ACB in [2, 4, 8, 16]:
    for LR in ['5e-7', '1e-7', '5e-6', '1e-5', '5e-5']:
        for SEED in [0, 42, 10061880]:

            # For each parameter setup, prepate arguments file
            SWEEP_PARAMS=f"--accumulate_grad_batches {ACB} --learning_rate {LR} --random_seed {SEED}"
            SETUP_NAME=f"{MODEL_WEIGHTS}_rs{SEED}_acb{ACB}_lr{LR}"

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