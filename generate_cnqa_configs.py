model = "roberta-large"

for taskname in ["physicaliqa-10pc","physicaliqa-25pc","physicaliqa"]:
    # Generate single task configs
    template = f'''task_name: {taskname}
model: {model}
accumulate_grad_batches: 16
use_amp: false
max_epochs: 4
learning_rate: 2e-6
adam_epsilon: 10e-8
warmup_steps: 150
batch_size: 4
max_length: 128
formula: "goal -> sol1|sol2"
train_x: "cache/{taskname}-train-dev/{taskname}-train-dev/train.jsonl"
train_y: "cache/{taskname}-train-dev/{taskname}-train-dev/train-labels.lst"
val_x: "cache/{taskname}-train-dev/{taskname}-train-dev/dev.jsonl"
val_y: "cache/{taskname}-train-dev/{taskname}-train-dev/dev-labels.lst"
save_path: "outputs/{model}-{taskname}"
'''
    with open(f'configs/{taskname}.yaml', 'w') as confile:
        print(template, file=confile)

    # Generate double task configs
    for taskname2 in ["cn_all_cs_10k", "cn_all_cs_20k", "cn_all_cs_40k", "cn_physical_10k"]:
        with open(f'configs/{taskname}-{taskname2}.yaml', 'w') as confile:
            template = f'''task_name: {taskname}
model: {model}
accumulate_grad_batches: 16
use_amp: false
max_epochs: 4
learning_rate: 2e-6
adam_epsilon: 10e-8
warmup_steps: 150
batch_size: 4
max_length: 128
formula: "goal -> sol1|sol2"
train_x: "cache/{taskname}-train-dev/{taskname}-train-dev/train.jsonl"
train_y: "cache/{taskname}-train-dev/{taskname}-train-dev/train-labels.lst"
val_x: "cache/{taskname}-train-dev/{taskname}-train-dev/dev.jsonl"
val_y: "cache/{taskname}-train-dev/{taskname}-train-dev/dev-labels.lst"
save_path: "outputs/{model}-{taskname}-{taskname2}"
task_name2: {taskname2}
train2_x: "cache/{taskname2}-train-dev/{taskname2}-train-dev/train.jsonl"
train2_y: "cache/{taskname2}-train-dev/{taskname2}-train-dev/train-labels.lst"
val2_x: "cache/{taskname2}-train-dev/{taskname2}-train-dev/dev.jsonl"
val2_y: "cache/{taskname2}-train-dev/{taskname2}-train-dev/dev-labels.lst"
formula2: "e1 + e2 -> sol1|sol2|sol3|sol4|sol5|sol6|sol7"   
'''
            print(template, file=confile)
