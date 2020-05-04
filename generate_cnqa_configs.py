# Default values - these can be overwritten in the training script
model = "roberta-large"
batchsize = 3
accumulate_grad_batches = 8
learning_rate = '5e-6'
dropout = 0.3
adam_epsilon = '1e-8'
epochs = 6
warmup_steps = 300
random_seed = 10061880

for taskname in ["physicaliqa-10pc","physicaliqa-25pc","physicaliqa"]:
    # Generate single task configs
    template = f'''task_name: {taskname}
model: {model}
batch_size: {batchsize}
accumulate_grad_batches: {accumulate_grad_batches}
use_amp: false
max_epochs: {epochs}
learning_rate: {learning_rate}
adam_epsilon: {adam_epsilon}
dropout: {dropout}
warmup_steps: {warmup_steps}
max_length: 128
formula: "goal -> sol1|sol2"
train_x: "cache/{taskname}-train-dev/{taskname}-train-dev/train.jsonl"
train_y: "cache/{taskname}-train-dev/{taskname}-train-dev/train-labels.lst"
val_x: "cache/{taskname}-train-dev/{taskname}-train-dev/dev.jsonl"
val_y: "cache/{taskname}-train-dev/{taskname}-train-dev/dev-labels.lst"
save_path: "outputs/{model}-{taskname}"
save_best_only: true
random_seed: {random_seed}
'''
    with open(f'configs/{taskname}.yaml', 'w') as confile:
        print(template, file=confile)

#     # Generate double task configs
#     for taskname2 in ["cn_all_cs_10k", "cn_all_cs_20k", "cn_all_cs_40k", "cn_physical_10k"]:
#         with open(f'configs/{taskname}-{taskname2}.yaml', 'w') as confile:
#             template = f'''task_name: {taskname}
# model: {model}
# batch_size: {batchsize}
# accumulate_grad_batches: {accumulate_grad_batches}
# use_amp: false
# max_epochs: {epochs}
# learning_rate: {learning_rate}
# adam_epsilon: {adam_epsilon}
# dropout: {dropout}
# warmup_steps: {warmup_steps}
# max_length: 128
# formula: "goal -> sol1|sol2"
# train_x: "cache/{taskname}-train-dev/{taskname}-train-dev/train.jsonl"
# train_y: "cache/{taskname}-train-dev/{taskname}-train-dev/train-labels.lst"
# val_x: "cache/{taskname}-train-dev/{taskname}-train-dev/dev.jsonl"
# val_y: "cache/{taskname}-train-dev/{taskname}-train-dev/dev-labels.lst"
# save_path: "outputs/{model}-{taskname}-{taskname2}"
# task_name2: {taskname2}
# train2_x: "cache/{taskname2}-train-dev/{taskname2}-train-dev/train.jsonl"
# train2_y: "cache/{taskname2}-train-dev/{taskname2}-train-dev/train-labels.lst"
# val2_x: "cache/{taskname2}-train-dev/{taskname2}-train-dev/dev.jsonl"
# val2_y: "cache/{taskname2}-train-dev/{taskname2}-train-dev/dev-labels.lst"
# formula2: "e1 + e2 -> sol1|sol2|sol3|sol4|sol5|sol6|sol7"
# save_best_only: true
# random_seed: {random_seed}
# '''
#             print(template, file=confile)
