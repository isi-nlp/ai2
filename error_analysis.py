import pandas as pd

for task in ['physicaliqa', 'socialiqa']:
    gold_labels = pd.read_csv(f"task_data/{task}-train-dev/internal-dev-labels.lst", sep='\t', header=None).values.squeeze().tolist()
    with open(f"task_data/{task}-train-dev/internal-dev.jsonl", 'r') as dataf:
        data = dataf.readlines()

    preds_90 = pd.read_csv(f'outputs/{task}_90_standard_42/predictions.lst', sep='\t', header=None).values.squeeze().tolist()
    preds_25 = pd.read_csv(f'outputs/{task}_25_standard_42/predictions.lst', sep='\t', header=None).values.squeeze().tolist()

    for i, gold in enumerate(gold_labels):
        if gold == preds_90[i] and gold != preds_25[i]:
            print(preds_90[i], preds_25[i])
            print(data[i])