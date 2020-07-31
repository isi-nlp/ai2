import pandas as pd
import ast

for task in ['physicaliqa', 'socialiqa']:
    gold_labels = pd.read_csv(f"task_data/{task}-train-dev/internal-dev-labels.lst", sep='\t', header=None).values.squeeze().tolist()
    with open(f"task_data/{task}-train-dev/internal-dev.jsonl", 'r') as dataf:
        data = dataf.readlines()

    preds_90 = pd.read_csv(f'outputs/{task}_90_standard_42/predictions.lst', sep='\t', header=None).values.squeeze().tolist()
    preds_25 = pd.read_csv(f'outputs/{task}_25_standard_42/predictions.lst', sep='\t', header=None).values.squeeze().tolist()

    results = []
    for i, gold in enumerate(gold_labels):
        if gold == preds_90[i] and gold != preds_25[i]:
            qa_dict = ast.literal_eval(data[i])
            if task == 'physicaliqa':
                results.append((preds_90[i], preds_25[i], qa_dict["goal"], qa_dict['sol1'], qa_dict['sol2']))
            else:
                results.append((preds_90[i], preds_25[i], qa_dict["context"], qa_dict["question"], qa_dict['answerA'], qa_dict['answerB'], qa_dict['answerC']))

    if task == 'physicaliqa':
        df = pd.DataFrame(results, columns =['90', '25', 'Question', 'Answer 1', 'Answer 2'])
    else:
        df = pd.DataFrame(results, columns =['90', '25', 'Question', 'Context', 'Answer 1', 'Answer 2', 'Answer 3'])
    df.to_csv(f'{task}_errors_90vs25.csv', na_rep='-')