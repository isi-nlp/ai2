import itertools
import os
import numpy as np
from collections import Counter

from more_itertools import powerset
from sklearn.metrics import accuracy_score
import pandas as pd
from scipy.stats.stats import pearsonr

tasks_to_threshold = {
    # 'alphanli':0.7,
    # 'hellaswag':0.7,
    'physicaliqa':0.75,
    # 'socialiqa':0.7
}
models = [name for name in os.listdir("outputs/.") if name != 'slurm']

model_to_predictions = {}
model_to_confidences = {}

for task in tasks_to_threshold.keys():
    print(f'Running ensemble for {task}')
    relevant_models = [model for model in models if task in model and '90' in model and 'standard' in model]
    gold_labels_path = f'task_data/{task}-train-dev/internal-dev-labels.lst'
    labels = pd.read_csv(gold_labels_path, sep='\t', header=None).values.squeeze().tolist()

    successful_models = []
    # Get Accuracies
    print('Accuracy of each model:')
    for model in relevant_models:
        path = 'outputs/'+model
        try:
            preds = pd.read_csv(path + '/predictions.lst', sep='\t', header=None).values.squeeze().tolist()
            confs = pd.read_csv(path + '/confidence.lst', sep='\t', header=None).values.squeeze().tolist()
            accuracy = accuracy_score(labels, preds)
            if accuracy > tasks_to_threshold[task]:
                successful_models.append(model)
                model_to_predictions[model] = preds
                model_to_confidences[model] = confs
                print(f'{model},{accuracy}')
        except:
            print(f'Couldn\'t find preds for {model}')
            continue


    # Compare Models
    # print('Compare pairs of predictions of each model')
    # print('ID1,ID22,Pred Sim,Pred Cor,Correctness Cor,Confidence Cor,ConfCor Both Correct,ConfCor One Correct,ConfCor Both Wrong')
    # for id1, id2 in itertools.combinations(relevant_models, 2):
    #     model1, rs1 = tuple(id1.split('_'))
    #     model2, rs2 = tuple(id2.split('_'))
    #     if model1 != model2 and rs1 != rs2: continue  # skip if both the model and rs are different
    #     preds1, conf1 = model_to_predictions[id1], model_to_confidences[id1]
    #     correctness1 = [int(p == labels[i]) for i, p in enumerate(preds1)]
    #     preds2, conf2 = model_to_predictions[id2], model_to_confidences[id2]
    #     correctness2 = [int(p == labels[i]) for i, p in enumerate(preds2)]
    #     # ConfCor Both Correct
    #     ccbc = pearsonr(*zip(*[(conf1[i], conf2[i]) for i in range(len(preds1)) if correctness1[i] and correctness2[i]]))[0]
    #     # ConfCor Only One Correct
    #     ccoc = pearsonr(*zip(*[(conf1[i], conf2[i]) for i in range(len(preds1)) if correctness1[i] != correctness2[i]]))[0]
    #     # ConfCor Both Wrong
    #     ccbw = \
    #         pearsonr(*zip(*[(conf1[i], conf2[i]) for i in range(len(preds1)) if correctness1[i] == correctness2[i] == 0]))[
    #             0]
    #     print(
    #         f'{id1},{id2},{accuracy_score(preds1, preds2)},{pearsonr(preds1, preds2)[0]},{pearsonr(correctness1, correctness2)[0]},{pearsonr(conf1, conf2)[0]},{ccbc},{ccoc},{ccbw}')
    # print('\n')

    ensemble_results = {}

    # Run ensemble
    predictions_df = pd.DataFrame.from_dict(model_to_predictions)
    confidences_df = pd.DataFrame.from_dict(model_to_confidences).applymap(np.asarray)
    # print(f'accuracy,{list(model_to_path.keys())}'.replace(' ','').replace('\'','').replace('[','').replace(']','')) # print for csv
    for subset in powerset(successful_models):
        if len(subset) <= 1: continue
        subset = list(subset)
        # confidences_df[confidences_df < 0.2] = 0  # Set low confidence values to 0.
        # confidences_df = confidences_df.eq(confidences_df.where(confidences_df != 0).max(1), axis=0).astype(int)  # Get the most confident

        # unweighted_votes = predictions_df[subset].mode(axis=1).too_nutolist()
        relevant_confidences = confidences_df[subset]
        weighted_votes = relevant_confidences.sum(axis=1).apply(np.argmax).to_numpy()
        if task in ['socialiqa', 'alphanli']: weighted_votes+=1
        final_predictions = weighted_votes.tolist()
        accuracy = accuracy_score(labels, final_predictions)

        # print('Predictions', predictions_df)
        # print('Confidences', confidences_df)
        # print(f'{accuracy},{[int(i in subset) for i in model_to_path.keys()]}'.replace(' ','').replace('[','').replace(']','')) # CSV

        # if accuracy > 0.8:
        #     print(f'{accuracy},{subset}')

        ensemble_results[tuple(subset)]=accuracy
    best = sorted(ensemble_results, key=ensemble_results.get, reverse=True)[:10]
    best_performers = [m for ms in best for m in ms]
    counts = Counter(best_performers)
    print(counts.most_common())

