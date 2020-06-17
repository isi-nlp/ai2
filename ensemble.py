import csv
import itertools
import os
import numpy as np
from collections import Counter, defaultdict
import heapq

from more_itertools import powerset
from sklearn.metrics import accuracy_score
import pandas as pd
from scipy.stats.stats import pearsonr

tasks_to_threshold = {
    # 'alphanli':0.7,
    'hellaswag':0.6,
    'physicaliqa':0.6,
    'socialiqa':0.6
}
models = [name for name in os.listdir("outputs/.") if name != 'slurm']

def run_ensemble(predictions_df, confidences_df, subset):
    # confidences_df[confidences_df < 0.2] = 0  # Set low confidence values to 0.
    # confidences_df = confidences_df.eq(confidences_df.where(confidences_df != 0).max(1), axis=0).astype(int)  # Get the most confident

    relevant_confidences = confidences_df[subset]
    weighted_votes = relevant_confidences.sum(axis=1).apply(np.argmax).to_numpy()
    if task in ['socialiqa', 'alphanli']: weighted_votes += 1
    final_predictions = weighted_votes.tolist()
    # print(f'{accuracy},{[int(i in subset) for i in model_to_path.keys()]}'.replace(' ','').replace('[','').replace(']','')) # CSV
    # unweighted_votes = predictions_df[subset].mode(axis=1).too_nutolist()
    return accuracy_score(labels, final_predictions)

all_results = {}

for task in tasks_to_threshold.keys():
    for data_size in ['10','25','90']:
        results = {}
        print(f'\nRunning ensemble for {task.upper()}, {data_size}')
        relevant_models = [model for model in models if task in model and data_size == model.split('_')[1]]
        gold_labels_path = f'task_data/{task}-train-dev/internal-dev-labels.lst'
        labels = pd.read_csv(gold_labels_path, sep='\t', header=None).values.squeeze().tolist()

        best_per_seed_group = defaultdict(float)
        successful_models = []
        model_to_predictions = {}
        model_to_confidences = {}
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
                    print(f'{model},{round(accuracy*100,2)}')
                    results[model.replace(task+'_'+data_size+'_','')] = round(accuracy*100,2)

                    # model_without_seed = model.strip('_'+model.split('_')[-1])
                    # if accuracy > best_per_seed_group[model_without_seed]:
                    #     best_per_seed_group[model_without_seed] = accuracy
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

        predictions_df = pd.DataFrame.from_dict(model_to_predictions)
        confidences_df = pd.DataFrame.from_dict(model_to_confidences).applymap(np.asarray)
        # print(f'accuracy,{list(model_to_path.keys())}'.replace(' ','').replace('\'','').replace('[','').replace(']','')) # print for csv
        # Grid search for ensembling
        # ensemble_results = {}
        # for subset in powerset(successful_models):
        #     if len(subset) <= 1: continue
        #     subset = list(subset)
        #     ensemble_results[tuple(subset)]=run_ensemble(predictions_df, confidences_df, subset)
        # best = heapq.nlargest(10, ensemble_results, key=ensemble_results.get)
        # print(ensemble_results[best[0]])
        # best_performers = [m for ms in best for m in ms]
        # counts = Counter(best_performers)
        # print(counts.most_common())

        all_accuracy = round(run_ensemble(predictions_df, confidences_df, successful_models)*100,2)
        print('Ensemble of all models:', all_accuracy)
        results['All'] = all_accuracy
        # print('Ensemble of best per seed-group:', round(run_ensemble(predictions_df, confidences_df, best_per_seed_group.keys()),4))
        for factor in ['cn_10k', 'standard', 'include_answers_in_context', 'embed_all_sep_mean']:
            without_factor = [m for m in successful_models if factor not in m]
            wf_accuracy = round(run_ensemble(predictions_df, confidences_df, without_factor) * 100, 2)
            print(f'Without {factor}:', wf_accuracy)
            results[f'Without {factor}'] = wf_accuracy
    all_results[task + '_' + data_size] = results

print(all_results)
with open("test_output.csv", "wb") as f:
    w = csv.writer(f)
    types = all_results.values()[0].keys()
    for key in all_results.keys():
        w.writerow([key, [all_results[key][type] for type in types]])
