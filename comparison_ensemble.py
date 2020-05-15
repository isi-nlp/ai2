import itertools
from sklearn.metrics import accuracy_score
import pandas as pd
from scipy.stats.stats import pearsonr

versions_to_predictions = {
    'standard_rs0': "outputs/roberta-large-physicaliqa_rs0/roberta-large-physicaliqa",
    # 'standard_rs42': "outputs/roberta-large-physicaliqa_rs42/roberta-large-physicaliqa",
    'standard_rs10061880': "outputs/roberta-large-physicaliqa",
    'arc1_rs0': 'outputs/roberta-large_rs0_acb2_lr5e-6/roberta-large-physicaliqa-arc1',
    # 'arc1_rs42': 'outputs/roberta-large_rs42_acb4_lr5e-6/roberta-large-physicaliqa-arc1',
    'arc1_rs10061880': 'outputs/roberta-large_rs10061880_acb2_lr5e-6/roberta-large-physicaliqa-arc1',
    'arc2_rs0': 'outputs/roberta-large_rs0_acb1_lr1e-6/roberta-large-physicaliqa-arc2',
    # 'arc2_rs42': 'outputs/roberta-large_rs42_acb8_lr5e-6/roberta-large-physicaliqa-arc2',
    'arc2_rs10061880': 'outputs/roberta-large_rs10061880_acb1_lr5e-6/roberta-large-physicaliqa-arc2',
}

gold_labels_path = 'cache/physicaliqa-train-dev/physicaliqa-train-dev/dev-labels.lst'
labels = pd.read_csv(gold_labels_path, sep='\t', header=None).values.tolist()

# Check accuracy of each model
for key in versions_to_predictions.keys():
    print('Accuracy of each model:')
    preds = pd.read_csv(versions_to_predictions[key]+'/pred.lst', sep='\t', header=None).values.tolist()
    accuracy = accuracy_score(labels, preds)
    print(f'{key},{accuracy}')

# Compare pairs of predictions of each model
print('ID1,ID22,Prediction Sim,Prediction Cor,Correctness Sim,Correctness Cor,Confidence Cor')
for id1, id2 in itertools.combinations(versions_to_predictions.keys(), 2):
    print(f'Pairwise comparison of {id1} and {id2}')
    model1, rs1 = tuple(id1.split('_'))
    model2, rs2 = tuple(id2.split('_'))
    if model1 != model2 and rs1 != rs2: continue # skip if both the model and rs are different

    preds1 = pd.read_csv(versions_to_predictions[id1]+'/pred.lst', sep='\t', header=None).values.tolist()
    conf1 = pd.read_csv(versions_to_predictions[id1]+'/pred.lst.cnf', sep='\t', header=None).values.tolist()
    correctness1 = [int(p == labels[i]) for i, p in enumerate(preds1)]
    preds2 = pd.read_csv(versions_to_predictions[id2]+'/pred.lst', sep='\t', header=None).values.tolist()
    conf2 = pd.read_csv(versions_to_predictions[id2]+'/pred.lst.cnf', sep='\t', header=None).values.tolist()
    correctness2 = [int(p == labels[i]) for i, p in enumerate(preds2)]

    print(f'{id1},{id2},{accuracy_score(preds1, preds2)},{pearsonr(preds1, preds2)[0]}\
            ,{accuracy_score(correctness1, correctness2)},{pearsonr(correctness1, correctness2)[0]}\
            ,{pearsonr(conf1, conf2)[0]}')
