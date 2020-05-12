import itertools
from sklearn.metrics import accuracy_score
import pandas as pd

versions_to_predictions = {
    # 'standard_rs0': "outputs/roberta-large-physicaliqa_rs0/roberta-large-physicaliqa",
    # 'standard_rs42': "outputs/roberta-large-physicaliqa_rs42/roberta-large-physicaliqa",
    'standard_rs10061880': "outputs/roberta-large-physicaliqa/roberta-large-physicaliqa",
    'arc1_rs0': 'outputs/roberta-large_rs0_acb2_lr5e-6/roberta-large-physicaliqa-arc1',
    'arc1_rs42': 'outputs/roberta-large_rs42_acb4_lr5e-6/roberta-large-physicaliqa-arc1',
    'arc1_rs10061880': 'outputs/roberta-large_rs10061880_acb2_lr5e-6/roberta-large-physicaliqa-arc1',
    'arc2_rs0': 'outputs/roberta-large_rs0_acb1_lr1e-6/roberta-large-physicaliqa-arc2',
    'arc2_rs42': 'outputs/roberta-large_rs42_acb8_lr5e-6/roberta-large-physicaliqa-arc2',
    'arc2_rs10061880': 'outputs/roberta-large_rs10061880_acb1_lr5e-6/roberta-large-physicaliqa-arc2',
}

gold_labels_path = 'cache/physicaliqa-train-dev/physicaliqa-train-dev/dev-labels.lst'
gold_labels = pd.read_csv(gold_labels_path, sep='\t', header=None).values.tolist()

# Compare pairs of predictions of each model
for id1, id2 in itertools.product(versions_to_predictions.keys(), repeat=2):
    if id1 == id2: continue
    preds1 = pd.read_csv(versions_to_predictions[id1]+'/pred.lst', sep='\t', header=None).values.tolist()
    preds2 = pd.read_csv(versions_to_predictions[id2]+'/pred.lst', sep='\t', header=None).values.tolist()
    similarity = accuracy_score(preds1, preds2)
    print(preds1, preds2, similarity)

# Check accuracy of each model
for key in versions_to_predictions.keys():
    preds = pd.read_csv(versions_to_predictions[key]+'/pred.lst', sep='\t', header=None).values.tolist()
    accuracy = accuracy_score(gold_labels, preds)
    print(key, accuracy)