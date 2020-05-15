import itertools
from sklearn.metrics import accuracy_score
import pandas as pd
from scipy.stats.stats import pearsonr

model_to_path = {
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
model_to_predictions = {}
model_to_confidences = {}

gold_labels_path = 'cache/physicaliqa-train-dev/physicaliqa-train-dev/dev-labels.lst'
labels = pd.read_csv(gold_labels_path, sep='\t', header=None).values.squeeze().tolist()

# Check accuracy of each model
for key, path in model_to_path.items():
    print('Accuracy of each model:')
    preds = pd.read_csv(path + '/pred.lst', sep='\t', header=None).values.squeeze().tolist()
    confs = pd.read_csv(path + '/pred.lst.cnf', sep='\t', header=None).values.squeeze().tolist()
    accuracy = accuracy_score(labels, preds)
    model_to_predictions[key] = preds
    model_to_confidences[key] = confs
    print(f'{key},{accuracy}')

# Compare pairs of predictions of each model
print(
    'ID1,ID22,Pred Sim,Pred Cor,Correctness Cor,Confidence Cor,ConfCor Both Correct,ConfCor One Correct,ConfCor Both Wrong')
for id1, id2 in itertools.combinations(model_to_predictions.keys(), 2):
    model1, rs1 = tuple(id1.split('_'))
    model2, rs2 = tuple(id2.split('_'))
    if model1 != model2 and rs1 != rs2: continue  # skip if both the model and rs are different
    preds1, conf1 = model_to_predictions[id1], model_to_confidences[id1]
    correctness1 = [int(p == labels[i]) for i, p in enumerate(preds1)]
    preds2, conf2 = model_to_predictions[id2], model_to_confidences[id2]
    correctness2 = [int(p == labels[i]) for i, p in enumerate(preds2)]

    # ConfCor Both Correct
    ccbc = pearsonr(*zip(*[(conf1[i], conf2[i]) for i in range(len(preds1)) if correctness1[i] and correctness2[i]]))[0]
    # ConfCor Only One Correct
    ccoc = pearsonr(*zip(*[(conf1[i], conf2[i]) for i in range(len(preds1)) if correctness1[i] != correctness2[i]]))[0]
    # ConfCor Both Wrong
    ccbw = \
    pearsonr(*zip(*[(conf1[i], conf2[i]) for i in range(len(preds1)) if correctness1[i] == correctness2[i] == 0]))[0]

    print(
        f'{id1},{id2},{accuracy_score(preds1, preds2)},{pearsonr(preds1, preds2)[0]},{pearsonr(correctness1, correctness2)[0]},{pearsonr(conf1, conf2)[0]},{ccbc},{ccoc},{ccbw}')

# Run ensemble
# subset = ['standard_rs0', 'standard_rs10061880']
predictions_df = (pd.DataFrame.from_dict(model_to_predictions) - 0.5) * 2  # Project to predictions to [-1, 1]
confidences_df = pd.DataFrame.from_dict(model_to_confidences)
confidences_df[confidences_df < 0.2] = 0  # Set low confidence values to 0.
confidences_df.eq(confidences_df.where(confidences_df != 0).max(1), axis=0).astype(int)

scaled_df = predictions_df.mul(confidences_df, fill_value=1)  # Scale the predictions by multiplying with confidence
print('Predictions', predictions_df)
print('Confidences', confidences_df)
print('Scaled', scaled_df)
final_predictions = scaled_df.mean(axis=1) > 0  # Take the average of each row for ensembled predictions
print(accuracy_score(labels, final_predictions.values.squeeze().tolist()))
