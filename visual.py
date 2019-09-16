import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def read_csv(filename, header=True, sep='\t'):

    headers = []
    content = []
    with open(filename, "r") as input_file:
        for i, line in enumerate(input_file):
            if i == 0:
                headers.extend(line.strip('\r\n').split(sep))
            else:
                tmp = line.strip('\r\n').split(sep)
                assert len(tmp) == len(headers), f"Wrong fields in {line} -> {tmp} {headers}"
                content.append(tmp)

    return pd.DataFrame(content, columns=headers)


def get_difficulty(df, num_choice):

    def difficulty(d):
        # print(d)
        # exit(0)
        assert all(d['Premise'] == d['Premise'].values.tolist()[0]), d
        assert len(d) == num_choice, 'Wrong number of choices'

        res = d[d['Truth'] == 'True'].values.tolist()[0].count('True') - 1

        close = []

        for col in d.columns:
            if col.startswith('Probability'):
                predictions = d[col.replace('Probability-', '')]
                probabilities = d[col]

                if predictions[d['Truth'] == 'True'].values.tolist()[0] != 'True':
                    # print(probabilities[d['Truth'] == 'True'])
                    close.append(abs(float(probabilities[d['Truth'] == 'True'].values.tolist()[0])
                                     - float(probabilities[predictions == 'True'].values.tolist()[0])))
                else:
                    close.append(np.nan)

        return res, close

    for i in range(len(df)//num_choice):
        yield difficulty(df.iloc[i*num_choice:(i+1)*num_choice, :])


def rank(path):

    for task, num_choice in tqdm([('anli', 2), ('hellaswag', 4), ('physicaliqa', 2), ('socialiqa', 3), ('vcrqa', 4), ('vcrqar', 4)]):
        final = None
        for root, dirs, files in os.walk(path):
            for f in files:
                if f'{task}-eval' in f or not f.startswith(task) or not f.endswith('.tsv'):
                    continue
                print(os.path.join(root, f))
                df = read_csv(os.path.join(root, f), sep='\t')
                assert len(df) % num_choice == 0, f"{len(df)} {num_choice}"
                df.rename(columns={'Prediction': f"{f.replace('eval.tsv', '').replace(task, '').strip('-')}",
                                   'Probability': f"Probability-{f.replace('eval.tsv', '').replace(task, '').strip('-')}"}, inplace=True)
                if final is None:
                    final = df
                else:
                    final[f"{f.replace('eval.tsv', '').replace(task, '').strip('-')}"] = df[f"{f.replace('eval.tsv', '').replace(task, '').strip('-')}"]
                    final[f"Probability-{f.replace('eval.tsv', '').replace(task, '').strip('-')}"] = df[f"Probability-{f.replace('eval.tsv', '').replace(task, '').strip('-')}"]

        scores, closeness = zip(*list(get_difficulty(final, num_choice)))
        models = [x.replace('Probability-', '') for x in final.columns if x.startswith('Probability')]
        closeness = {
            n: x for n, x in zip(models, zip(*closeness))
        }

        choices = final['Hypothesis'].values.tolist()
        truth = final['Truth'].values.tolist()
        choices = ['\n'.join(['[Correct] ' + x if t == 'True' else x for x, t in zip(choices[i*num_choice: (i+1)*num_choice],
                                                                                     truth[i*num_choice: (i+1)*num_choice]
                                                                                     )]) for i in range(len(choices)//num_choice)]

        premises = final['Premise'].values.tolist()
        premises = [premises[i*num_choice: (i+1)*num_choice][0] for i in range(len(premises)//num_choice)]

        data = {'Premise': premises, 'Choices': choices, 'Score': scores}
        data.update(closeness)
        scores = pd.DataFrame(data)

        scores = scores.sort_values(by=['Score'])
        scores.to_csv(os.path.join(path, f'{task}-eval-rank.tsv'), sep='\t')


def heatmap(path):
    for task, num_choice in tqdm([('anli', 2), ('hellaswag', 4), ('physicaliqa', 2), ('socialiqa', 3), ('vcrqa', 4), ('vcrqar', 4)]):
        final = None
        for root, dirs, files in os.walk(path):
            for f in files:
                if f'{task}-eval-rank' in f:
                    # print(os.path.join(root, f))
                    df = pd.read_csv(os.path.join(root, f), sep='\t')
                    acc = df[[x for x in df.columns if x not in ['Unnamed: 0', 'Premise', 'Choices', 'Score']]].isna().sum()
                    acc /= len(df)
                    df.rename(
                        columns={x: f"{x} ({a*100:.2f})" for x,
                                 a in zip([x for x in df.columns if x not in ['Unnamed: 0', 'Premise', 'Choices', 'Score']],
                                          acc.values)},
                        inplace=True)
                    # print(df.columns)
                    data = df[[x for x in df.columns if x not in ['Unnamed: 0', 'Premise', 'Choices', 'Score']]].transpose()
                    ax = sns.heatmap(data, cmap="autumn", xticklabels=False, cbar_kws={"orientation": "horizontal"})
                    # ax.set(xticklabels=[])
                    ax.invert_xaxis()
                    ax.hlines([i for i in range(data.shape[1])], linewidth=0.5, *ax.get_xlim())
                    ax.vlines([data.columns[-1], data.columns[0]], linewidth=0.5, *ax.get_ylim())
                    #ax.axhline(y=1, color='k',linewidth=1)
                    #ax.axhline(y=data.shape[1], color='k',linewidth=1)
                    #ax.axvline(x=1, color='k',linewidth=1)
                    #ax.axvline(x=data.shape[0], color='k',linewidth=1)
                    ax.figure.tight_layout()
                    ax.figure.savefig(f"{task}.svg")

                    plt.clf()


def merge(path):

    for task, num_choice in tqdm([('anli', 2), ('hellaswag', 4), ('physicaliqa', 2), ('socialiqa', 3), ('vcrqa', 4), ('vcrqar', 4)]):
        final = None
        for root, dirs, files in os.walk(path):
            for f in files:
                if f'{task}-eval' in f or not f.startswith(task) or not f.endswith('.tsv'):
                    continue
                df = read_csv(os.path.join(root, f), sep='\t')
                assert len(df) % num_choice == 0, f"{len(df)} {num_choice}"
                df.rename(columns={'Prediction': f"{f.replace('eval.tsv', '').replace(task, '').strip('-')}",
                                   'Probability': f"Probability-{f.replace('eval.tsv', '').replace(task, '').strip('-')}"}, inplace=True)
                if final is None:
                    final = df
                else:
                    final[f"{f.replace('eval.tsv', '').replace(task, '').strip('-')}"] = df[f"{f.replace('eval.tsv', '').replace(task, '').strip('-')}"]
                    final[f"Probability-{f.replace('eval.tsv', '').replace(task, '').strip('-')}"] = df[f"Probability-{f.replace('eval.tsv', '').replace(task, '').strip('-')}"]

        final.to_csv(os.path.join(path, f'{task}-eval-proba.tsv'), sep='\t')


if __name__ == "__main__":
    # rank('.')
    # sns.set_style("darkgrid")
    # sns.set_style("white")
    # sns.set(rc={'axes.facecolor':'white'})

    merge('.')
