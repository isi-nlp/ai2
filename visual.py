import pandas as pd
import os


def read_csv(filename, header=True, sep='\t'):

    headers = []
    content = []
    with open(filename, "r") as input_file:
        for i, line in enumerate(input_file):
            if i == 0:
                headers.extend(line.strip('\r\n').split(sep))
            else:
                tmp = line.strip('\r\n').split(sep)
                assert len(tmp) == len(headers), f"Wrong fields in {line} -> {tmp}"
                content.append(tmp)

    return pd.DataFrame(content, columns=headers)


def get_difficulty(df):

    # print(df)
    # print(df.head())
    # exit(0)

    def difficulty(d):
        d = d.groupby(level=2)
        g = d.get_group('True')
        return g.values.tolist()[0].count('True')

    return df.apply(difficulty).values.tolist()


if __name__ == "__main__":
    for task, num_choice in [('anli', 2), ('hellaswag', 4), ('physicaliqa', 2), ('socialiqa', 3)]:
        final = None
        for root, dirs, files in os.walk("./data"):
            for f in files:
                if f'{task}-eval' in f or not f.startswith(task):
                    continue
                df = read_csv(os.path.join(root, f), sep='\t')
                df.rename(columns={'Prediction': f"{f.replace('eval.tsv', '').replace(task, '').strip('-')}"}, inplace=True)

                multi_index = pd.MultiIndex.from_tuples(
                    df[['Premise', 'Hypothesis', 'Truth']].values.tolist(),
                    names=['Premise', 'Hypothesis', 'Truth'])
                df = pd.DataFrame(df[f"{f.replace('eval.tsv', '').replace(task, '').strip('-')}"].values.tolist(), index=multi_index,
                                  columns=[f"{f.replace('eval.tsv', '').replace(task, '').strip('-')}"])

                final = df if final is None else pd.merge(
                    final, df, how='left', left_on=['Premise', 'Hypothesis', 'Truth'],
                    right_on=['Premise', 'Hypothesis', 'Truth'])

        scores = get_difficulty(final.groupby(level=0))
        scores = pd.DataFrame({'Premise': final.groupby(level=1).apply(lambda x: x.name).values.tolist(),
                               'Score': scores})
        scores = scores.sort_values(by=['Score'])
        print(scores.head())
        # print(score.head())
        # if final is not None:
        #     final.to_csv(f"{task}-eval.tsv", sep='\t')
