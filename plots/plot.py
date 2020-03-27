import plotly.graph_objects as go
from collections import defaultdict
import pandas as pd

seed = '-10061880'
eval_file = 'eval_results'+seed+'.out'

data = defaultdict(dict)
errors = defaultdict(dict)
with open(eval_file) as eval:
    lines = eval.readlines()
    for i in range(0, len(lines), 3):
        experiment = lines[i].strip('roberta-large-').replace(seed, '').strip()
        # Get task name
        if 'cn' in experiment:
            if 'pc' in experiment:
                task1 = experiment[:experiment.index('-', 12)]
                task2 = experiment[experiment.index('-', 12)+1:]
            else:
                task1 = experiment[:experiment.index('-')]
                task2 = experiment[experiment.index('-')+1:]
        else:
            task1 = experiment
            task2 = 'No Conceptnet'
        task1 = task1.replace('pc', '-%').replace('physicaliqa-','').replace('physicaliqa', "100-%")
        task2 = task2.replace('all_cs_','').replace('cn', 'CN')

        # Get value
        result_tokens = lines[i+2].strip().split()
        low, high, ave = result_tokens[-5], result_tokens[-3].strip(','), result_tokens[-1]
        acc = lines[i+1].strip().split()[-1]
        e = (float(high) - float(low))/2
        data[task1][task2] = '{:2.1f}'.format(float(acc)*100)
        errors[task1][task2] = e

# Convert dictionary to dataframe
rows = []
for task_name, v in data.items():
    v['Name'] = task_name
    rows.append(v)
df = pd.DataFrame(rows)
df = df.set_index('Name')

# Error data frane
e_rows = []
for task_name, v in errors.items():
    v['Name'] = task_name
    e_rows.append(v)
e_df = pd.DataFrame(e_rows)
e_df = e_df.set_index('Name')


for enum, frames in enumerate([(df, e_df), (df.T, e_df.T)]):
    fig = go.Figure(data=[
        go.Bar(name=id,
               x=frames[0].columns,
               y=row,
               text=row,
               error_y=dict(
                type='data', # value of error bar given in data coordinates
                array=row_e,
                ),
            textposition='auto',) for (id, row), (id_e, row_e) in zip(frames[0].iterrows(), frames[1].iterrows())
    ])
    fig.update_yaxes(range=[50, 85], title='Accuracy')
    fig.update_layout(
    title="Accuracy vs Training Dataset random seed:"+seed,
    yaxis_title="Accuracy",
    font=dict(
        size=30,
    )
    )
    fig.show()
    # fig.write_image(eval_file + str(enum) + '.png')

