import plotly.graph_objects as go
from collections import defaultdict
import pandas as pd

# import plotly.graph_objects as go
# animals=['giraffes', 'orangutans', 'monkeys']
#
# fig = go.Figure(data=[
#     go.Bar(name='SF Zoo', x=animals, y=[20, 14, 23]),
#     go.Bar(name='LA Zoo', x=animals, y=[12, 18, 29])
# ])
# # Change the bar mode
# fig.update_layout(barmode='group')
# fig.show()
task1s = []
task2s = []

data = defaultdict(dict)
errors = defaultdict(dict)
with open('eval_results.out') as eval:
    lines = eval.readlines()
    for i in range(0, len(lines), 2):
        experiment = lines[i].strip('roberta-large-').strip()
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
        if task1 not in task1s: task1s.append(task1)
        if task2 not in task2s: task2s.append(task2)
        task1 = task1.replace('pc', 'percent')
        task2 = task2.replace('all_cs_','').replace('cn', 'CN')

        result_tokens = lines[i+1].strip().split()
        low, high, ave = result_tokens[-5], result_tokens[-3].strip(','), result_tokens[-1]
        e = (float(high) - float(low))/2
        data[task1][task2] = ave
        errors[task1][task2] = e

rows = []
for task_name, v in data.items():
    v['Name'] = task_name
    rows.append(v)

df = pd.DataFrame(rows)
df = df.set_index('Name')

# for d in [df, df.T]:
fig = go.Figure(data=[
    go.Bar(name=id, x=df.columns, y=row, text=row,
            textposition='auto',) for id, row in df.iterrows()
])
fig.show()
# # Change the bar mode
# fig.update_layout(barmode='group')
# fig.show()
#
# months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
#           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
#
# fig = go.Figure()
# fig.add_trace(go.Bar(
#     x=months,
#     y=[20, 14, 25, 16, 18, 22, 19, 15, 12, 16, 14, 17],
#     name='Primary Product',
#     marker_color='indianred'
# ))
# fig.add_trace(go.Bar(
#     x=months,
#     y=[19, 14, 22, 14, 16, 19, 15, 14, 10, 12, 12, 16],
#     name='Secondary Product',
#     marker_color='lightsalmon'
# ))
#
# # Here we modify the tickangle of the xaxis, resulting in rotated labels.
# fig.update_layout(barmode='group', xaxis_tickangle=-45)
# fig.show()
#
# figurename = 'asd'
# fig.write_image(figurename+'.png')

