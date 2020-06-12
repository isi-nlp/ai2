with open('results.txt', 'r') as r:
    lines = r.readlines()

with open('results.csv', 'w') as out:
    print('Task1,DataAmount,Task2,Architecture,RandomSeed,Accuracy,ConfLow,ConfHigh,Average', file=out)
    for i, line in enumerate(lines):
        if 'outputs/slurm/' in line:
            model = line.split("outputs/slurm/")[1].split('.out')[0]
            task2 = '-' if 'cn_10k' not in model else 'cn_10k'
            model = model.replace('embed_all_sep_mean','embed-all-sep-mean').replace('include_answers_in_context','include-answers-in-context')
            task1,data,arc,rs = model.replace('_cn_10k','').split('_')
            if 'Accuracy' in lines[i+1]:
                accuracy = lines[i+1].split()[-1]
                low, _, high,_,ave = tuple(lines[i+2].split()[-5:])
                high = high.strip(',')
                print(','.join([task1,data,task2,arc,rs,accuracy,low,high,ave]), file=out)
            else:
                print(','.join([task1,data,task2,arc,rs,accuracy,'-','-','-']), file=out)