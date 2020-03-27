with open('eval_results-10061880.out') as eval:
    lines = eval.readlines()
    for i in range(0, len(lines), 2):
        experiment = lines[i].strip()
        result_tokens = lines[i+1].strip().split()
        low, high, ave = result_tokens[-5], result_tokens[-3].strip(','), result_tokens[-1]
        print(experiment[experiment.index('-',10)+1:])
        print(f"{ave} {{{low} - {high}}}")