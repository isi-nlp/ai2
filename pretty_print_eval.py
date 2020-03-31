with open('eval_results-42.out') as eval:
    lines = eval.readlines()
    for i in range(0, len(lines), 3):
        experiment = lines[i].strip()
        f1 = lines[i+2].strip().split()[-1]
        result_tokens = lines[i+2].strip().split()
        low, high, ave = result_tokens[-5], result_tokens[-3].strip(','), result_tokens[-1]
        print(experiment[experiment.index('-',10)+1:])
        # print(f1)
        print(f"{f1} {{{low} - {high}}}")