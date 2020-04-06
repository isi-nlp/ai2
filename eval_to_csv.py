# Split params text
def mysplit(s):
     head = s.rstrip('.0123456789')
     tail = s[len(head):]
     return head, tail


with open("search_eval_results.out", 'r') as eval:
    with open('param_search.csv', 'w') as out:
        csv_keys = 'AI2_Data,CN_Data,rs,bs,acb,ws,dr,low,high,accuracy'
        print(csv_keys, file=out)

        lines = eval.readlines()
        for i in range(0, len(lines), 3):
            outptus_line = lines[i].strip().replace('outputs/', '')
            param_string, experiment = outptus_line.split('/')

            # Set param strings
            params = [mysplit(param) for param in param_string.split('_')]
            csv_data = dict(params)

            # Set experiment strings
            experiment = experiment[experiment.index('-', 10) + 1:]
            data = experiment.split('-cn_')
            data_1 = data[0]
            data_2 = 'No CN'
            if len(data) > 1: data_2 = data[1]
            csv_data['AI2_Data'] = data_1
            csv_data['CN_Data'] = data_2

            accuracy = lines[i + 2].strip().split()[-1]
            result_tokens = lines[i + 2].strip().split()
            low, high, ave = result_tokens[-5], result_tokens[-3].strip(','), result_tokens[-1]
            csv_data['accuracy'] = accuracy
            csv_data['low'] = low
            csv_data['high'] = high

            print(','.join(csv_data[key] for key in csv_keys.split(',')), file=out)
