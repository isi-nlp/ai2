# find outputs/baseline outputs/pretrained/ -name *.out | xargs -I % bash -c "echo % && sed -n 6p % && tail -8 % | head -2" >results.txt

from pprint import pprint
import re

lines = open('results.txt').readlines()
exp = [re.match(r'outputs/(\w+)/.*', l).group(1) for l in lines[::4]]
task = [re.search(r"'task_name': '(\w+)'", l).group(1) for l in lines[1::4]]
acc = [re.search(r"Accuracy score: (\d\.\d+)", l).group(1) for l in lines[2::4]]
interval = [(m.group(1), m.group(2)) for m in [re.search(r"interval (\d+\.\d+) and (\d+\.\d+)", l) for l in lines[3::4]]]
results = sorted(zip(exp, task, acc, interval))
pprint(results)
