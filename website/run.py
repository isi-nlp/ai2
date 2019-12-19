from bottle import route, run, template, static_file, request
import bottle
import os
import json
bottle.TEMPLATE_PATH.insert(0, 'views')

# pylint: disable=no-member
predictions = {
    'roberta': {
        "anli": "../output/roberta-roberta-large-alphanli-pred",
        "hellaswag": "../output/roberta-roberta-large-hellaswag-pred",
        "piqa": "../output/roberta-roberta-large-physicaliqa-pred",
        "siqa": "../output/roberta-roberta-large-socialiqa-pred"
    },
    'bert': {
        "anli":         "../output/bert-bert-large-cased-alphanli-pred",
        "hellaswag":    "../output/bert-bert-large-cased-hellaswag-pred",
        "piqa":         "../output/bert-bert-large-cased-physicaliqa-pred",
        "siqa":         "../output/bert-bert-large-cased-socialiqa-pred"
    },
    'xlnet': {
        "anli":         "../output/xlnet-xlnet-large-cased-alphanli-pred",
        "hellaswag":    "../output/xlnet-xlnet-large-cased-hellaswag-pred",
        "piqa":         "../output/xlnet-xlnet-large-cased-physicaliqa-pred",
        "siqa":         "../output/xlnet-xlnet-large-cased-socialiqa-pred"
    }

}

dataset = {
    "anli": {
        "labels":   "../cache/alphanli-train-dev/dev-labels.lst",
        "data":     "../cache/alphanli-train-dev/dev.jsonl",
        "offset": 1,
        "ctx": lambda x: x['obs1'] + "\n" + x["obs2"],
        "choices": lambda x: [
            x["hyp1"],
            x["hyp2"]
        ],
        "num": 2
    },
    "hellaswag": {
        "labels": "../cache/hellaswag-train-dev/hellaswag-train-dev/valid-labels.lst",
        "data": "../cache/hellaswag-train-dev/hellaswag-train-dev/valid.jsonl",
        "offset": 0,
        "ctx": lambda x: x["ctx"],
        "choices": lambda x: x["ending_options"],
        "num": 4
    },
    "piqa": {
        "labels": "../cache/physicaliqa-train-dev/physicaliqa-train-dev/dev-labels.lst",
        "data": "../cache/physicaliqa-train-dev/physicaliqa-train-dev/dev.jsonl",
        "offset": 0,
        "ctx": lambda x: x['goal'],
        "choices": lambda x: [
            x["sol1"],
            x["sol2"]
        ],
        "num": 2
    },
    "siqa": {
        "labels": "../cache/socialiqa-train-dev/socialiqa-train-dev/dev-labels.lst",
        "data": "../cache/socialiqa-train-dev/socialiqa-train-dev/dev.jsonl",
        "offset": 1,
        "ctx": lambda x: x["context"] + "\n" + x['question'],
        "choices": lambda x: [
            x["answerA"],
            x["answerB"],
            x["answerC"]
        ],
        "num": 3
    }
}


@route('/<filename:path>')
def send_static(filename):
    return static_file(filename, root='static/')


@route('/', method='GET')
def index():
    return template(
        'index.html', tasks=['anli', 'hellaswag', 'piqa', 'siqa'],
        models=['roberta', 'bert', 'xlnet'],
        filters={},
        task="anli", result={}, total=0, closest={})


@route('/', method='POST')
def retrieve():

    task = request.forms.get('task')
    labels = list(map(int, [l for l in open(dataset[task]['labels']).read().split("\n") if l]))
    filters = {}
    closest = {}
    for model in ['roberta', 'bert', 'xlnet']:
        if request.forms.get(model, None) is not None:
            filters[model] = request.forms.get(model, None)
            closest[model] = get_closest_train(model, range(len(labels)), task)

    # dataset = get_dataset("")

    results = [get_model_result(os.path.join(predictions[model][task], "dev-predictions.lst"),
                                labels=labels,
                                mode=filters[model],
                                source=model) for model in filters]

    d = get_dataset(dataset[task]['data'])

    def datum(x, p):

        res = {"ctx": dataset[task]["ctx"](x), "choices": [{"choice": c, "models": [(m[0], m[2])
                                                                                    for m in p if m[1] - dataset[task]["offset"] == i]} for i, c in enumerate(dataset[task]["choices"](x))]}

        return res

    result = {
        i: datum(d[i], c)
        for i, c in merge(*results).items()
    }


    return template(
        'index.html', tasks=['anli', 'hellaswag', 'piqa', 'siqa'],
        models=['roberta', 'bert', 'xlnet'],
        filters=filters, task=task, result=result, total=len(labels), closest=closest)


def get_dataset(path):
    import json
    with open(path) as f:
        data = [json.loads(l) for l in f]
    return data


def get_model_result(pred_path, labels, mode, source):

    import json

    with open(pred_path) as f:
        preds = list(map(int, [l for l in f.read().split('\n')]))
    print(pred_path)
    assert len(preds) == len(labels), f"{len(preds), len(labels)}"
    return {i: (source, preds[i], labels[i] == preds[i]) for i in range(len(labels)) if (mode == 'correct' and labels[i] == preds[i]) or (mode == 'wrong' and labels[i] != preds[i])}


def merge(*results):

    return {
        i: [r[i] for r in results] for i in results[0] if all(i in r for r in results)
    }


def get_closest_train(model, indices, task):

    root = "../embeds"
    task_name = "physicaliqa" if task == "piqa" else \
        "socialiqa" if task == "siqa" else \
        task

    pcorrect = os.path.join(root, f"{model}-{task_name}-pcorrect-wrong.json")
    pwrong = os.path.join(root, f"{model}-{task_name}-pwrong-wrong.json")

    result = {}

    if os.path.exists(pcorrect) and os.path.exists(pwrong):

        with open(pcorrect) as pc, open(pwrong) as pw:

            pc_data = json.loads(pc.read())
            pw_data = json.loads(pw.read())

            pc_data.update(pw_data)

            for i in indices:

                for j in range(i*dataset[task]["num"], (i+1)*dataset[task]["num"]):
                    if str(j) in pc_data:
                        result[i] = pc_data[str(j)]['train']['text'].replace("<s>", "").replace("</s>", " ")
        # print(result)
    return result




run(host='localhost', port=9999, reloader=True)
