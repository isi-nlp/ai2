import argparse
import json
from typing import List

def dict_inc(d, key):
    if key in d:
        d[key] += 1
    else:
        d[key] = 1


    
def read_lines(input_file: str) -> List[str]:
    lines = []
    with open(input_file, "rb") as f:
        for l in f:
            lines.append(l.decode().strip())
    return lines

def accuracy_string(category, correct, total):
    return "{0:<20} {1:>10.2f} ({2}/{3})".format(category, float(correct)/total, correct, total)
        
class AccuracyBreakdown:

    def __init__(self):
        self._category_correct = {}
        self._category_total = {}
        self.total_correct = 0
        self.total = 0

    def __str__(self):
        string = accuracy_string("Total", self.total_correct, self.total) + '\n'
        for cat in self.categories():
            string += accuracy_string(cat, self.category_correct(cat), self.category_total(cat))+'\n'
        return string

    def add_example(self, categories, correct):
        self.total += 1
        for cat in categories:
            dict_inc(self._category_total, cat)

        if correct:
            self.total_correct += 1
            for cat in categories:
                dict_inc(self._category_correct, cat)

    def category_accuracy(self, cat):
        return self.category_correct(cat) / float(self._category_total[cat])

    def category_correct(self, cat):
        if cat in self._category_correct:
            return self._category_correct[cat]
        else:
            return 0

    def category_total(self, cat):
        if cat in self._category_total:
            return self._category_total[cat]
        else:
            return 0

    def categories(self):
        return self._category_total.keys()

    def to_json_dict(self):
        json_dict = {'accuracy':float(self.total_correct)/self.total}
        for cat in self.categories():
            json_dict[cat] = {'correct':self.category_correct(cat),
                              'total':self.category_total(cat),
                              'accuracy':self.category_accuracy(cat)}
        return json_dict

def main(args):
    answers = read_lines(args.answers_file)
    preds = read_lines(args.preds_file)
    questions = read_lines(args.questions_file) if args.questions_file else None
    
    if len(preds) != len(answers):
        raise Exception("The prediction file does not contain the same number of lines as the number of test instances.")

    breakdown = AccuracyBreakdown()

    for i, pred in enumerate(preds):
        pred = int(pred)
        answer = json.loads(answers[i])['correct_answer']
        categories = []
        if questions:
            try:
                question = json.loads(questions[i])
                categories = question['categories'] + [question['questionType']]
            except:
                #print("Could not parse question: {0}".format(questions[i]))
                pass
        breakdown.add_example(categories, correct=(answer==pred))

    print(breakdown)

    with open(args.metrics_output_file, 'w') as f:
        json_dict = breakdown.to_json_dict()
        f.write(json.dumps(json_dict, indent=2))

def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate CycIC predictions")
    # Required Parameters
    parser.add_argument('--answers_file', type=str, help='Location of test answers', default=None, required=True)
    parser.add_argument('--preds_file', type=str, help='Location of predictions', default=None, required=True)
    parser.add_argument('--metrics_output_file',
                        type=str,
                        help='Location of output metrics file',
                        default="metrics.json")
    #Optional: pass in the questions file to get accuracy broken down by category
    parser.add_argument("--questions_file", type=str, help="Location of test questions. Pass this parameter to get accuracy broken down by question categories.")
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
