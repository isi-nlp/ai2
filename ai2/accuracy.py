import argparse
from sklearn.metrics import accuracy_score
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AI2 Submission.')
    parser.add_argument('--preds', type=str, required=True, help='Location of test records', default=None)
    parser.add_argument('--labels', type=str, required=True, help='Location of predictions', default=None)

    args = parser.parse_args()

    preds = pd.read_csv(args.preds, sep='\t', header=None).values.squeeze().tolist()
    labels = pd.read_csv(args.labels, sep='\t', header=None).values.squeeze().tolist()
    print(round(accuracy_score(preds,labels)*100,2))