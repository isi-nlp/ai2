import argparse
import warnings
import torch
from ai2.model import Classifier

warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Eval ai2 darpa tasks with pytorch-transformers')
    parser.add_argument('--weights_path', help='Saved model weights file')
    parser.add_argument('--tags_csv', help='tags_csv from output directory')
    parser.add_argument('--output', '-o', help='Output file')

    args = parser.parse_args()

    pretrained_model = Classifier.load_from_metrics(
        weights_path=args.weights_path,
        tags_csv=args.tags_csv,
        on_gpu=torch.cuda.is_available(),
        map_location=None
    )

    # predict
    pretrained_model.eval()
    pretrained_model.freeze()

    dataloader = pretrained_model.val_dataloader()

    outputs = []

    for i, batch in enumerate(dataloader):
        res = pretrained_model.validation_step(batch, i)
        outputs.append(res)

    truth = torch.cat([x['truth'] for x in outputs], dim=0).reshape(-1).cpu().detach().numpy().tolist()
    pred = torch.cat([x['pred'] for x in outputs], dim=0).reshape(-1).cpu().detach().numpy().tolist()

    assert truth == list(map(int, pretrained_model.dev_y))

    with open(args.output, "w") as output:
        output.write(f"Premise\tHypothesis\tTruth\tPrediction\n")
        for example, p in zip(pretrained_model.helper.preprocess(pretrained_model.dev_x, pretrained_model.dev_y), pred):
            for i, pair in enumerate(example.pairs):
                output.write(f"{pair.premise}\t{pair.hypothesis}\t{exmple.label == i}\t{p == i}\n")
