from typing import *
import torch
from torch.utils.data import DataLoader
from model import Classifier
from loguru import logger
from tqdm import tqdm
import yaml

if __name__ == "__main__":

    torch.manual_seed(42)
    import argparse

    parser = argparse.ArgumentParser("evaluate script")
    parser.add_argument("--input_x", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--input_y", type=str)
    # Optional parameters
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--accumulate_grad_batches', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--adam_epsilon', type=float, default=None)
    parser.add_argument('--warmup_steps', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--save_path', type=str, default=None)

    args = parser.parse_args()

    device = 'cpu' if not torch.cuda.is_available() else "cuda"
    checkpoint = torch.load(args.checkpoint, map_location=device)
    with open(args.config_file, 'r') as ymlfile:
        config = yaml.load(ymlfile)
        if args.max_epochs is not None:
            config['max_epochs'] = args.max_epochs
        if args.accumulate_grad_batches is not None:
            config['accumulate_grad_batches'] = args.accumulate_grad_batches
        if args.batch_size is not None:
            config['batch_size'] = args.batch_size
        if args.learning_rate is not None:
            config['learning_rate'] = args.learning_rate
        if args.adam_epsilon is not None:
            config['adam_epsilon'] = args.adam_epsilon
        if args.warmup_steps is not None:
            config['warmup_steps'] = args.warmup_steps
        if args.dropout is not None:
            config['dropout'] = args.dropout
        if args.random_seed is not None:
            config['random_seed'] = args.random_seed
        model = Classifier(config)

    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    preds: List[int] = []
    confidences: List[float] = []
    for batch in tqdm(
            DataLoader(model.dataloader(args.input_x, args.input_y), batch_size=model.hparams["batch_size"] * 2,
                       collate_fn=model.collate, shuffle=False)):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        with torch.no_grad():
            logits = model.forward(batch)
        preds.extend(torch.argmax(logits, dim=1).cpu().detach().numpy().tolist())
        softmax = torch.nn.functional.softmax(logits, dim=1)
        confidences.extend((torch.max(softmax, dim=1)[0].cpu().detach().numpy().tolist()))
    preds = [p + model.label_offset for p in preds]

    if args.input_y:

        from sklearn.metrics import accuracy_score
        import pandas as pd
        import numpy as np
        from scipy.stats.stats import pearsonr

        np.random.seed(42)

        labels = pd.read_csv(args.input_y, sep='\t', header=None).values.tolist()

        # Check correlation between confidence and correctness
        correctness = [int(p == labels[i]) for i, p in enumerate(preds)]
        print(len(correctness), len(confidences))
        print(pearsonr(correctness, confidences))

        stats = []
        for _ in range(1000):
            indices = [i for i in np.random.random_integers(0, len(preds) - 1, size=len(preds))]
            stats.append(accuracy_score([labels[j] for j in indices], [preds[j] for j in indices]))

        alpha = 0.95
        p = ((1.0 - alpha) / 2.0) * 100
        lower = max(0.0, np.percentile(stats, p))
        p = (alpha + ((1.0 - alpha) / 2.0)) * 100
        upper = min(1.0, np.percentile(stats, p))
        logger.info(f"Accuracy score: {accuracy_score(labels, preds):.3f}")
        logger.info(
            f'{alpha * 100:.1f} confidence interval {lower * 100:.1f} and {upper * 100:.1f}, average: {np.mean(stats) * 100:.1f}')

    with open(args.output, "w") as f:
        f.write("\n".join(map(str, preds)))
    with open(args.output + '.cnf', "w") as f:
        f.write("\n".join(map(str, confidences)))
