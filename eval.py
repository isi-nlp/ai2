from typing import *
import torch
from torch.utils.data import DataLoader
from model import Classifier
from loguru import logger
import yaml

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser("evaluate script")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--input_y", type=str)

    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    with open(args.config, "r") as f:
        model = Classifier(yaml.safe_load(f.read()))
    model.load_state_dict(checkpoint['state_dict'])

    preds: List[int] = []
    for batch in DataLoader(model.__dataloader(args.input_x, args.input_y), batch_size=model.config["batch_size"], collate_fn=model.__collate):

        logits = model.forward(batch)
        preds.extend(torch.argmax(logits, dim=1).cpu().detach().numpy().tolist())

    preds = [p - Classifier.label_offset for p in preds]

    if args.input_y:
        from sklearn.metrics import f1_score
        import pandas as pd
        labels = pd.read_csv(args.input_y, sep='\t', header=None).values.tolist()
        logger.info(f"F1 score: {f1_score(labels, preds):.3f}")

    with open(args.output, "w") as f:
        f.write("\n".join(map(str, preds)))







