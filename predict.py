import argparse
import json
from typing import *

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Classifier


# Parse the input file from JSONL to a list of dictionaries.
def read_jsonl_lines(input_file: str) -> List[dict]:
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]


@hydra.main(config_path="config/train.yaml", strict=False)
def load_config(config: DictConfig):
    return config


def main(input_file, output_file, config):

    model_to_predictions = {}
    model_to_confidences = {}
    task = ''
    for t in ['physicaliqa', 'socialiqa', 'alphanli', 'hellaswag']:
        if t in input_file:
            task = t
            break
    model_data = {'physicaliqa': [
        'physicaliqa_100_cn_10k_include_answers_in_context_0.ckpt',
        'physicaliqa_100_cn_10k_standard_42.ckpt',
        'physicaliqa_100_include_answers_in_context_10061880.ckpt',
        'physicaliqa_100_standard_42.ckpt'
    ]}

    #  Eval for each sub model
    for ckpt in model_data[task]:
        device = 'cpu' if not torch.cuda.is_available() else "cuda"
        checkpoint = torch.load(f'{task}_submission_models/{ckpt}', map_location=device)

        config['task'] = task
        if 'cn_10k' in ckpt: config['task2'] = 'cn_10k'
        if 'include_answers_in_context' in ckpt: config['architecture'] = 'include_answers_in_context'
        elif 'embed_all_sep_mean' in ckpt: config['architecture'] = 'embed_all_sep_mean'

        model = Classifier(config)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()

        preds: List[int] = []
        confidences: List[float] = []
        for batch in tqdm(
                DataLoader(model.dataloader(input_file), batch_size=model.hparams["batch_size"] * 2,
                           collate_fn=model.collate, shuffle=False)):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            with torch.no_grad():
                logits = model.forward(batch)

            preds.extend(torch.argmax(logits, dim=1).cpu().detach().numpy().tolist())
            softmax = torch.nn.functional.softmax(logits, dim=1)
            confidences.extend((torch.max(softmax, dim=1)[0].cpu().detach().numpy().tolist()))

        model_to_predictions[key] = [p + model.label_offset for p in preds]
        model_to_confidences[key] = [(c - 0.5) * 2 for c in confidences]

    predictions_df = (pd.DataFrame.from_dict(model_to_predictions) - 0.5) * 2  # Project to predictions to [-1, 1]
    confidences_df = pd.DataFrame.from_dict(model_to_confidences)
    scaled_df = predictions_df.mul(confidences_df, fill_value=1)  # Scale the predictions by multiplying with confidence

    predicted_answers = (scaled_df.mean(axis=1) > 0).astype(int).values.squeeze().tolist()  # Take the average of each row for ensembled predictions

    # Write the predictions to the output file.
    with open(output_file, "w") as f:
        for p in predicted_answers:
            f.write(str(p))
            f.write("\n")
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AI2 Submission.')
    parser.add_argument('--input-file', type=str, required=True, help='Location of test records', default=None)
    parser.add_argument('--output-file', type=str, required=True, help='Location of predictions', default=None)

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    config = load_config()
    main(args.input_file, args.output_file, config)