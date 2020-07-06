import argparse
import json
import os
from typing import *
import torch.nn.functional as F

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model import Classifier


# Parse the input file from JSONL to a list of dictionaries.
def read_jsonl_lines(input_file: str) -> List[dict]:
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]


def main(input_file, output_file):

    model_to_predictions = {}
    model_to_confidences = {}
    task = ''
    for t in ['physicaliqa', 'socialiqa', 'hellaswag']:
        if t in input_file:
            task = t
            break

    if 'anli' in input_file:
        task = 'alphanli'

    #  Eval for each sub model
    for ckpt in os.listdir(f"{task}_submission_models/."):
        config = {
            'random_seed': ckpt.strip('.ckpt').split('_')[-1],
            'architecture': 'standard',
            'with_true_label': True,
            'model': "roberta-large",
            'accumulate_grad_batches': 8,
            'use_amp': False,  # Half precision only works best with Volta architectures such as V100
            'max_epochs': 4,
            'learning_rate': 5e-6,
            'adam_epsilon': 1e-8,
            'warmup_steps': 300,
            'batch_size': 3,
            'dropout': 0.3,
            'max_length': 128,
        }
        device = 'cpu' if not torch.cuda.is_available() else "cuda"
        checkpoint = torch.load(f'{task}_submission_models/{ckpt}', map_location=device)

        with open(f'config/task/{task}.yaml', 'r') as ymlfile:
            config.update(yaml.load(ymlfile))

        if 'cn_10k' in ckpt:
            with open(f'config/task2/cn_10k.yaml', 'r') as ymlfile:
                config.update(yaml.load(ymlfile))
        if 'include_answers_in_context' in ckpt: config['architecture'] = 'include_answers_in_context'
        if 'embed_all_sep_mean' in ckpt: config['architecture'] = 'embed_all_sep_mean'

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

            num_choice = batch["num_choice"][0].item()
            logits = logits.reshape(-1, num_choice)

            preds.extend(torch.argmax(logits, dim=1).cpu().detach().numpy().tolist())
            confidences.extend(F.softmax(logits, dim=-1).cpu().detach().numpy().tolist())

        model_to_predictions[ckpt] = preds
        model_to_confidences[ckpt] = confidences

    confidences_df = pd.DataFrame.from_dict(model_to_confidences).applymap(np.asarray)
    confidences_df.to_csv(f'{task}_conf_predict.csv') # TODO Remove
    weighted_votes = confidences_df.sum(axis=1).apply(np.argmax).to_numpy()
    if task in ['socialiqa', 'alphanli']: weighted_votes += 1
    predicted_answers = weighted_votes.tolist()

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

    main(args.input_file, args.output_file)