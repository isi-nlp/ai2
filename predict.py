import argparse
import json
import pandas as pd
import yaml
from typing import *
import torch
from torch.utils.data import DataLoader
from model import Classifier
from tqdm import tqdm


# Parse the input file from JSONL to a list of dictionaries.
def read_jsonl_lines(input_file: str) -> List[dict]:
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]


def main(input_file, output_file):
    # Run ensembling
    model_to_predictions = {}
    model_to_confidences = {}
    model_data = [
        ('standard_rs10061880', 'configs/physicaliqa.yaml'),
        ('arc1_rs0', 'configs/physicaliqa-arc1.yaml'),
        ('arc2_rs10061880', 'configs/physicaliqa-arc2.yaml'),
        ('cn_20k_rs10061880', 'configs/physicaliqa-cn_all_cs_20k.yaml'),
    ]

    #  Eval for each sub model
    for key, config in model_data:
        device = 'cpu' if not torch.cuda.is_available() else "cuda"
        checkpoint = torch.load(f'submission_models/{key}.ckpt', map_location=device)
        with open(config, 'r') as ymlfile:
            config = yaml.load(ymlfile)
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

    predicted_answers = (scaled_df.mean(axis=1) > 0).values.squeeze().tolist()  # Take the average of each row for ensembled predictions

    # Write the predictions to the output file.
    with open(output_file, "w") as f:
        for p in predicted_answers:
            f.write(p)
            f.write("\n")
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A random baseline.')
    parser.add_argument('--input-file', type=str, required=True, help='Location of test records', default=None)
    parser.add_argument('--output-file', type=str, required=True, help='Location of predictions', default=None)

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(args.input_file, args.output_file)
