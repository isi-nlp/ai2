# Minimal Code Base For AI2 Commonsense Leaderboard

## Dependencies

Create and run a virtual environment with Python 3.7:

```
conda create --name ai2 python=3.7
conda activate ai2
```

Then run:

```bash
pip install -r requirements.txt
```

## Train


Modify `config.yaml` as you like and run `python train.py` to train a model. It loads the config file and outputs all the logs/checkpoints in `outputs`

## Eval

### Get predictions without evaluation
```bash
python eval.py \
    --input_x cache/physicaliqa-train-dev/physicaliqa-train-dev/dev.jsonl \
    --config config.yaml \
    --checkpoint outputs/2020-02-26/20-26-22/lightning_logs/version_6341419/checkpoints/_ckpt_epoch_3_v0.ckpt \
    --output pred.lst
```

### Get predictions with evaluation(accuracy, confidence interval)

```bash
python eval.py \
    --input_x cache/physicaliqa-train-dev/physicaliqa-train-dev/dev.jsonl \
    --config config.yaml \
    --checkpoint outputs/2020-02-26/20-26-22/lightning_logs/version_6341419/checkpoints/_ckpt_epoch_3_v0.ckpt \
    --input_y cache/physicaliqa-train-dev/physicaliqa-train-dev/dev-labels.lst \
    --output pred.lst
```

## Results

### PIQA
|     Model     | Bootstrapped Accuracy Mean | Bootstrapped Accuracy CI | Accuracy |
|:-------------:|:--------------------------:|:------------------------:|:--------:|
| Roberta large (SAGA) |            76.9            |        75.2 - 78.6       |   76.9   |
