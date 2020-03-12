# Minimal Code Base For AI2 Commonsense Leaderboard

## Dependencies

install apex if you want to use half precision: https://github.com/NVIDIA/apex. Conda env file is also included for reference, the apex might not be compatiable with conda directly so you can remove that before you create an environment.

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
| Roberta large (V100) |            77.4            |        75.7 - 79.4       |   77.3   |
| Roberta large (K80)  |            74.0            |        72.4 - 76.2       |   74.2   |
