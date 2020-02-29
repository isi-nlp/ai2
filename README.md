# Minimal Code Base For AI2 Commonsense Leaderboard

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

### Get predictions with evaluation(f1, CI)

```bash
python eval.py \
    --input_x cache/physicaliqa-train-dev/physicaliqa-train-dev/dev.jsonl \
    --config config.yaml \
    --checkpoint outputs/2020-02-26/20-26-22/lightning_logs/version_6341419/checkpoints/_ckpt_epoch_3_v0.ckpt \
    --input_y cache/physicaliqa-train-dev/physicaliqa-train-dev/dev-labels.lst \
    --output pred.lst
```
