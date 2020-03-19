# Minimal Code Base For AI2 Commonsense Leaderboard

## Dependencies

Create and run a virtual environment with Python 3.7. If you're using conda, make sure to use conda version `>=4.8.2`.

```
conda create --name ai2 python=3.7
conda activate ai2
```

Then run:

```bash
pip install -r requirements.txt
```

## Train


The main code to train a model is in `train.py`. It loads the configuration file `config.yaml` and outputs all the logs/checkpoints in `outputs`. 

To submit it as a job on SAGA cluster, you should be able to simply run:

```
sbatch run_saga.sh
```

## Eval

### Get predictions without evaluation
```bash
python eval.py \
    --input_x cache/physicaliqa-train-dev/physicaliqa-train-dev/dev.jsonl \
    --config config.yaml \
    --checkpoint outputs/path_to_checkpoint/_ckpt_epoch_3_v0.ckpt \
    --output pred.lst
```

### Get predictions with evaluation(accuracy, confidence interval)

```bash
python eval.py \
    --input_x cache/physicaliqa-train-dev/physicaliqa-train-dev/dev.jsonl \
    --config config.yaml \
    --checkpoint outputs/path_to_checkpoint/_ckpt_epoch_3_v0.ckpt \
    --input_y cache/physicaliqa-train-dev/physicaliqa-train-dev/dev-labels.lst \
    --output pred.lst
```

## Results

### PIQA
|     Model     | Bootstrapped Accuracy Mean | Bootstrapped Accuracy CI | Accuracy |
|:-------------:|:--------------------------:|:------------------------:|:--------:|
| Roberta large (SAGA) |            76.0            |        74.0 - 78.0       |   76.0   |
