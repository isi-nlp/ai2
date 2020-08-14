# Minimal Code Base For AI2 Commonsense Leaderboard

## Datasets

- αNLI
  - https://leaderboard.allenai.org/anli/submissions/about
  - https://arxiv.org/abs/1908.05739
- HellaSwag
  - https://leaderboard.allenai.org/hellaswag/submissions/about
  - https://rowanzellers.com/hellaswag/
  - https://arxiv.org/abs/1905.07830
- PIQA
  - https://leaderboard.allenai.org/physicaliqa/submissions/about
  - https://yonatanbisk.com/piqa/
  - https://arxiv.org/abs/1911.11641
- SIQA
  - https://leaderboard.allenai.org/socialiqa/submissions/about
  - https://maartensap.github.io/social-iqa/
  - https://arxiv.org/abs/1904.09728

## Dependencies

Create and run a virtual environment with Python 3.7. If you're using conda, make sure to use conda version `>=4.8.2`.

```bash
conda create --name ai2_stable python=3.7
conda activate ai2_stable
```

Then run:

```bash
pip install -r requirements.txt
```

## Train

The main code to train a model is in `train.py`. It loads the configuration file `config/train.yaml` and outputs all the logs/checkpoints in `outputs`.

To submit it as a job on SAGA cluster, you should be able to simply run:

```bash
sbatch slurm/run_saga.sh
```

## Eval

### Get predictions without evaluation

```bash
python eval.py \
    --input_x task_data/physicaliqa-train-dev/dev.jsonl \
    --config config.yaml \
    --checkpoint outputs/path_to_checkpoint/_ckpt_epoch_4.ckpt \
    --output pred.lst
```

### Get predictions with evaluation(accuracy, confidence interval)

```bash
python eval.py \
    --input_x task_data/physicaliqa-train-dev/dev.jsonl \
    --config config.yaml \
    --checkpoint outputs/path_to_checkpoint/_ckpt_epoch_4.ckpt \
    --input_y task_data/physicaliqa-train-dev/dev-labels.lst \
    --output pred.lst
```

## Ensembling using Pegasus

### Setup

The ensembling workflow is defined and run using the Pegasus workflow management system. To run the
workflow, you'll need to install the [Pegasus wrapper][pegasus_wrapper].

Note that before running you'll need to set up your user-specific parameters file,
`parameters/root.params`. See `parameters/root.sample.params` for an example.

### Running the workflow

Once the wrapper is installed, generate the workflow:

```bash
python ai2/pegasus.py parameters/pegasus.params
```

Then submit the workflow:

```bash
cd path/to/experiment_root/ensemble
sh submit.sh
```

[pegasus_wrapper]: https://github.com/isi-vista/vista-pegasus-wrapper/

## Results

### PIQA

|        Model         | Bootstrapped Accuracy Mean | Bootstrapped Accuracy CI | Accuracy |
| :------------------: | :------------------------: | :----------------------: | :------: |
| Roberta large (SAGA) |            76.0            |       74.0 - 78.0        |   76.0   |
