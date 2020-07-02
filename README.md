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

## Dockerize

### Submission of Full CycIC on 2020-06-22

```bash
time DOCKER_BUILDKIT=1 docker build -t cycic-20200622 .
time docker run -it -v ${PWD}/data:/data -v ${PWD}/results:/results cycic-20200622 bash run_model.sh
time beaker image create --name cycic-20200622 cycic-20200622
```

### Submission of Full CycIC on 2020-07-02

```bash
time DOCKER_BUILDKIT=1 docker build --file dockerfiles/cycic_full_20200702 --tag cycic-full-20200702 .
time docker run -it -v ${PWD}/data:/data -v ${PWD}/results:/results cycic-full-20200702 bash run_model.sh
time beaker image create --name cycic-full-20200702 cycic-full-20200702
```

### Submission of 20% CycIC on 2020-07-02

```bash
time DOCKER_BUILDKIT=1 docker build --file dockerfiles/cycic_20p_20200702 --tag cycic-20p-20200702 .
time docker run -it -v ${PWD}/data:/data -v ${PWD}/results:/results cycic-20p-20200702 bash run_model.sh
time beaker image create --name cycic-20p-20200702 cycic-20p-20200702
```

### Submission of 10% CycIC on 2020-07-02

```bash
time DOCKER_BUILDKIT=1 docker build --file dockerfiles/cycic_10p_20200702 --tag cycic-10p-20200702 .
time docker run -it -v ${PWD}/data:/data -v ${PWD}/results:/results cycic-10p-20200702 bash run_model.sh
time beaker image create --name cycic-10p-20200702 cycic-10p-20200702
```

## Results

### PIQA

|        Model         | Bootstrapped Accuracy Mean | Bootstrapped Accuracy CI | Accuracy |
| :------------------: | :------------------------: | :----------------------: | :------: |
| Roberta large (SAGA) |            76.0            |       74.0 - 78.0        |   76.0   |
