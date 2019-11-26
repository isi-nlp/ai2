# 1. AI2 DARPA TASKS Eval Tool

- [1. AI2 DARPA TASKS Eval Tool](#1-ai2-darpa-tasks-eval-tool)
  - [1.1. Environment](#11-environment)
  - [1.2. Baseline models](#12-baseline-models)
  - [1.3. Baseline Scores](#13-baseline-scores)
  - [1.4. Fine-tuning Time Reference](#14-fine-tuning-time-reference)
  - [1.5. Run Your Own Model](#15-run-your-own-model)
    - [1.5.1. Necessary Implementation](#151-necessary-implementation)
    - [1.5.2. Download the datasets](#152-download-the-datasets)
    - [1.5.3. Optional: HuggingFace Pretrained Models](#153-optional-huggingface-pretrained-models)
    - [1.5.4. Fine Tune the Model](#154-fine-tune-the-model)
    - [1.5.5. Eval the model](#155-eval-the-model)
    - [1.5.6. Visualize your model [Optional]](#156-visualize-your-model-optional)

## 1.1. Environment

- python >= 3.7
- `pip install -r requirements.txt`

## 1.2. Baseline models

| Models        | Size     | Category    |
| ------------- | -------- | ----------- |
| Bert base     | 110 M    | base        |
| Bert large    | 340 M    | large       |
| openai gpt    | 110 M    | base        |
| GPT2          | 117 M    | weird large |
| XLM           | >= 295 M | super large |
| XLnet         | 110 M    | base        |
| XLNet large   | 340 M    | large       |
| roberta       | 125 M    | base        |
| roberta large | 355 M    | large       |
| distilbert    | 60 M     | small       |

It is impossible to fit super large models in P100s on HPC. Weird large models are base models eating memory like a large one.

## 1.3. Baseline Scores

| Models                               | aNLI      | hellaswag | piqa      | siqa      | Config Commit                                                                              |
| ------------------------------------ | --------- | --------- | --------- | --------- | ------------------------------------------------------------------------------------------ |
| Bert (bert-base-cased)               | 63.32     | 37.83     | 65.29     | 60.33     | [commit](https://github.com/ChenghaoMou/ai2/tree/4729f25627281752b6f662f36b53ca6bddd606fa) |
| Bert (bert-large-cased)              | 66.28     | 43.84     | 68.67     | 65        | [commit](https://github.com/ChenghaoMou/ai2/tree/4729f25627281752b6f662f36b53ca6bddd606fa) |
| RoBERTa (roberta-base)               | 71.54     | 58.51     | 48.03     | 69.09     | [commit](https://github.com/ChenghaoMou/ai2/tree/4729f25627281752b6f662f36b53ca6bddd606fa) |
| RoBERTa (roberta-large)              | **84.39** | **82.42** | **76.96** | **77.12** | [commit](https://github.com/ChenghaoMou/ai2/tree/4729f25627281752b6f662f36b53ca6bddd606fa) |
| XLNet (xlnet-base-cased)             | 68.15     | 52.99     | 52.94     | 65.79     | [commit](https://github.com/ChenghaoMou/ai2/tree/4729f25627281752b6f662f36b53ca6bddd606fa) |
| XLNet (xlnet-large-cased)            | 80.16     | 80.38     | 69.27     | 75.23     | [commit](https://github.com/ChenghaoMou/ai2/tree/4729f25627281752b6f662f36b53ca6bddd606fa) |
| GPT (openai-gpt)                     | 64.23     | 38.15     | 67.11     | 61.73     | [commit](https://github.com/ChenghaoMou/ai2/tree/4729f25627281752b6f662f36b53ca6bddd606fa) |
| GPT2 (gpt2)                          | 53.46     | 26.52     | 48.05     | 35.16     | [commit](https://github.com/ChenghaoMou/ai2/tree/4729f25627281752b6f662f36b53ca6bddd606fa) |
| DistilBERT (distilbert-base-uncased) | 60.17     | 35.57     | 64.96     | 52.92     | [commit](https://github.com/ChenghaoMou/ai2/tree/4729f25627281752b6f662f36b53ca6bddd606fa) |

## 1.4. Fine-tuning Time Reference

With two P100s on HPC, it takes the following time to fine tune a model.

|    Tasks    | Base Model(3 epochs) | Large Model(3 epochs) |
| :---------: | :------------------: | :-------------------: |
|    aNLI     |      1 ~ 2 hrs       |        ~ 7 hrs        |
|  hellaswag  |      6 ~ 8 hrs       |        24 hrs         |
| physicaliqa |         1 hr         |       3 ~ 4 hrs       |
|  socialiqa  |         1 hr         |       4 ~ 5 hrs       |

## 1.5. Run Your Own Model

### 1.5.1. Necessary Implementation

1. Implement your own ModelLoader, TokenizerLoader, and Classifier, similar to those implemented in `huggingface.py`.
2. Change the classifier in both `test.py` as `train.py`

In step 1, when you follow the interface of a `textbook.interface.ModelLoader` or a `textbook.interface.TokenizerLoader`, you will use a `model_type` and `model_weight`
to direct your `load` function to find your model files.

Be sure to make modifications to test_dataset.py (change the tokenizer loader) and run your tokenizer on those datasets!

Later, you can specify the training parameters in `config/hyparams.yaml` in the following format:

```yaml
$task_name:
  $model_type:
    $model_weight:
      seed: 42
      lr: 2e-5
      dropout: 0.1
      batch_size: 32
      max_seq_len: 128
      max_nb_epochs: 3
      initializer_range: 0.02
      weight_decay: 0.0
      warmup_steps: 0
      adam_epsilon: 1e-8
      accumulate_grad_batches: 1
```

You can also specify default parameters within each task.

### 1.5.2. Download the datasets

You can use git lfs to pull down all datasets into `./cache` directory.

```bash
git lfs pull
```

### 1.5.3. Optional: HuggingFace Pretrained Models

You can also download all the pretrained models from HuggingFace by

```bash
python model_cache.py
```

### 1.5.4. Fine Tune the Model

Running train/eval should be straightforward.

```bash
$PYTHON -W ignore train.py --model_type $MODEL_TYPE --model_weight $MODEL_WEIGHT \
  --task_config_file config/tasks.yaml \
  --running_config_file config/hyparams.yaml \
  --task_name $TASK_NAME \
  --task_cache_dir ./cache \
  --output_dir output/$MODEL_TYPE-$MODEL_WEIGHT-$TASK_NAME-pred \
  --log_save_interval 25 --row_log_interval 25
```

Here, $MODEL_TYPE & $MODEL_WEIGHT are names you used to implement your model loader and tokenizer loader; \$TASK_NAME is one of four tasks `{alphanli, hellaswag, physicaliqa, socialiqa}`

During training, it also generates predictions for eval dataset in `output_dir`.

```shell
dev-probabilities.lst
dev-predictions.lst
dev-labels.lst
```

### 1.5.5. Eval the model

```bash
$PYTHON -W ignore test.py --model_type $MODEL_TYPE \
  --model_weight $MODEL_WEIGHT \
  --task_name $TASK_NAME \
  --task_config_file config/tasks.yaml \
  --task_cache_dir ./cache \
  --running_config_file config/hyparams.yaml \
  --test_input_dir ./cache/$TASK_NAME-$DATA-input \
  --output_dir output/$MODEL_TYPE-$MODEL_WEIGHT-$TASK_NAME-$DATA-pred \
  --weights_path output/$MODEL_TYPE-$MODEL_WEIGHT-checkpoints/$TASK_NAME/0/_ckpt_epoch_3.ckpt \
  --tags_csv output/$MODEL_TYPE-$MODEL_WEIGHT-log/$TASK_NAME/version_0/meta_tags.csv
```

It should generates all the predcitions in the `output_dir`:

```shell
probabilities.lst
predictions.lst
```

### 1.5.6. Visualize your model [Optional]

With integrated gradients, you can now visualize your model's token-level focus with Captum. Please refer to [Website](https://captum.io) for installation instructions.

In order to run visualization, you have to change some code `${MODEL}Embeddings`in the model you want to run in transformers:

````python
 def forward(self, input_ids, token_type_ids=None, position_ids=None):
        bc, seq_length, *_ = input_ids.shape # change this
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand((bc, seq_length)) # change this
        if token_type_ids is None:
            token_type_ids = torch.zeros((bc, seq_length))

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
```

The `--embedding_layer` parameter is the name of the word embedding layer in your model.

```bash
PYTHON=/Users/chenghaomou/Anaconda/envs/Elisa/bin/python
EVAL=gradient_visual.py
````

`bin/unittest-vis.sh`

```bash
$PYTHON -W ignore $EVAL --model_type distilbert \
  --model_weight distilbert-base-uncased \
  --task_name physicaliqa \
  --task_config_file config/tasks.yaml \
  --task_cache_dir ./cache \
  --running_config_file config/hyparams.yaml \
  --test_input_dir ./cache/physicaliqa-test-input \
  --output_dir output/distilbert-distilbert-base-uncased-physicaliqa-test-pred \
  --weights_path output/distilbert-distilbert-base-uncased-checkpoints/physicaliqa/0/_ckpt_epoch_5.ckpt \
  --tags_csv output/distilbert-distilbert-base-uncased-log/physicaliqa/version_0/meta_tags.csv \
  --embedding_layer encoder.model.embeddings.word_embeddings \
  --output visualization.html
```


## Submit your own model

Before you submit, you need to register with beaker and ai2 and contact ai2 for submission access. You can find details on how to contact them on the leaderboard.

Build a docker image. Make sure everything is self-contained in the code root directory. e.g. Everything including all the checkpoints in output will be packed into the docker image, except the dot directories (caches) and modify the submit/*.sh scripts you see fit.


```bash
docker build -t ${IMAGE_NAME} -f dockers/Dockerfile .
```


Tag your image
```bash
docker tag ${IMAGE_ID} ${USERNAM}/${REPO}:${TAG}
```

Setting beaker cli on your machine should be easy, just follow the official tutorials.

Uploaded it to beaker
```bash
beaker image create --name ${NAMEYOURMODEL} ${USERNAM}/${REPO}:${TAG}
```

Create the submission from the leaderboard!
