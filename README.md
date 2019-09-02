# ai2

Framework for testing models with AI2 leaderboards, integrated with all kinds of fancy features supported by pytorch-lightning, such as multi GPU training/tensorboard support, and most importantly, readability!

## 1. Quick start

```bash
python run_darpa.py --help

python run_darpa.py                         \
    --task anli                             \
    --model_type bert                       \
    --model_weight bert-base-cased          \
    --tokenizer_type bert                   \
    --tokenizer_weight bert-base-cased      \
    --model_config_type bert                \
    --model_config_weight bert-base-cased   \
    --train_config ai2/base-task.yaml
```

**If you are using HPC nodes, you should run this script first on your login node to cache the task dataset first.**

### 1.1 Task

One of the four tasks from ai2 leaderboards:

1. `anli`: _Abductive Natural Language Inference (aNLI)_
2. `hellaswag`: _HellaSwag: Can a Machine Really Finish Your Sentence?_
3. `physicaliqa`: _Physical IQa: Physical Interaction QA_
4. `socialiqa`: _Social IQA: Social Interaction QA_

### 1.2 Model/Tokenizer/Weight type

Seven models supported by huggingface:

1. `bert`: BERT (from Google) released with the paper BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
2. `gpt`: GPT (from OpenAI) released with the paper Improving Language Understanding by Generative Pre-Training by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.
3. `gpt2`: GPT-2 (from OpenAI) released with the paper Language Models are Unsupervised Multitask Learners by Alec Radford*, Jeffrey Wu*, Rewon Child, David Luan, Dario Amodei** and Ilya Sutskever**.
4. `transformerxl`: Transformer-XL (from Google/CMU) released with the paper Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context by Zihang Dai*, Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.
5. `xlnet`: XLNet (from Google/CMU) released with the paper ​XLNet: Generalized Autoregressive Pretraining for Language Understanding by Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
6. `xlm`: XLM (from Facebook) released together with the paper Cross-lingual Language Model Pretraining by Guillaume Lample and Alexis Conneau.
7. `roberta`: RoBERTa (from Facebook), released together with the paper a Robustly Optimized BERT Pretraining Approach by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.

**Model type** means one of the seven models while **model weight** means different pretrained model weight like `bert-base-cased`. The tokenizer/tokenizer_weight normally take the same type/weight. **Model config** is the huggingface's config class/weight which tells some superparameters about a model and the training settings, like batch size or learning rate. Please refer to huggingface's [repo](https://github.com/huggingface/pytorch-transformers) for more information.

### 1.3 Train config

You can specify your own training config file with YAML for batch size, learning rate, max sequence length, or max epochs.
