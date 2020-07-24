# @Date    : 2019-10-06 09:17:36
# @Author  : Chenghao Mou (chengham@isi.edu)
# @Link    : https://github.com/ChenghaoMou/ai2

from pathlib import Path

from transformers import *
from loguru import logger

#
# Script used to download transformer models with their pre-trained weights

# Define all models and their weights that we are interested in, uncomment models you want to download weights for.
MODELS = [
    (BertModel, BertTokenizer, 'bert-base-cased'),
    # (BertModel, BertTokenizer, 'bert-large-cased'),
    # (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
    # (OpenAIGPTModel, OpenAIGPTTokenizer, 'openai-gpt'),
    # (GPT2Model, GPT2Tokenizer, 'gpt2'),
    # (GPT2Model, GPT2Tokenizer, 'gpt2-large'),
    # (XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
    # (XLNetModel, XLNetTokenizer, 'xlnet-large-cased'),
    # (XLMModel, XLMTokenizer, 'xlm-mlm-en-2048'),
    # (RobertaModel, RobertaTokenizer, 'roberta-base'),
    (RobertaModel, RobertaTokenizer, 'roberta-large'),
]

# Find the absolute root path
root_dir = Path().absolute()

# For each model and tokenizer, load the pretrained_weights and put the downloads to cache_dir
for model_class, tokenizer_class, pretrained_weights in MODELS:
    logger.info(f"Download model weights for {pretrained_weights}")

    # Load the model and tokenizer - this forces the library to download pre-trained weights and put them in model_cache
    model = model_class.from_pretrained(pretrained_weights, cache_dir=root_dir / "model_cache")
    tokenizer = \
        tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=False, cache_dir=root_dir / "model_cache")

    # Delete the model and tokenizer to save memory
    del model
    del tokenizer
