"""
This python script downloads all the pretrained weights to the model_cache folder for future experiments.
"""

from pathlib import Path

import transformers as tf
from loguru import logger

# Retrieving the root path of the project folder
ROOT_PATH = Path(__file__).parent.parent.absolute()

# Define a list of all models and it's respective tokenizer (feel free to comment out lines for unwanted models)
MODELS = [
    (tf.BertModel, tf.BertTokenizer, 'bert-base-cased'),
    (tf.BertModel, tf.BertTokenizer, 'bert-large-cased'),
    (tf.RobertaModel, tf.RobertaTokenizer, 'roberta-base'),
    (tf.RobertaModel, tf.RobertaTokenizer, 'roberta-large'),
    (tf.DistilBertModel, tf.DistilBertTokenizer, 'distilbert-base-uncased'),
    (tf.OpenAIGPTModel, tf.OpenAIGPTTokenizer, 'openai-gpt'),
    (tf.GPT2Model, tf.GPT2Tokenizer, 'gpt2'),
    (tf.GPT2Model, tf.GPT2Tokenizer, 'gpt2-large'),
    (tf.XLNetModel, tf.XLNetTokenizer, 'xlnet-base-cased'),
    (tf.XLNetModel, tf.XLNetTokenizer, 'xlnet-large-cased'),
    (tf.XLMModel, tf.XLMTokenizer, 'xlm-mlm-en-2048'),
]

for model_class, tokenizer_class, pretrained_weights in MODELS:
    logger.info(f"Download model weights for {pretrained_weights}")

    # Initialize model and tokenizer to force download on pretrained weights
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=False,
                                                cache_dir=ROOT_PATH / "model_cache")
    model = model_class.from_pretrained(pretrained_weights, cache_dir=ROOT_PATH / "model_cache")

    # Delete the model to free up memory
    del model
    del tokenizer

    logger.success(f"Finish downloading model weights for {pretrained_weights}")
