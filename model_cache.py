from pathlib import Path
from transformers import *
from loguru import logger

if __name__ == "__main__":

    root_dir = Path().absolute()

    MODELS = [(BertModel, BertTokenizer, 'bert-base-cased'),
              (BertModel, BertTokenizer, 'bert-large-cased'),
              (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased')
              (OpenAIGPTModel, OpenAIGPTTokenizer, 'openai-gpt'),
              (GPT2Model, GPT2Tokenizer, 'gpt2'),
              (GPT2Model, GPT2Tokenizer, 'gpt2-large'),
              (XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
              (XLNetModel, XLNetTokenizer, 'xlnet-large-cased'),
              (XLMModel, XLMTokenizer, 'xlm-mlm-en-2048'),
              (RobertaModel, RobertaTokenizer, 'roberta-base'),
              (RobertaModel, RobertaTokenizer, 'roberta-large'),
              ]

    for model_class, tokenizer_class, pretrained_weights in MODELS:
        logger.info(f"Download model weights for {pretrained_weights}")

        tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=False, cache_dir=root_dir / "model_cache")
        model = model_class.from_pretrained(pretrained_weights, cache_dir=root_dir / "model_cache")

        del model
        del tokenizer
