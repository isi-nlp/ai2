from pytorch_transformers import *

MODELS = [(BertModel, BertTokenizer, 'bert-base-cased'),
          (BertModel, BertTokenizer, 'bert-large-cased'),
          (OpenAIGPTModel, OpenAIGPTTokenizer, 'openai-gpt'),
          (GPT2Model, GPT2Tokenizer, 'gpt2'),
          (XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
          (XLNetModel, XLNetTokenizer, 'xlnet-large-cased'),
          (XLMModel, XLMTokenizer, 'xlm-mlm-en-2048'),
          (RobertaModel, RobertaTokenizer, 'roberta-base'),
          (RobertaModel, RobertaTokenizer, 'roberta-large')]

for model_class, tokenizer_class, pretrained_weights in MODELS:
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=False, cache_dir="./model_cache")
    model = model_class.from_pretrained(pretrained_weights, cache_dir="./model_cache")
