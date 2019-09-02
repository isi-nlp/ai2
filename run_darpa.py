import argparse
import torch
from ai2.utility import load_config
from pytorch_lightning import Trainer
from test_tube import Experiment
from pytorch_transformers import *
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from ai2.model import Classifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

MODELS = {
    'bert': BertModel,
    'gpt': OpenAIGPTModel,
    'transformerxl': TransfoXLModel,
    'gpt2': GPT2Model,
    'xlnet': XLNetModel,
    'roberta': RobertaModel,
    'xlm': XLMModel
}
CONFIGS = {
    'bert': BertConfig,
    'gpt': OpenAIGPTConfig,
    'transformerxl': TransfoXLConfig,
    'gpt2': GPT2Config,
    'xlnet': XLNetConfig,
    'roberta': RobertaConfig,
    'xlm': XLMConfig
}

TOKENIZERS = {
    'bert': BertTokenizer,
    'gpt': OpenAIGPTTokenizer,
    'transformerxl': TransfoXLTokenizer,
    'gpt2': GPT2Tokenizer,
    'xlnet': XLNetTokenizer,
    'roberta': RobertaTokenizer,
    'xlm': XLMTokenizer
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Run ai2 darpa tasks with pytorch-transformers')
    parser.add_argument('--task', '-t', choices=['anli', 'hellaswag', 'physicaliqa', 'socialiqa'],
                        help='DARPA task, see https://leaderboard.allenai.org/?darpa_offset=0', required=True)
    parser.add_argument('--train_config', help='Training config file', required=True)
    parser.add_argument('--model_type', choices=MODELS, help='Model type', required=True)
    parser.add_argument('--tokenizer_type', choices=TOKENIZERS, help='Tokenizer type', required=True)
    parser.add_argument('--model_config_type', choices=CONFIGS, help='Model configuration type', required=False)
    parser.add_argument('--model_weight', help='Model weight from huggingface', required=True)
    parser.add_argument('--tokenizer_weight', help='Pretrained tokenizer from huggingface', required=True)
    parser.add_argument('--model_config_weight', help='Predefined configuration', required=False)

    args = parser.parse_args()

    if args.model_config_weight is None or args.model_config_type is None:
        args.model_config_weight = args.model_weight
        args.model_config_type = args.model_type

    TASK = load_config("ai2/tasks.yaml", args.task)

    exp = Experiment(save_dir='./output', name=f"{args.task}-{args.model_weight}")
    model = Classifier(task_config=TASK,
                       train_config=load_config(args.train_config),
                       model_class=MODELS[args.model_type],
                       model_path=args.model_weight,
                       tokenizer_class=TOKENIZERS[args.tokenizer_type],
                       tokenizer_path=args.tokenizer_weight,
                       model_config_class=CONFIGS[args.model_config_type],
                       model_config_path=args.model_config_weight)
    trainer = Trainer(exp, early_stop_callback=EarlyStopping(monitor='val_f1', patience=10, mode='max'),
                      checkpoint_callback=ModelCheckpoint(
                          filepath=f'./{args.task}-{args.model_weight}-models', monitor='val_f1', save_best_only=True),
                      gradient_clip=1.0, cluster=None, process_position=0, current_gpu_name=0, nb_gpu_nodes=1,
                      gpus=[i for i in range(torch.cuda.device_count())],
                      show_progress_bar=True, overfit_pct=0.0, track_grad_norm=-1, check_val_every_n_epoch=1, fast_dev_run=True,
                      accumulate_grad_batches=1, max_nb_epochs=load_config(args.train_config)['max_epochs'],
                      min_nb_epochs=1, train_percent_check=1.0, val_percent_check=1.0, test_percent_check=1.0, val_check_interval=0.1,
                      log_save_interval=10, add_log_row_interval=10, distributed_backend='dp', use_amp=False, print_nan_grads=False,
                      print_weights_summary=False, amp_level='O2', nb_sanity_val_steps=5)
    trainer.fit(model)
