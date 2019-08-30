import argparse
import torch
from ai2.utils import TASKS
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
    parser.add_argument('--model_type', choices=MODELS, help='Model type', required=True)
    parser.add_argument('--tokenizer_type', choices=TOKENIZERS, help='Tokenizer type', required=True)
    parser.add_argument('--model_weight', help='Model weight from huggingface', required=True)
    parser.add_argument('--tokenizer_weight', help='Pretrained tokenizer from huggingface', required=True)
    parser.add_argument('--d_model', type=int, help='Hidden dimension of the model')
    parser.add_argument('--batch_size', type=int, help='Batch size')

    args = parser.parse_args()

    exp = Experiment(save_dir='./output', name=f"{args.task}-{args.model_weight}")
    model = Classifier(TASKS[args.task], MODELS[args.model_type], args.model_weight,
                       TOKENIZERS[args.tokenizer_type], args.tokenizer_weight, args.d_model, batch_size=args.batch_size)
    trainer = Trainer(exp,
                      early_stop_callback=EarlyStopping(monitor='val_f1', patience=10, mode='max'),
                      checkpoint_callback=ModelCheckpoint(filepath='./models', monitor='val_f1', save_best_only=True),
                      gradient_clip=0,
                      cluster=None,
                      process_position=0,
                      current_gpu_name=0,
                      nb_gpu_nodes=1,
                      gpus=[i for i in range(torch.cuda.device_count())],
                      show_progress_bar=True,
                      overfit_pct=0.0,
                      track_grad_norm=-1,
                      check_val_every_n_epoch=1,
                      fast_dev_run=False,
                      accumulate_grad_batches=2,
                      max_nb_epochs=1000,
                      min_nb_epochs=1,
                      train_percent_check=1.0,
                      val_percent_check=1.0,
                      test_percent_check=1.0,
                      val_check_interval=0.05,
                      log_save_interval=100,
                      add_log_row_interval=10,
                      distributed_backend='dp',
                      use_amp=False,
                      print_nan_grads=False,
                      print_weights_summary=False,
                      amp_level='O2',
                      nb_sanity_val_steps=5)
    trainer.fit(model)
