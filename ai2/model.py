import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import inspect
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import *
from ai2.utility import AI2DatasetHelper, AI2Dataset, collate_fn
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score
from typing import *
from functools import partial
from loguru import logger


def forward(model, input_ids, token_type_ids, attention_mask):
    """Ignore some of the inputs according to model's function signature."""
    spec = inspect.getargspec(model.forward)

    args = {'input_ids': input_ids}
    if 'token_type_ids' in spec:
        args['token_type_ids'] = token_type_ids
    if 'attention_mask' in spec:
        args['attention_mask'] = attention_mask

    return model.forward(**args)


class Classifier(pl.LightningModule):

    def __init__(self,
                 task_config: Dict,
                 train_config: Dict,
                 model_class: callable,
                 model_path: str,
                 tokenizer_class: callable,
                 tokenizer_path: str,
                 model_config_class: callable,
                 model_config_path: str):
        super(Classifier, self).__init__()

        assert 'classes' in task_config, "Wrong config for Classifier, classes not found"

        self.task_config = task_config
        self.train_config = train_config

        self.model = model_class.from_pretrained(model_path, cache_dir='./.cache')
        self.model_config = model_config_class.from_pretrained(model_config_path, cache_dir='./.cache')
        self.model.train()

        dropout = self.model_config.__dict__.get(
            'hidden_dropout_prob',
            self.model_config.__dict__.get(
                'resid_pdrop',
                self.model_config.__dict__.get(
                    'dropout',
                    0.1
                )
            )
        )

        hidden_size = self.model_config.__dict__.get(
            'hidden_size',
            self.model_config.__dict__.get(
                'n_embd',
                self.model_config.__dict__.get(
                    'd_model',
                    self.model_config.__dict__.get(
                        'emb_dim',
                        None
                    )
                )
            )
        )

        initializer_range = self.model_config.__dict__.get(
            'init_range',
            self.model_config.__dict__.get(
                'initializer_range',
                0.02

            )
        )

        logger.debug(f'Dropout: {dropout}')
        logger.info(f'Hidden size: {hidden_size}')
        logger.info(f'Initializer range: {initializer_range}')

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)
        self.linear.weight.data.normal_(mean=0.0, std=initializer_range)
        self.linear.bias.data.zero_()

        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path, cache_dir='./.cache', do_lower_case=False)

        self.loss = nn.CrossEntropyLoss(reduction='sum')
        self.helper = AI2DatasetHelper(self.task_config)
        self.train_x, self.train_y, self.dev_x, self.dev_y = self.helper.download()
        self.batch_size = self.train_config['batch_size']

        self.index = 1 if 'bert' in model_class.__class__.__name__ else 0

    def forward(self, x, token_type_ids, attention_mask):
        """
        Inputs:
            x: [batch_size(B), num_choice(C), squence_length(S)]
        Output:
            logits: [batch_size(B), num_choice(C)]
        """
        B, C, S = x.shape

        pooled_output = forward(self.model,
                                input_ids=x.reshape((B*C, S)),
                                token_type_ids=token_type_ids.reshape((B*C, S)),
                                attention_mask=attention_mask.reshape((B*C, S)))[self.index]    # [B*C, H]
        if len(pooled_output.shape) == 3:
            pooled_output = pooled_output.mean(dim=1)  # [B*C, S, H] => [B*C, H]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        reshaped_logits = logits.view(-1, C)

        return reshaped_logits

    def training_step(self, batch, batch_nb):
        x, y, token_type_ids, attention_mask = batch['x'], batch['y'], batch['token_type_ids'], batch['attention_mask']
        y_hat = self.forward(x, token_type_ids, attention_mask)                 # [B, C]
        return {'loss': self.loss(y_hat, y)}

    def validation_step(self, batch, batch_nb):

        x, y, token_type_ids, attention_mask = batch['x'], batch['y'], batch['token_type_ids'], batch['attention_mask']
        y_hat = self.forward(x, token_type_ids, attention_mask)
        pred = y_hat.argmax(dim=-1)

        return {
            'batch_loss': self.loss(y_hat, y).reshape((1, 1)),
            'batch_acc': ((pred == y).sum()/y_hat.size(0)).reshape((1, 1)),
            'batch_f1': torch.tensor(f1_score(y.reshape(-1).cpu().detach().numpy().tolist(),
                                              pred.cpu().detach().numpy().tolist(), average='micro'),
                                     requires_grad=False).to(x.device).reshape((1, 1)),
            'truth': y,
            'pred': pred}

    def validation_end(self, outputs):

        truth = torch.cat([x['truth'] for x in outputs], dim=0).reshape(-1)
        pred = torch.cat([x['pred'] for x in outputs], dim=0).reshape(-1)

        return {
            'val_acc': accuracy_score(truth.cpu().detach().numpy().tolist(), pred.cpu().detach().numpy().tolist()),
            'val_f1': f1_score(truth.cpu().detach().numpy().tolist(), pred.cpu().detach().numpy().tolist(), average='micro')
        }

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=float(self.train_config['lr']))

    @pl.data_loader
    def tng_dataloader(self):
        # REQUIRED
        dataset = AI2Dataset(tokenizer=self.tokenizer,
                             examples=self.helper.preprocess(self.train_x, self.train_y),
                             max_sequence_length=self.train_config['max_sequence_length'])
        return DataLoader(dataset,
                          collate_fn=partial(collate_fn, padding_index=self.tokenizer.convert_tokens_to_ids([dataset.pad_token])[0]),
                          batch_size=self.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        dataset = AI2Dataset(tokenizer=self.tokenizer,
                             examples=self.helper.preprocess(self.dev_x, self.dev_y),
                             max_sequence_length=self.train_config['max_sequence_length'])
        return DataLoader(dataset,
                          collate_fn=partial(collate_fn, padding_index=self.tokenizer.convert_tokens_to_ids([dataset.pad_token])[0]),
                          batch_size=self.batch_size)
