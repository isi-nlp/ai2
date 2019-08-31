import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import *
from ai2.utility import AI2DatasetHelper, AI2Dataset, collate_fn
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score
from typing import *
from functools import partial


class Classifier(pl.LightningModule):

    def __init__(self, config: Dict,
                 model_class: callable,
                 model_path: str,
                 tokenizer_class: callable,
                 tokenizer_path: str,
                 config_class: callable,
                 config_path: str,
                 batch_size: int = 64):
        super(Classifier, self).__init__()

        assert 'classes' in config, "Wrong config for Classifier, classes not found"

        self.config = config
        self.model = model_class.from_pretrained(model_path, cache_dir='./.cache')
        self.model_config = config_class.from_pretrained(config_path, cache_dir='./.cache')
        self.model.train()
        self.dropout = nn.Dropout(self.model_config.hidden_dropout_prob)
        self.linear = nn.Linear(self.model_config.hidden_size, 1)
        self.linear.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
        self.linear.bias.data.zero_()
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path, cache_dir='./.cache', do_lower_case=False)
        self.loss = nn.CrossEntropyLoss(reduction='sum')
        self.helper = AI2DatasetHelper(self.config)
        self.train_x, self.train_y, self.dev_x, self.dev_y = self.helper.download()
        self.batch_size = batch_size
        self.padding_index = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]

    def forward(self, x, token_type_ids, attention_mask):
        """
        Inputs:
            x: [batch_size(B), num_choice(C), squence_length(S)]
        Output:
            logits: [batch_size(B), num_choice(C)]
        """
        B, C, S = x.shape

        pooled_output = self.model(x.reshape((B*C, S)), token_type_ids.reshape((B*C, S)), attention_mask.reshape((B*C, S)))[1]     # [B*C, H]
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
                                              pred.cpu().detach().numpy().tolist()),
                                     requires_grad=False).to(x.device).reshape((1, 1)),
            'truth': y,
            'pred': pred}

    def validation_end(self, outputs):

        truth = torch.cat([x['truth'] for x in outputs], dim=0).reshape(-1)
        pred = torch.cat([x['pred'] for x in outputs], dim=0).reshape(-1)

        return {
            'val_acc': accuracy_score(truth.cpu().detach().numpy().tolist(), pred.cpu().detach().numpy().tolist()),
            'val_f1': f1_score(truth.cpu().detach().numpy().tolist(), pred.cpu().detach().numpy().tolist())
        }

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

    @pl.data_loader
    def tng_dataloader(self):
        # REQUIRED
        dataset = AI2Dataset(self.tokenizer, self.helper.preprocess(self.train_x, self.train_y))
        return DataLoader(dataset,
                          collate_fn=partial(collate_fn, padding_index=self.padding_index),
                          batch_size=self.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        dataset = AI2Dataset(self.tokenizer, self.helper.preprocess(self.dev_x, self.dev_y))
        return DataLoader(dataset,
                          collate_fn=partial(collate_fn, padding_index=self.padding_index),
                          batch_size=self.batch_size)
