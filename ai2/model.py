import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import *
from .utility import AI2DatasetHelper, AI2Dataset, collate_fn
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score
from typing import *
from functools import partial


class Classifier(pl.LightningModule):

    def __init__(self, config: Dict, model_class: callable, model_path: str, tokenizer_class: callable, tokenizer_path: str, d_model: int, batch_size: int = 64):
        super(Classifier, self).__init__()

        assert 'classes' in config, "Wrong config for Classifier, classes not found"

        self.config = config

        self.model = model_class.from_pretrained(model_path, cache_dir='./.cache')
        self.model.train()
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path, cache_dir='./.cache', do_lower_case=False)

        self.linear = nn.Linear(d_model, 1)
        self.loss = nn.CrossEntropyLoss(reduction='sum')

        self.helper = AI2DatasetHelper(self.config)
        self.train_x, self.train_y, self.dev_x, self.dev_y = self.helper.download()

        self.d_model = d_model
        self.batch_size = batch_size
        self.padding_index = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]

    def forward(self, x):
        """
        Inputs:
            x: [batch_size(B), num_choice(C), squence_length(S)]
        Output: 
            logits: [batch_size(B), num_choice(C)]
        """
        B, C, S = x.shape
        output = self.model(x.reshape((-1, x.size(-1))))[1]     # [B*C, H]
        output = output.reshape((B, C, -1))                     # [B, C, H]
        return torch.squeeze(self.linear(output), dim=-1)       # [B, C]

    def training_step(self, batch, batch_nb):
        x, y = batch['x'], batch['y']
        y_hat = self.forward(x)                 # [B, C]
        return {'loss': self.loss(y_hat, y)}

    def validation_step(self, batch, batch_nb):

        x, y = batch['x'], batch['y']
        y_hat = self.forward(x)
        pred = y_hat.argmax(dim=-1)

        # print(pred, y)

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


if __name__ == "__main__":

    from .utility import load_config
    from pytorch_lightning import Trainer
    from test_tube import Experiment

    exp = Experiment('./output')
    model = Classifier(load_config(
        "/Users/chenghaomou/Code/Code-ProjectsPyCharm/ai2/ai2/tasks.yaml",
    ), BertModel, 'bert-base-uncased', BertTokenizer, 'bert-base-uncased', 768, 2)
    trainer = Trainer(exp)
    trainer.fit(model)
