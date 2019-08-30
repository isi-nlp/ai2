import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import *
from ai2.utils import AI2Dataset, AI2Preprocessor
from torch.nn import functional as F
from sklearn.metrics import f1_score


class Classifier(pl.LightningModule):

    def __init__(self, config, model_class, model_path, tokenizer_class, tokenizer_path, d_model, batch_size=64):

        super(Classifier, self).__init__()
        self.config = config
        self.model = model_class.from_pretrained(model_path, cache_dir='./.cache')
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path, cache_dir='./.cache', do_lower_case=False)
        assert 'classes' in self.config, "Wrong config for Classifier, classes not found"
        self.linear = nn.Linear(d_model, self.config['classes'])

        self.preprocessor = AI2Preprocessor(self.config)
        self.train_x, self.train_y, self.dev_x, self.dev_y = self.preprocessor.download()
        self.d_model = d_model
        self.batch_size = batch_size

    def forward(self, x):

        output = self.model(x)[1]  # pooled context vectors
        return torch.sigmoid(self.linear(output))

    def training_step(self, batch, batch_nb):
        x, y = batch['x'], batch['y']
        y_hat = self.forward(x)
        return {'loss': F.binary_cross_entropy(y_hat.reshape(-1), y.reshape(-1))}

    def validation_step(self, batch, batch_nb):
        # Must return tensors
        x, y = batch['x'], batch['y']
        y_hat = self.forward(x)

        return {
            'val_loss': (F.binary_cross_entropy(y_hat.reshape(-1), y.reshape(-1))).reshape((1, 1)),
            'val_acc': (((y_hat.reshape(-1) == y.reshape(-1)).sum()/y_hat.reshape(-1).size(0)).float()).reshape((1, 1)),
            'val_f1': torch.tensor(f1_score(y.reshape(-1).cpu().detach().numpy().tolist(),
                                            (y_hat.reshape(-1) >= 0.5).long().cpu().detach().numpy().tolist()
                                            ), requires_grad=False).to(x.device).reshape((1, 1)),
            'truth': y.reshape((1, -1)),
            'pred': y_hat.reshape((1, -1)).long()}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs], dim=-1).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs], dim=-1).mean()
        avg_f1 = torch.stack([x['val_f1'] for x in outputs], dim=-1).mean()

        truth = torch.stack([x['truth'] for x in outputs], dim=-1).reshape(-1)
        pred = torch.stack([x['pred'] for x in outputs], dim=-1).reshape(-1)

        return {'avg_val_loss': avg_loss,
                'avg_val_acc': avg_acc,
                'avg_val_f1': avg_f1,
                'val_f1': f1_score(truth.cpu().detach().numpy().tolist(), pred.cpu().detach().numpy().tolist())
                }

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.02)

    def collate_fn(self, samples):
        inputs = torch.zeros((len(samples), max(map(lambda x: len(x['x']), samples))))
        targets = torch.zeros((len(samples), 1))

        for i, sample in enumerate(samples):

            inputs[i, :len(sample['x'])] = torch.from_numpy(np.asarray(sample['x']))
            targets[i, :] = sample['y']

        return {'x': inputs.long(), 'y': targets}

    @pl.data_loader
    def tng_dataloader(self):
        # REQUIRED
        dataset = AI2Dataset(self.preprocessor.preprocess(self.train_x, self.train_y), self.tokenizer)
        # dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return DataLoader(dataset,
                          #   sampler=dist_sampler,
                          collate_fn=self.collate_fn,
                          batch_size=self.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        dataset = AI2Dataset(self.preprocessor.preprocess(self.dev_x, self.dev_y), self.tokenizer)
        # dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return DataLoader(dataset,
                          #   sampler=dist_sampler,
                          collate_fn=self.collate_fn,
                          batch_size=self.batch_size)


if __name__ == "__main__":
    from utils import TASKS
    from pytorch_lightning import Trainer
    from test_tube import Experiment

    exp = Experiment('./output')
    model = Classifier(TASKS['anli'], BertModel, 'bert-base-uncased', BertTokenizer, 'bert-base-uncased', 768)
    trainer = Trainer(exp)
    trainer.fit(model)
