import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import *
from ai2.utils import AI2Dataset, AI2Preprocessor
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score


class Classifier(pl.LightningModule):

    def __init__(self, config, model_class, model_path, tokenizer_class, tokenizer_path, d_model, batch_size=64):
        super(Classifier, self).__init__()
        self.config = config
        self.model = model_class.from_pretrained(model_path, cache_dir='./.cache')
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path, cache_dir='./.cache', do_lower_case=False)
        assert 'classes' in self.config, "Wrong config for Classifier, classes not found"
        self.linear = nn.Linear(d_model, 1)
        self.loss = nn.CrossEntropyLoss(reduction='sum')

        self.preprocessor = AI2Preprocessor(self.config)
        self.train_x, self.train_y, self.dev_x, self.dev_y = self.preprocessor.download()
        self.d_model = d_model
        self.batch_size = batch_size

    def forward(self, x):
        """
        x has shape [batch_size(B), num_choice(C), squence_length(S)]
        """
        B, C, S = x.shape
        output = self.model(x.reshape((-1, x.size(-1))))[1]  # pooled context vectors: [B*C, H]
        output = output.reshape((B, C, -1))  # [B, C, H]
        return torch.squeeze(self.linear(output), dim=-1)  # [B, C]

    def training_step(self, batch, batch_nb):
        x, y = batch['x'], batch['y']
        y_hat = self.forward(x)  # [B, C]
        assert y_hat.size(0) == y.size(0), "Batch size mismatch during training!"
        return {'loss': self.loss(y_hat, y)}

    def validation_step(self, batch, batch_nb):

        x, y = batch['x'], batch['y']
        y_hat = self.forward(x)

        pred = y_hat.argmax(dim=-1)

        return {
            'batch_loss': (self.loss(y_hat, y)).reshape((1, 1)),
            'batch_acc': (((pred == y).sum()/y_hat.size(0)).float()).reshape((1, 1)),
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
        return torch.optim.AdamW(self.parameters(), lr=0.02)

    def collate_fn(self, samples):
        """
        Pad x with the longest example in the batch
        """
        C = samples[0]['x'].size(0)

        inputs = torch.zeros((len(samples), C, max(map(lambda x: x['x'].size(1), samples))), requires_grad=False)
        inputs += self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        targets = torch.zeros((len(samples)))
        for i, sample in enumerate(samples):
            inputs[i, :, :sample['x'].size(1)] = sample['x']
            targets[i] = sample['y']
        return {'x': inputs.long(), 'y': targets.long()}

    @pl.data_loader
    def tng_dataloader(self):
        # REQUIRED
        dataset = AI2Dataset(self.preprocessor.preprocess(self.train_x, self.train_y),
                             self.tokenizer, self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0])
        # dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return DataLoader(dataset,
                          #   sampler=dist_sampler,
                          collate_fn=self.collate_fn,
                          batch_size=self.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        dataset = AI2Dataset(self.preprocessor.preprocess(self.dev_x, self.dev_y),
                             self.tokenizer, self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0])
        # dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return DataLoader(dataset,
                          #   sampler=dist_sampler,
                          collate_fn=self.collate_fn,
                          batch_size=self.batch_size)


# if __name__ == "__main__":
#     from utils import TASKS
#     from pytorch_lightning import Trainer
#     from test_tube import Experiment

#     exp = Experiment('./output')
#     model = Classifier(TASKS['anli'], BertModel, 'bert-base-uncased', BertTokenizer, 'bert-base-uncased', 768)
#     trainer = Trainer(exp)
#     trainer.fit(model)
