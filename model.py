import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from transformers import AutoTokenizer, AutoModel
from transformers import AdamW


class MCQAModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args
        
        self.model = AutoModel.from_pretrained(self.hparams.pretrained_model_name)
        
        self.lin_out = nn.Linear(self.model.hidden_size, 1)
        self.dropout = nn.Dropout(self.model.hidden_dropout_prob)

    def forward(self, inputs):
        for key in ["input_ids", "attention_mask"]:
            inputs[key] = torch.cat([el.unsqueeze(0) for el in inputs[key]], dim=0)
            inputs[key] = inputs[key].transpose(0, 1).view(-1, inputs[key].size(-1))

        outputs = self.model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        outputs = self.dropout(outputs)
        outputs = self.lin_out(outputs)
        return outputs.view(-1, 4)

    def compute_acc(self, logits, labels):
        return torch.mean(logits.argmax(dim=1) == labels).cpu().numpy()

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch["label"]

        loss = F.cross_entropy(logits, labels)
        acc = self.compute_acc(logits, labels)

        self.log("loss", loss)
        self.log("acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch["label"]

        loss = F.cross_entropy(logits, labels)
        acc = self.compute_acc(logits, labels)

        self.log("val_loss", loss)
        self.log("val_acc", acc)

        return loss

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch["label"]

        loss = F.cross_entropy(logits, labels)
        acc = self.compute_acc(logits, labels)

        self.log("test_loss", loss)
        self.log("test_acc", acc)

        return loss
