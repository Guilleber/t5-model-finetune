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
        self.hpar = args
        
        self.model = AutoModel.from_pretrained(self.hpar.pretrained_model_name)
        
        self.lin_out = nn.Linear(self.model.config.hidden_size, 1)
        self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)

    def forward(self, inputs):
        for key in ["input_ids", "attention_mask"]:
            inputs[key] = torch.cat([el.unsqueeze(0) for el in inputs[key]], dim=0)
            inputs[key] = inputs[key].transpose(0, 1).contiguous().view(-1, inputs[key].size(-1))

        outputs = self.model(inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]
        outputs = outputs[:,0,:]
        outputs = self.dropout(outputs)
        outputs = self.lin_out(outputs)
        return outputs.view(-1, 4)

    def compute_acc(self, logits, labels):
        return torch.mean((logits.argmax(dim=1) == labels).float())

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hpar.lr)

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch["label"]

        loss = F.cross_entropy(logits, labels)
        acc = self.compute_acc(logits, labels)

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch["label"]

        loss = F.cross_entropy(logits, labels)
        acc = self.compute_acc(logits, labels)

        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([it["val_loss"] for it in outputs]).mean()
        mean_acc = torch.stack([it["val_acc"] for it in outputs]).mean()

        print("*** VALIDATION ***")
        print("val_loss: " + str(mean_loss))
        print("val_acc: " + str(mean_acc))

        self.log("val_loss", mean_loss)
        self.log("val_acc", mean_acc)

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch["label"]

        loss = F.cross_entropy(logits, labels)
        acc = self.compute_acc(logits, labels)

        return {"test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        self.log("test_loss", torch.stack([it["test_loss"] for it in outputs]).mean())
        self.log("test_acc", torch.stack([it["test_acc"] for it in outputs]).mean())
