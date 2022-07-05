import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Adafactor, AdamW

from datetime import datetime
import sys


class T5Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.hpar = args
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hpar.pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/unifiedqa-t5-large')
        return
        

    def forward(self, inputs):
        outputs = self.model(**inputs["input_seq"], 
                             labels=inputs["output_seq"]["input_ids"],
                             return_dict=True)
        return outputs


    def trim_padding(self, inputs):
        max_len_in = -1
        for seq in inputs["input_seq"]["attention_mask"]:
            for t in range(1, len(seq)):
                if seq[t] == 1:
                    max_len_in = max(t+1, max_len_in)
        if max_len_in == -1:
            max_len_in = inputs["input_seq"]["input_ids"].size(1)
        inputs["input_seq"]["input_ids"] = inputs["input_seq"]["input_ids"][:,:max_len_in].contiguous()
        inputs["input_seq"]["attention_mask"] = inputs["input_seq"]["attention_mask"][:,:max_len_in].contiguous()

        max_len_out = -1
        for seq in inputs["output_seq"]["attention_mask"]:
            for t in range(1, len(seq)):
                if seq[t] == 1:
                    max_len_out = max(t+1, max_len_out)
        if max_len_out == -1:
            max_len_out = inputs["output_seq"]["input_ids"].size(1)
        inputs["output_seq"]["input_ids"] = inputs["output_seq"]["input_ids"][:,:max_len_out].contiguous()
        inputs["output_seq"]["attention_mask"] = inputs["output_seq"]["attention_mask"][:,:max_len_out].contiguous()
        return inputs


    def generate(self, **inputs):
        return self.model.generate(**inputs, max_length=self.hpar.max_len_out)


    def compute_acc(self, logits, labels):
        return torch.mean((logits.argmax(dim=2) == labels).float())


    def configure_optimizers(self):
        optimizer = Adafactor(self.model.parameters(),
                              lr=self.hpar.lr,
                              eps=(1e-30, 1e-3),
                              clip_threshold=1.0,
                              decay_rate=-0.8,
                              beta1=None,
                              weight_decay=0.0,
                              scale_parameter=False,
                              relative_step=False,
                              warmup_init=False)
        return optimizer


    def training_step(self, batch, batch_idx):
        batch = self.trim_padding(batch)

        outputs = self(batch)

        labels = batch["output_seq"]["input_ids"]
        logits = outputs.logits
        loss = outputs.loss

        acc = self.compute_acc(logits, labels)

        self.log("train_loss", loss, sync_dist=True)
        self.log("train_acc", acc, sync_dist=True)

        return loss

    def training_epoch_end(self, outputs):
        print("epoch end time = {}".format(datetime.now().strftime("%d/%m/%Y %H:%M")), file=sys.stderr)
        return

    def validation_step(self, batch, batch_idx):
        batch = self.trim_padding(batch)
        print(batch)

        outputs = self(batch)

        labels = batch["output_seq"]["input_ids"]
        logits = outputs.logits
        loss = outputs.loss

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
        batch = self.trim_padding(batch)

        outputs = self(batch)

        labels = batch["output_seq"]["input_ids"]
        logits = outputs.logits
        loss = outputs.loss

        acc = self.compute_acc(logits, labels)

        return {"test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        self.log("test_loss", torch.stack([it["test_loss"] for it in outputs]).mean())
        self.log("test_acc", torch.stack([it["test_acc"] for it in outputs]).mean())
