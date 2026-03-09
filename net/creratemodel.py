from copy import deepcopy
import pytorch_lightning as pl
import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from torchmetrics import Accuracy, Dice
from torchmetrics.classification import BinaryJaccardIndex

from net.model import TextGuidedSegmentationModel


class CreateModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = TextGuidedSegmentationModel(args.bert_type, args.vision_type)
        self.lr = args.lr
        self.history = {}

        self.loss_fn = DiceCELoss(sigmoid=True)

        metrics_dict = {
            "acc": Accuracy(task="binary"),
            "dice": Dice(),
            "MIoU": BinaryJaccardIndex(),
        }
        self.train_metrics = nn.ModuleDict(metrics_dict)
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def forward(self, x):
        return self.model(x)  

    def shared_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)  

        if y.ndim == 4 and y.shape[1] != logits.shape[1]:
            y = y[:, 0:1, ...]

        if y.dtype != torch.float32:
            y = y.float()

        y_bin = (y > 0.5).float()

        loss = self.loss_fn(logits, y_bin)
        probs = torch.sigmoid(logits)

        y_int = y_bin.long()
        return {"loss": loss, "preds": probs.detach(), "y": y_int.detach()}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x = batch[0]
        else:
            x = batch
        logits = self(x)
        return torch.sigmoid(logits)

    def shared_step_end(self, outputs, stage):
        metrics = self.train_metrics if stage == "train" else (self.val_metrics if stage == "val" else self.test_metrics)

        for name in metrics:
            step_metric = metrics[name](outputs["preds"], outputs["y"]).item()
            if stage == "train":
                self.log(name, step_metric, prog_bar=True)

        return outputs["loss"].mean()

    def training_step_end(self, outputs):
        return {"loss": self.shared_step_end(outputs, "train")}

    def validation_step_end(self, outputs):
        return {"val_loss": self.shared_step_end(outputs, "val")}

    def test_step_end(self, outputs):
        return {"test_loss": self.shared_step_end(outputs, "test")}

    def shared_epoch_end(self, outputs, stage="train"):
        metrics = self.train_metrics if stage == "train" else (self.val_metrics if stage == "val" else self.test_metrics)

        key = "loss" if stage == "train" else (stage + "_loss")
        stage_loss = torch.stack([t[key] for t in outputs]).mean().item()

        epoch = self.trainer.current_epoch
        dic = {"epoch": epoch, stage + "_loss": stage_loss}

        for name in metrics:
            epoch_metric = metrics[name].compute().item()
            metrics[name].reset()
            dic[stage + "_" + name] = epoch_metric

        if stage in ("val", "test"):
            dic[stage + "_score"] = 0.5 * dic[stage + "_dice"] + 0.5 * dic[stage + "_MIoU"]

        if stage != "test":
            self.history[epoch] = dict(self.history.get(epoch, {}), **dic)
        return dic

    def training_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="train")
        self.print(dic)
        dic.pop("epoch", None)
        self.log_dict(dic, logger=True)

    def validation_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="val")
        self.print(dic)
        dic.pop("epoch", None)
        self.log_dict(dic, logger=True)

    def test_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="test")
        self.print(dic)
        dic.pop("epoch", None)
        self.log_dict(dic, logger=True)
