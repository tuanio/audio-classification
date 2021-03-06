import torch
from torch import nn
import pytorch_lightning as pl
from torchvision.models.alexnet import AlexNet


class ModelModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.5,
        lr: float = 0.01,
        optim_configs: dict = {}
    ):
        super().__init__()
        self.alexnet = AlexNet(num_classes=num_classes, dropout=dropout)
        self.lr = lr
        self.optim_configs = optim_configs

    def forward(self, x: torch.Tensor):
        output = self.alexnet(x)
        return output.argmax(dim=-1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, **self.optim_configs
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.alexnet(x)
        loss = nn.functional.cross_entropy(out, y)

        self.log("train_loss", loss.item())
        self.log("lr", self.lr)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.alexnet(x)
        loss = nn.functional.cross_entropy(out, y)
        pred = out.argmax(dim=-1)

        acc = (pred == y).sum() / y.size(0)

        self.log("val_loss", loss.item())
        self.log("val_acc", acc.item())

        return loss, acc

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.alexnet(x)
        loss = nn.functional.cross_entropy(out, y)
        pred = out.argmax(dim=-1)
        acc = (pred == y).sum() / y.size(0)

        self.log("test_loss", loss.item())
        self.log("test_acc", acc.item())

        return loss, acc
