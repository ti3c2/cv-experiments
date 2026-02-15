import typing as tp
from pathlib import Path

import lightning as L
import torch
import torchmetrics as tm
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader

from ..settings import settings


class LightningClassification(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: nn.Module,
        metrics: tp.Sequence[tm.Metric] | None = None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        # Register metrics as submodules so Lightning moves them to the model device.
        self.train_metrics = nn.ModuleDict(
            {
                f"{metric.__class__.__name__.lower()}_{idx}": metric.clone()
                for idx, metric in enumerate(metrics or [])
            }
        )
        self.val_metrics = nn.ModuleDict(
            {
                f"{metric.__class__.__name__.lower()}_{idx}": metric.clone()
                for idx, metric in enumerate(metrics or [])
            }
        )
        self.test_metrics = nn.ModuleDict(
            {
                f"{metric.__class__.__name__.lower()}_{idx}": metric.clone()
                for idx, metric in enumerate(metrics or [])
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("loss/train", loss.detach(), on_step=False, on_epoch=True)
        for metric_name, metric in self.train_metrics.items():
            self.log(
                f"metrics-train/{metric_name}",
                metric(y_hat, y),
                on_step=False,
                on_epoch=True,
            )
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("loss/val", loss.detach(), on_step=False, on_epoch=True)
        for metric_name, metric in self.val_metrics.items():
            self.log(
                f"metrics-val/{metric_name}",
                metric(y_hat, y),
                on_step=False,
                on_epoch=True,
            )
        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("loss/test", loss.detach(), on_step=False, on_epoch=True)
        for metric_name, metric in self.test_metrics.items():
            self.log(
                f"metrics-test/{metric_name}",
                metric(y_hat, y),
                on_step=False,
                on_epoch=True,
            )
        return loss


    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]


def get_classification_metrics(
    num_classes: int,
) -> list[tm.Metric]:
    return [
        tm.Accuracy(task="multiclass", num_classes=num_classes),
        tm.Precision(task="multiclass", average="macro", num_classes=num_classes),
        tm.Recall(task="multiclass", average="macro", num_classes=num_classes),
        tm.F1Score(task="multiclass", average="macro", num_classes=num_classes),
    ]


def run_trainer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    num_classes: int,
    trainer_config: dict[str, tp.Any] | None = None,
) -> None:
    trainer_config = settings.get_trainer_config()
    trainer_config.update(trainer_config or {})

    logger = TensorBoardLogger(
        save_dir=settings.path_logs,
        name=settings.tensorboard_logger_name,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(logger.log_dir) / "checkpoints",
        filename=settings.checkpoint_filename,
        monitor=settings.checkpoint_monitor,
        mode=settings.checkpoint_mode,
        save_top_k=settings.checkpoint_save_top_k,
        every_n_epochs=settings.checkpoint_every_n_epochs,
        save_last=settings.checkpoint_save_last,
    )
    trainer = L.Trainer(
        **trainer_config,
        callbacks=[checkpoint_callback],
        default_root_dir=settings.path_logs,
        logger=logger,
    )
    model = LightningClassification(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        metrics=get_classification_metrics(num_classes),
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    return trainer.test(dataloaders=test_dataloader)
