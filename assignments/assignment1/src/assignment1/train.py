"""Hydra-backed training entrypoint for the coursework experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from . import data as data_module
from . import models
from . import utils

log = logging.getLogger(__name__)


def _instantiate_optimizer(model: nn.Module, cfg: DictConfig) -> Optimizer:
    constructor = getattr(torch.optim, cfg.name)
    return constructor(model.parameters(), **cfg.params)


def _instantiate_scheduler(optimizer: Optimizer, cfg: DictConfig) -> Any:
    if not cfg:
        return None
    constructor = getattr(torch.optim.lr_scheduler, cfg.name)
    return constructor(optimizer, **cfg.params)


def _setup_wandb(cfg: DictConfig, model: nn.Module) -> None:
    logging_cfg = cfg.logging.wandb
    if not logging_cfg.enabled:
        return
    try:
        import wandb

        run = wandb.init(
            project=logging_cfg.project,
            entity=logging_cfg.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=logging_cfg.mode,
            tags=list(logging_cfg.tags),
        )
        if run is not None:
            run.name = cfg.experiment_name
            wandb.watch(
                model,
                log=logging_cfg.watch.log,
                log_freq=logging_cfg.watch.log_freq,
            )
    except ImportError:
        log.warning("wandb is not installed; skipping experiment tracking.")


def _log_samples(fig_path: Path, batch, writer: SummaryWriter | None, step: int, cfg: DictConfig) -> None:
    if not cfg.logging.matplotlib.enabled:
        return

    images, labels = batch
    num_samples = min(cfg.logging.matplotlib.num_samples, images.size(0))
    grid_size = int(num_samples ** 0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(4, 4), dpi=cfg.logging.matplotlib.dpi)
    for ax, image, label in zip(axes.flatten(), images[: num_samples], labels[: num_samples]):
        ax.imshow(image.squeeze(0), cmap="gray")
        ax.set_title(str(int(label.item())))
        ax.axis("off")

    utils.save_matplotlib_grid(fig, fig_path)
    plt.close(fig)

    if writer is not None:
        grid = make_grid(images[:num_samples], nrow=grid_size)
        writer.add_image("samples/train", grid, step)

    try:
        import wandb

        if wandb.run is not None:
            wandb.log({"samples": wandb.Image(fig_path)}, step=step)
    except ImportError:
        pass


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def train(cfg: DictConfig) -> Dict[str, float]:
    utils.seed_everything(cfg.training.seed)

    device = torch.device("cuda" if cfg.training.use_cuda and torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    dataloaders = data_module.prepare_dataloaders(cfg)

    model = models.build_model(cfg.model).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.training.label_smoothing))
    optimizer = _instantiate_optimizer(model, cfg.training.optimizer)
    scheduler = _instantiate_scheduler(optimizer, cfg.training.scheduler)

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.training.mixed_precision) and device.type == "cuda")

    writer: SummaryWriter | None = None
    if cfg.logging.tensorboard.enabled:
        writer = SummaryWriter(log_dir=cfg.logging.tensorboard.log_dir)

    _setup_wandb(cfg, model)

    best_val_acc = 0.0
    metrics: Dict[str, float] = {}
    global_step = 0

    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for batch_idx, batch in enumerate(dataloaders["train"]):
            result = utils.step(
                model,
                batch,
                criterion,
                device,
                optimizer=optimizer,
                scaler=scaler if scaler.is_enabled() else None,
                max_grad_norm=cfg.training.max_grad_norm,
            )
            train_loss += result.loss
            train_acc += result.accuracy

            if batch_idx == 0 and epoch == 0:
                fig_path = Path("figures/samples_epoch0.png")
                _log_samples(fig_path, batch, writer, global_step, cfg)

            if (batch_idx + 1) % cfg.training.log_interval == 0:
                avg_loss = train_loss / (batch_idx + 1)
                avg_acc = train_acc / (batch_idx + 1)
                log.info(
                    "Epoch %d [%d/%d] - loss: %.4f acc: %.4f",
                    epoch + 1,
                    batch_idx + 1,
                    len(dataloaders["train"]),
                    avg_loss,
                    avg_acc,
                )
                utils.maybe_log_to_wandb(
                    {"train/loss": avg_loss, "train/accuracy": avg_acc, "epoch": epoch + 1},
                    global_step,
                )
                if writer is not None:
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar("train/accuracy", avg_acc, global_step)
            global_step += 1

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % cfg.training.eval_interval == 0:
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            with torch.no_grad():
                for batch in dataloaders["val"]:
                    result = utils.step(model, batch, criterion, device)
                    val_loss += result.loss
                    val_acc += result.accuracy

            val_loss /= len(dataloaders["val"])
            val_acc /= len(dataloaders["val"])
            log.info("Epoch %d validation loss %.4f acc %.4f", epoch + 1, val_loss, val_acc)
            metrics.update({"validation/loss": val_loss, "validation/accuracy": val_acc})
            utils.maybe_log_to_wandb(metrics, global_step)
            if writer is not None:
                writer.add_scalar("validation/loss", val_loss, epoch + 1)
                writer.add_scalar("validation/accuracy", val_acc, epoch + 1)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                artifact_path = Path("artifacts/best_model.pt")
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), artifact_path)

    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for batch in dataloaders["test"]:
            result = utils.step(model, batch, criterion, device)
            test_loss += result.loss
            test_acc += result.accuracy
    test_loss /= len(dataloaders["test"])
    test_acc /= len(dataloaders["test"])

    metrics.update({"test/loss": test_loss, "test/accuracy": test_acc})
    utils.maybe_log_to_wandb(metrics, global_step)
    if writer is not None:
        writer.add_scalar("test/loss", test_loss, cfg.training.epochs)
        writer.add_scalar("test/accuracy", test_acc, cfg.training.epochs)
        writer.close()

    try:
        import wandb

        if wandb.run is not None:
            wandb.summary.update(metrics)
            wandb.finish()
    except ImportError:
        pass

    log.info("Training complete. Test accuracy %.4f", test_acc)
    return metrics


if __name__ == "__main__":
    train()
