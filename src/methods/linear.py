# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Main imports
import logging
from typing import Any, Callable, Dict, List, Tuple, Union

# Pytorch imports
import omegaconf
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
from pytorch_lightning.loggers import WandbLogger
import torchmetrics

# Third party imports
from src.utils.lars import LARS
from src.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.utils.metrics import accuracy_at_k, weighted_mean
from src.utils.misc import (
    omegaconf_select,
    param_groups_layer_decay,
    remove_bias_and_norm_from_weight_decay,
)
from src.backbones.vit.chada_vit import ChAdaViT
from src.utils.slurm_logger import SLURMLogger

class LinearModel(pl.LightningModule):
    _OPTIMIZERS = {
        "sgd": torch.optim.SGD,
        "lars": LARS,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    _SCHEDULERS = [
        "reduce",
        "warmup_cosine",
        "step",
        "exponential",
        "none",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        cfg: omegaconf.DictConfig,
        loss_func: Callable = None,
        mixup_func: Callable = None,
    ):
        """Implements linear and finetune evaluation.

        .. note:: Cfg defaults are set in init by calling `cfg = add_and_assert_specific_cfg(cfg)`

        backbone (nn.Module): backbone architecture for feature extraction.
        Cfg basic structure:
            data:
                num_classes (int): number of classes in the dataset.
            max_epochs (int): total number of epochs.

            optimizer:
                name (str): name of the optimizer.
                batch_size (int): number of samples in the batch.
                lr (float): learning rate.
                weight_decay (float): weight decay for optimizer.
                kwargs (Dict): extra named arguments for the optimizer.
            scheduler:
                name (str): name of the scheduler.
                min_lr (float): minimum learning rate for warmup scheduler. Defaults to 0.0.
                warmup_start_lr (float): initial learning rate for warmup scheduler.
                    Defaults to 0.00003.
                warmup_epochs (float): number of warmup epochs. Defaults to 10.
                lr_decay_steps (Sequence, optional): steps to decay the learning rate
                    if scheduler is step. Defaults to None.
                interval (str): interval to update the lr scheduler. Defaults to 'step'.

            finetune (bool): whether or not to finetune the backbone. Defaults to False.

            performance:
                disable_channel_last (bool). Disables channel last conversion operation which
                speeds up training considerably. Defaults to False.
                https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models

        loss_func (Callable): loss function to use (for mixup, label smoothing or default).
        Defaults to None mixup_func (Callable, optional). function to convert data and targets
        with mixup/cutmix. Defaults to None.
        """

        super().__init__()

        # add default values and assert that config has the basic needed settings
        cfg = self.add_and_assert_specific_cfg(cfg)

        # backbone
        self.backbone = backbone
        if hasattr(self.backbone, "inplanes"):
            features_dim = self.backbone.inplanes
        else:
            features_dim = self.backbone.num_features

        # return all tokens
        self.return_all_tokens = cfg.backbone.kwargs.return_all_tokens

        # channels strategy
        if not cfg.backbone.kwargs.return_all_tokens:
            if cfg.channels_strategy == "one_channel":
                # we need to modify the classifier to accept the concatenated X channels as 1 big features vector
                features_dim *= cfg.data.img_channels
        else:
            if cfg.channels_strategy == "one_channel":
                features_dim = cfg.data.img_channels * self.backbone.patch_embed.num_patches * self.backbone.embed_dim
            elif cfg.channels_strategy == "multi_channels":
                features_dim = cfg.data.img_channels * self.backbone.token_learner.num_patches * self.backbone.embed_dim
            else:
                features_dim = self.backbone.patch_embed.num_patches * self.backbone.embed_dim
        
        # Num of classes
        self.num_classes = cfg.data.num_classes

        # classifier
        self.classifier = nn.Linear(features_dim, cfg.data.num_classes)  # type: ignore

        # mixup/cutmix function
        self.mixup_func: Callable = mixup_func

        if loss_func is None:
            loss_func = nn.CrossEntropyLoss()
        self.loss_func = loss_func

        # training related
        self.max_epochs: int = cfg.max_epochs
        self.accumulate_grad_batches: Union[int, None] = cfg.accumulate_grad_batches

        # optimizer related
        self.optimizer: str = cfg.optimizer.name
        self.batch_size: int = cfg.optimizer.batch_size
        self.lr: float = cfg.optimizer.lr
        self.weight_decay: float = cfg.optimizer.weight_decay
        self.extra_optimizer_args: Dict[str, Any] = cfg.optimizer.kwargs
        self.exclude_bias_n_norm_wd: bool = cfg.optimizer.exclude_bias_n_norm_wd
        self.layer_decay: float = cfg.optimizer.layer_decay

        # scheduler related
        self.scheduler: str = cfg.scheduler.name
        self.lr_decay_steps: Union[List[int], None] = cfg.scheduler.lr_decay_steps
        self.min_lr: float = cfg.scheduler.min_lr
        self.warmup_start_lr: float = cfg.scheduler.warmup_start_lr
        self.warmup_epochs: int = cfg.scheduler.warmup_epochs
        self.scheduler_interval: str = cfg.scheduler.interval
        assert self.scheduler_interval in ["step", "epoch"]
        if self.scheduler_interval == "step":
            logging.warning(
                f"Using scheduler_interval={self.scheduler_interval} might generate "
                "issues when resuming a checkpoint."
            )

        # if finetuning the backbone
        self.finetune: bool = cfg.finetune

        # for performance
        self.no_channel_last = cfg.performance.disable_channel_last

        if not self.finetune:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # keep track of validation metrics
        self.validation_step_metrics = []
        self.validation_step_targets = []
        self.validation_step_preds = []

        # channels strategy
        self.channels_strategy = cfg.channels_strategy
        self.mixed_channels = cfg.mixed_channels
        self.list_num_channels = []

        # SLURM Handling
        self.slurm_enabled = cfg.slurm.enabled
        self.wandb_enabled = cfg.wandb.enabled
        if cfg.slurm.enabled:
            self.job_id = cfg.slurm.job_id

        # Metrics
        self.acc_at_1 = torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes, top_k=1)
        self.acc_at_5 = torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes, top_k=5)
        self.recall = torchmetrics.classification.MulticlassRecall(num_classes=self.num_classes, top_k=1)
        self.precision = torchmetrics.classification.MulticlassPrecision(num_classes=self.num_classes, top_k=1)
        self.auroc = torchmetrics.classification.MulticlassAUROC(num_classes=self.num_classes)
        self.f1_score = torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes, top_k=1, average='macro')
        self.cm = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=self.num_classes)


    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        # default parameters for optimizer
        cfg.optimizer.exclude_bias_n_norm_wd = omegaconf_select(
            cfg, "optimizer.exclude_bias_n_norm_wd", False
        )
        # default for extra optimizer kwargs (use pytorch's default if not available)
        cfg.optimizer.kwargs = omegaconf_select(cfg, "optimizer.kwargs", {})
        cfg.optimizer.layer_decay = omegaconf_select(cfg, "optimizer.layer_decay", 0.0)

        # whether or not to finetune the backbone
        cfg.finetune = omegaconf_select(cfg, "finetune", False)

        # default for acc grad batches
        cfg.accumulate_grad_batches = omegaconf_select(cfg, "accumulate_grad_batches", 1)

        # default parameters for the scheduler
        cfg.scheduler.lr_decay_steps = omegaconf_select(cfg, "scheduler.lr_decay_steps", None)
        cfg.scheduler.min_lr = omegaconf_select(cfg, "scheduler.min_lr", 0.0)
        cfg.scheduler.warmup_start_lr = omegaconf_select(cfg, "scheduler.warmup_start_lr", 3e-5)
        cfg.scheduler.warmup_epochs = omegaconf_select(cfg, "scheduler.warmup_epochs", 10)
        cfg.scheduler.interval = omegaconf_select(cfg, "scheduler.interval", "step")

        # default parameters for performance optimization
        cfg.performance = omegaconf_select(cfg, "performance", {})
        cfg.performance.disable_channel_last = omegaconf_select(
            cfg, "performance.disable_channel_last", False
        )

        # channels strategy
        cfg.channels_strategy = omegaconf_select(cfg, "channels_strategy", None)

        return cfg

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        if self.layer_decay > 0:
            assert self.finetune, "Only with use layer weight decay with finetune on."
            msg = (
                "Method should implement no_weight_decay() that returns "
                "a set of parameter names to ignore from weight decay"
            )
            assert hasattr(self.backbone, "no_weight_decay"), msg

            learnable_params = param_groups_layer_decay(
                self.backbone,
                self.weight_decay,
                no_weight_decay_list=self.backbone.no_weight_decay(),
                layer_decay=self.layer_decay,
            )
            learnable_params.append({"name": "classifier", "params": self.classifier.parameters()})
        else:
            learnable_params = (
                self.classifier.parameters()
                if not self.finetune
                else [
                    {"name": "backbone", "params": self.backbone.parameters()},
                    {"name": "classifier", "params": self.classifier.parameters()},
                ]
            )

        # exclude bias and norm from weight decay
        if self.exclude_bias_n_norm_wd:
            learnable_params = remove_bias_and_norm_from_weight_decay(learnable_params)

        assert self.optimizer in self._OPTIMIZERS
        optimizer = self._OPTIMIZERS[self.optimizer]

        optimizer = optimizer(
            learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        # select scheduler
        if self.scheduler == "none":
            return optimizer

        if self.scheduler == "warmup_cosine":
            max_warmup_steps = (
                self.warmup_epochs * (self.trainer.estimated_stepping_batches / self.max_epochs)
                if self.scheduler_interval == "step"
                else self.warmup_epochs
            )
            max_scheduler_steps = (
                self.trainer.estimated_stepping_batches
                if self.scheduler_interval == "step"
                else self.max_epochs
            )
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=max_warmup_steps,
                    max_epochs=max_scheduler_steps,
                    warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr,
                    eta_min=self.min_lr,
                ),
                "interval": self.scheduler_interval,
                "frequency": 1,
            }
        elif self.scheduler == "reduce":
            scheduler = ReduceLROnPlateau(optimizer)
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.1)
        elif self.scheduler == "exponential":
            scheduler = ExponentialLR(optimizer, self.weight_decay)
        else:
            raise ValueError(
                f"{self.scheduler} not in (warmup_cosine, cosine, reduce, step, exponential)"
            )

        return [optimizer], [scheduler]

    def forward(self, X: torch.tensor, index:int) -> Dict[str, Any]:
        """Performs forward pass of the frozen backbone and the linear layer for evaluation.

        Args:
            X (torch.tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing features and logits.
        """

        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)

        with torch.set_grad_enabled(self.finetune):
            if self.channels_strategy == "multi_channels":
                # Assert that backbone is of class ChAdaViT
                assert isinstance(self.backbone, ChAdaViT), "Only backbone of class ChAdaViT is currently supported for multi_channels strategy."
                feats = self.backbone(X, index, self.list_num_channels)
            else:
                feats = self.backbone.forward_features(X)[:,1:] if self.return_all_tokens else self.backbone(X)
            
            if not self.mixed_channels:
                if not self.return_all_tokens and self.channels_strategy == "one_channel":
                    # Concatenate feature embeddings per image
                    chunks = torch.split(feats, self.list_num_channels[index], dim=0)
                    # Concatenate the chunks along the batch dimension
                    feats = torch.stack(chunks, dim=0)
                    # Assuming tensor is of shape (batch_size, num_tokens, backbone_output_dim)
                    feats = feats.flatten(start_dim=1)

                elif self.return_all_tokens:
                    if (self.channels_strategy == "one_channel" or self.channels_strategy == "multi_channels"):
                        # Concatenate feature embeddings per image
                        chunks = feats.view(sum(self.list_num_channels[index]), -1, feats.shape[-1])
                        chunks = torch.split(chunks, self.list_num_channels[index], dim=0)
                        # Concatenate the chunks along the batch dimension
                        feats = torch.stack(chunks, dim=0)
                    # Assuming tensor is of shape (batch_size, num_tokens, backbone_output_dim)
                    feats = feats.flatten(start_dim=1)

        logits = self.classifier(feats)  # feats = (batch_size, num_tokens * backbone_output_dim)
        return {"logits": logits, "feats": feats}

    def shared_step(self, batch: Tuple, batch_idx: int, index: int) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs operations that are shared between the training nd validation steps.

        Args:
            batch (Tuple): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
                batch size, loss, accuracy @1 and accuracy @5.
        """

        if self.channels_strategy == "one_channel" or self.channels_strategy == "multi_channels":
            X, target, list_num_channels = batch
            self.list_num_channels = list_num_channels
        else:
            X, target = batch

        metrics = {"batch_size": X.size(0)}
        if self.training and self.mixup_func is not None:
            X, target = self.mixup_func(X, target)
            out = self(X, index)["logits"]
            loss = self.loss_func(out, target)
            metrics.update({"loss": loss})
        else:
            out = self(X, index)["logits"]
            # Compute CE loss on logits
            loss = F.cross_entropy(out, target)

            # Compute accuracies on logits
            acc1, acc5 = accuracy_at_k(out, target, top_k=(1, 5))
            acc_1 = self.acc_at_1(out, target) * 100
            acc_5 = self.acc_at_5(out, target) * 100

            # Compute other classification metrics
            recall = self.recall(out, target)
            precision = self.precision(out, target)
            auroc = self.auroc(out, target)
            f1_score = self.f1_score(out, target)

            metrics.update({"loss": loss, "acc1": acc1, "acc5": acc5, "acc_1_macro": acc_1, "acc_5_macro":acc_5, "recall": recall, "precision": precision, "auroc": auroc, "f1_score": f1_score})

            # if isinstance(self.logger, WandbLogger):
            #     cm, fig, ax = confusion_matrix(out, target, num_classes=self.num_classes, plot=True)
            #     metrics.update({"confusion_matrix": wandb.Image(fig)})
            #     plt.close(fig)

            # elif isinstance(self.logger, SLURMLogger):
            #     # Save image locally
            #     cm, fig, ax = confusion_matrix(out, target, num_classes=self.num_classes, plot=True)
            #     fig.savefig(f"val_confusion_matrix_{self.job_id}.png")
            #     plt.close(fig)

        # Keep track of targets and preds for confusion matrix

        _, pred = out.topk(1, 1, True, True)
        pred = pred.t().squeeze(0)
        self.validation_step_targets.append(target)
        self.validation_step_preds.append(pred)

        return metrics

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Performs the training step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            torch.Tensor: cross-entropy loss between the predictions and the ground truth.
        """

        # set backbone to eval mode
        if not self.finetune:
            self.backbone.eval()

        out = self.shared_step(batch, batch_idx, index=0)

        log = {"train_loss": out["loss"]}
        if self.mixup_func is None:
            log.update({"train_acc1": out["acc1"], "train_acc5": out["acc5"], "train_acc1_macro": out['acc_1_macro'], "train_acc5_macro": out['acc_5_macro'], "train_recall": out["recall"], "train_precision": out["precision"], "train_auroc": out["auroc"], "train_f1_score": out["f1_score"]})

        self.log_dict(log, on_epoch=True, sync_dist=True)
        return out["loss"]

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        """Performs the validation step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Dict[str, Any]:
                dict with the batch_size (used for averaging),
                the classification loss and accuracies.
        """

        out = self.shared_step(batch, batch_idx, index=0)

        metrics = {
            "batch_size": out["batch_size"],
            "val_loss": out["loss"],
            "val_acc1": out["acc1"],
            "val_acc5": out["acc5"],
            "val_recall": out["recall"],
            "val_precision": out["precision"],
            "val_auroc": out["auroc"],
            "val_f1_score": out["f1_score"]
        }

        self.validation_step_metrics.append(metrics)
        return metrics

    def on_validation_epoch_end(self):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.
        """

        val_loss = weighted_mean(self.validation_step_metrics, "val_loss", "batch_size")
        val_acc1 = weighted_mean(self.validation_step_metrics, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(self.validation_step_metrics, "val_acc5", "batch_size")
        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5, "val_recall": self.recall, "val_precision": self.precision, "val_auroc": self.auroc, "val_f1_score": self.f1_score}

        # confusion matrix
        pred = torch.cat(self.validation_step_preds)
        targets = torch.cat(self.validation_step_targets)

        confusion_matrix_computed = self.cm(pred, targets).detach().cpu().numpy().astype(int)
        #confusion_matrix.compute().detach().cpu().numpy().astype(int)

        df_cm = pd.DataFrame(confusion_matrix_computed)
        plt.figure(figsize = (20,13))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='crest', fmt='d').get_figure()

        # Save image locally
        if isinstance(self.logger, SLURMLogger):
            # Save image locally
            fig_.savefig(f"val_confusion_matrix_{self.job_id}.png")
            plt.close(fig_)

        elif isinstance(self.logger, WandbLogger):
            # adding captions
            self.logger.log_image(key="Confusion Matrix", images=[wandb.Image(fig_)], caption=["Confusion_matrix"])
            plt.close(fig_)

        self.validation_step_metrics.clear()
        self.validation_step_targets.clear()
        self.validation_step_preds.clear()

        self.log_dict(log, sync_dist=True)
