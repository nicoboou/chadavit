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

import inspect
import os
import signal

import hydra
import torch
import wandb
from torchsummary import summary
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, RichModelSummary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.plugins.environments import SLURMEnvironment
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from src.args.linear import parse_cfg
from src.data.classification_dataloader import prepare_data
from src.methods.base import BaseMethod
from src.methods.linear import LinearModel
from src.utils.auto_resumer import AutoResumer
from src.utils.checkpointer import Checkpointer
from src.utils.slurm_logger import SLURMLogger
from src.utils.misc import make_contiguous
from src.data.channels_strategies import modify_first_layer

try:
    from src.data.dali_dataloader import ClassificationDALIDataModule
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

try:
    import idr_torch
except ImportError:
    _idr_torch_available = False
else:
    _idr_torch_available = True

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["_WANDB_STARTUP_DEBUG"] = "true"


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    # hydra doesn't allow us to add new keys for "safety"
    # set_struct(..., False) disables this behavior and allows us to add more parameters
    # without making the user specify every single thing about the model
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)

    seed_everything(cfg.seed)

    # initialize backbone
    backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]

    # Assert pretraining rule
    ckpt_path = cfg.pretrained_feature_extractor
    assert ckpt_path is not None, "pretrained_feature_extractor is required"

    # load imagenet weights
    if ckpt_path == "imagenet-weights":
        pretrained = True
        backbone = backbone_model(
            method=cfg.pretrain_method, pretrained=pretrained, **cfg.backbone.kwargs
        )
        backbone = modify_first_layer(backbone=backbone, cfg=cfg, pretrained=pretrained)

    # load custom pretrained weights for OneChannel or MultiChannels
    elif (
        cfg.channels_strategy == "one_channel"
        or cfg.channels_strategy == "multi_channels"
    ):
        assert (
            ckpt_path.endswith(".ckpt")
            or ckpt_path.endswith(".pth")
            or ckpt_path.endswith(".pt")
        ), "If not loading pretrained imagenet weights on backbone, pretrained_feature_extractor must be a .ckpt or .pth file"
        pretrained = False
        backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
        backbone = modify_first_layer(backbone=backbone, cfg=cfg, pretrained=pretrained)
        state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        for k in list(state.keys()):
            if "encoder" in k:
                state[k.replace("encoder", "backbone")] = state[k]
            if "backbone" in k:
                state[k.replace("backbone.", "")] = state[k]
            del state[k]
        backbone.load_state_dict(state, strict=False)

    # load custom pretrained weights for Standard architecture (trained on 3-Channels/RGB images)
    else:
        assert (
            ckpt_path.endswith(".ckpt")
            or ckpt_path.endswith(".pth")
            or ckpt_path.endswith(".pt")
        ), "If not loading pretrained imagenet weights on backbone, pretrained_feature_extractor must be a .ckpt or .pth file"
        pretrained = False
        backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
        state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        for k in list(state.keys()):
            if "encoder" in k:
                state[k.replace("encoder", "backbone")] = state[k]
            if "backbone" in k:
                state[k.replace("backbone.", "")] = state[k]
            del state[k]
        backbone.load_state_dict(state, strict=False)
        backbone = modify_first_layer(backbone=backbone, cfg=cfg, pretrained=pretrained)

    # check if mixup or cutmix is enabled
    mixup_func = None
    mixup_active = cfg.mixup > 0 or cfg.cutmix > 0
    if mixup_active:
        mixup_func = Mixup(
            mixup_alpha=cfg.mixup,
            cutmix_alpha=cfg.cutmix,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=cfg.label_smoothing,
            num_classes=cfg.data.num_classes,
        )
        # smoothing is handled with mixup label transform
        loss_func = SoftTargetCrossEntropy()
    elif cfg.label_smoothing > 0:
        loss_func = LabelSmoothingCrossEntropy(smoothing=cfg.label_smoothing)
    else:
        loss_func = torch.nn.CrossEntropyLoss()

    model = LinearModel(backbone, loss_func=loss_func, mixup_func=mixup_func, cfg=cfg)
    make_contiguous(model)
    # can provide up to ~20% speed up
    if not cfg.performance.disable_channel_last:
        model = model.to(memory_format=torch.channels_last)

    # print model
    if cfg.slurm.enabled and _idr_torch_available:
        if idr_torch.rank == 0:
            print(summary(model))

    if cfg.data.format == "dali":
        val_data_format = "image_folder"
    else:
        val_data_format = cfg.data.format

    train_loader, val_loader = prepare_data(
        cfg.data.dataset,
        train_data_path=cfg.data.train_path,
        val_data_path=cfg.data.val_path,
        data_format=val_data_format,
        batch_size=cfg.optimizer.batch_size,
        num_workers=cfg.data.num_workers,
        auto_augment=cfg.auto_augment,
        channel_strategy=cfg.channels_strategy,
        sample_ratio=cfg.data.sample_ratio,
    )

    if cfg.data.format == "dali":
        assert _dali_avaliable, "Dali is not currently avaiable, please install it first with pip3 install .[dali]."

        assert not cfg.auto_augment, "Auto augmentation is not supported with Dali."

        dali_datamodule = ClassificationDALIDataModule(
            dataset=cfg.data.dataset,
            train_data_path=cfg.data.train_path,
            val_data_path=cfg.data.val_path,
            num_workers=cfg.data.num_workers,
            batch_size=cfg.optimizer.batch_size,
            data_fraction=cfg.data.fraction,
            dali_device=cfg.dali.device,
        )

        # use normal torchvision dataloader for validation to save memory
        dali_datamodule.val_dataloader = lambda: val_loader

    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    ckpt_path, wandb_run_id = None, None
    if (
        cfg.auto_resume.enabled
        and cfg.resume_from_checkpoint is None
        and not cfg.slurm.enabled
    ):
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(cfg.checkpoint.dir, "linear"),
            max_hours=cfg.auto_resume.max_hours,
        )
        resume_from_checkpoint, wandb_run_id = auto_resumer.find_checkpoint(cfg)
        if resume_from_checkpoint is not None:
            print(
                "Resuming from previous checkpoint that matches specifications:",
                f"'{resume_from_checkpoint}'",
            )
            ckpt_path = resume_from_checkpoint
    elif cfg.resume_from_checkpoint is not None:
        ckpt_path = cfg.resume_from_checkpoint
        del cfg.resume_from_checkpoint

    callbacks = []

    if cfg.checkpoint.enabled:
        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            cfg,
            logdir=os.path.join(cfg.checkpoint.dir, "linear"),
            frequency=cfg.checkpoint.frequency,
            keep_prev=cfg.checkpoint.keep_prev,
        )
        callbacks.append(ckpt)

    # wandb logging
    if cfg.wandb.enabled:
        if not cfg.slurm.enabled:
            logger = WandbLogger(
                name=cfg.name,
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                offline=cfg.wandb.offline,
                resume="allow" if (wandb_run_id or cfg.slurm.enabled) else None,
                id=os.environ["SLURM_JOB_ID"] if cfg.slurm.enabled else wandb_run_id,
            )
            logger.watch(model, log="gradients", log_freq=100)
            logger.log_hyperparams(OmegaConf.to_container(cfg))

        else:
            logger = SLURMLogger(
                save_dir=os.path.join(
                    cfg.checkpoint.dir, "linear", str(cfg.slurm.job_id)
                ),
                name=cfg.name,
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                offline=cfg.wandb.offline,
                version=os.environ["SLURM_JOB_ID"] if cfg.slurm.enabled else None,
            )
            logger.log_hyperparams(OmegaConf.to_container(cfg))

    # lr logging
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # model summary
    model_summary = RichModelSummary()
    callbacks.append(model_summary)

    trainer_kwargs = OmegaConf.to_container(cfg)
    # we only want to pass in valid Trainer args, the rest may be user specific
    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {
        name: trainer_kwargs[name] for name in valid_kwargs if name in trainer_kwargs
    }
    trainer_kwargs.update(
        {
            "logger": logger if cfg.wandb.enabled else None,
            "callbacks": callbacks,
            "enable_checkpointing": False,
            "strategy": DDPStrategy(find_unused_parameters=False)
            if cfg.strategy == "ddp"
            else cfg.strategy,
            "plugins": [SLURMEnvironment(requeue_signal=signal.SIGUSR1)]
            if cfg.slurm.enabled
            else None,
        }
    )
    trainer = Trainer(**trainer_kwargs)

    # fix for incompatibility with nvidia-dali and pytorch lightning
    # with dali 1.15 (this will be fixed on 1.16)
    # https://github.com/Lightning-AI/lightning/issues/12956
    try:
        from pytorch_lightning.loops import FitLoop

        class WorkaroundFitLoop(FitLoop):
            @property
            def prefetch_batches(self) -> int:
                return 1

        trainer.fit_loop = WorkaroundFitLoop(
            trainer.fit_loop.min_epochs, trainer.fit_loop.max_epochs
        )
    except ImportError:
        pass

    # Start training without Submitit
    if cfg.data.format == "dali":
        trainer.fit(model, ckpt_path=ckpt_path, datamodule=dali_datamodule)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    print("TRAINING FINISHED")

    # Workaround to log metrics to wandb at the end of a full run when using slurm autoresubmit
    if isinstance(logger, SLURMLogger) and cfg.slurm.enabled and _idr_torch_available:
        if idr_torch.rank == 0:
            run = wandb.init(**logger._wandb_init)

            print("initialized WanDB, run name : ", logger._name)
            print("WANDB Run id : ", run.id)
            print("logging to WANDB...")

            wandb.config.update(logger.hyperparams, allow_val_change=True)

            # read the log file and log each line to wandb
            with open(logger._save_dir + "/training_logs.txt", "r") as f:
                for line in f:
                    # convert line to dict
                    wandb.log(
                        eval(line)
                    )  # allows to delete duplicates if any in terms of epoch

            # Check if confusion matrix picture is available
            # check .png file begins wit val_confusion_matrix
            for file in os.listdir(os.getcwd()):
                if file.startswith("val_confusion_matrix"):
                    # check if has the same id as the run
                    if file.split("_")[3].split(".")[0] == os.environ["SLURM_JOB_ID"]:
                        wandb.log(
                            {
                                "val_confusion_matrix": wandb.Image(
                                    os.path.join(os.getcwd(), file)
                                )
                            }
                        )

            wandb.finish()


if __name__ == "__main__":
    main()
