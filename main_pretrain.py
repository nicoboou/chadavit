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
import logging
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

from src.args.pretrain import parse_cfg
from src.data.classification_dataloader import prepare_data as prepare_data_classification
from src.data.pretrain_dataloader import (
    FullTransformPipeline,
    FullTransformAlbumentationPipeline,
    NCropAugmentation,
    NCropAlbumentationAugmentation,
    build_transform_pipeline,
    prepare_dataloader,
    prepare_datasets,
)
from src.methods import METHODS
from src.utils.auto_resumer import AutoResumer
from src.utils.checkpointer import Checkpointer
from src.utils.slurm_logger import SLURMLogger
from src.utils.misc import make_contiguous, omegaconf_select
from src.data.channels_strategies import RandomDiscarder

try:
    from src.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True

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

    # initialize logging
    logging_level = logging.INFO
    logging.basicConfig(level=logging_level, filename=f'./logs/{cfg.name}.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

    seed_everything(cfg.seed)

    assert cfg.method in METHODS, f"Choose from {METHODS.keys()}"

    if cfg.data.num_large_crops != 2:
        assert cfg.method in ["wmse", "mae", "dino"]

    model = METHODS[cfg.method](cfg)
    make_contiguous(model)
    # can provide up to ~20% speed up
    if not cfg.performance.disable_channel_last:
        model = model.to(memory_format=torch.channels_last)

    # print model
    if cfg.slurm.enabled and _idr_torch_available:
        if idr_torch.rank == 0:
            print(summary(model))

    # ------------------------ SSL Validation Loss ACTIVATED ------------------------ #
    if cfg.ssl_val_loss:
        # pretrain dataloader
        pipelines = []
        for aug_cfg in cfg.augmentations:
            if cfg.mixed_channels:
                pipelines.append(
                    NCropAlbumentationAugmentation(
                        build_transform_pipeline(cfg.data.dataset, aug_cfg), aug_cfg.num_crops
                    )
                )
                transform = FullTransformAlbumentationPipeline(pipelines)
            else:
                pipelines.append(
                    NCropAugmentation(
                        build_transform_pipeline(cfg.data.dataset, aug_cfg), aug_cfg.num_crops
                    )
                )
                transform = FullTransformPipeline(pipelines)

        if cfg.debug_augmentations:
            print("Transforms:")
            print(transform)

        train_dataset, val_dataset = prepare_datasets(
            cfg.data.dataset,
            transform,
            train_data_path=cfg.data.train_path,
            val_data_path=cfg.data.val_path,
            data_format=cfg.data.format,
            no_labels=cfg.data.no_labels,
            data_fraction=cfg.data.fraction,
            return_val_dataset=cfg.ssl_val_loss,
            sample_ratio=cfg.data.sample_ratio
        )

        train_loader = prepare_dataloader(dataset=train_dataset, batch_size=cfg.optimizer.batch_size, num_workers=cfg.data.num_workers, channel_strategy=cfg.channels_strategy, shuffle=True)
        val_loader = prepare_dataloader(dataset=val_dataset, batch_size=cfg.optimizer.batch_size, num_workers=cfg.data.num_workers, channel_strategy=cfg.channels_strategy, shuffle=False)

    # ------------------------ SSL Validation Loss DEACTIVATED ------------------------ #

    else:
        # validation dataloader for when it is available
        val_data_format = cfg.data.format
        _, val_loader = prepare_data_classification(
            cfg.data.dataset,
            train_data_path=cfg.data.train_path,
            val_data_path=cfg.data.val_path,
            data_format=val_data_format,
            batch_size=cfg.optimizer.batch_size,
            num_workers=cfg.data.num_workers,
            channel_strategy=cfg.channels_strategy,
            sample_ratio=cfg.data.sample_ratio
        )

        # pretrain dataloader
        pipelines = []
        for aug_cfg in cfg.augmentations:
            if cfg.mixed_channels:
                pipelines.append(
                    NCropAlbumentationAugmentation(
                        build_transform_pipeline(cfg.data.dataset, aug_cfg), aug_cfg.num_crops
                    )
                )
                transform = FullTransformAlbumentationPipeline(pipelines)
            else:
                pipelines.append(
                    NCropAugmentation(
                        build_transform_pipeline(cfg.data.dataset, aug_cfg), aug_cfg.num_crops
                    )
                )
                transform = FullTransformPipeline(pipelines)

        if cfg.debug_augmentations:
            print("Transforms:")
            print(transform)

        train_dataset, _ = prepare_datasets(
            cfg.data.dataset,
            transform,
            train_data_path=cfg.data.train_path,
            val_data_path=cfg.data.val_path,
            data_format=cfg.data.format,
            no_labels=cfg.data.no_labels,
            data_fraction=cfg.data.fraction,
            return_val_dataset=cfg.ssl_val_loss,
            sample_ratio=cfg.data.sample_ratio
        )

        train_loader = prepare_dataloader(
            dataset=train_dataset, batch_size=cfg.optimizer.batch_size, num_workers=cfg.data.num_workers, channel_strategy=cfg.channels_strategy
        )

    # AutoResumer
    ckpt_path, wandb_run_id = None, None
    if cfg.auto_resume.enabled and cfg.resume_from_checkpoint is None and not cfg.slurm.enabled:
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(cfg.checkpoint.dir, cfg.method),
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
            logdir=os.path.join(cfg.checkpoint.dir, cfg.method),
            frequency=cfg.checkpoint.frequency,
            keep_prev=cfg.checkpoint.keep_prev,
        )
        callbacks.append(ckpt)

    if omegaconf_select(cfg, "auto_umap.enabled", False):
        assert (
            _umap_available
        ), "UMAP is not currently available, please install it first with [umap]."
        auto_umap = AutoUMAP(
            cfg.name,
            logdir=os.path.join(cfg.auto_umap.dir, cfg.method),
            frequency=cfg.auto_umap.frequency,
        )
        callbacks.append(auto_umap)

    # Logger
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
                save_dir=os.path.join(cfg.checkpoint.dir, cfg.method, str(cfg.slurm.job_id)),
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
    trainer_kwargs = {name: trainer_kwargs[name] for name in valid_kwargs if name in trainer_kwargs}
    trainer_kwargs.update(
        {
            "logger": logger if cfg.wandb.enabled else None,
            "callbacks": callbacks,
            "enable_checkpointing": False,
            "strategy": DDPStrategy(find_unused_parameters=False) if cfg.strategy == "ddp" else cfg.strategy,
            "plugins": [SLURMEnvironment(requeue_signal=signal.SIGUSR1)] if cfg.slurm.enabled else None,
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
    except:
        pass

    # Train
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    print("TRAINING FINISHED")
    # Workaround to log metrics to wandb at the end of a full run when using slurm autoresubmit
    if isinstance(logger,SLURMLogger) and cfg.slurm.enabled and _idr_torch_available:
        if idr_torch.rank == 0:
            run = wandb.init(**logger._wandb_init)

            print("initialized WanDB, run name : ",logger._name)
            print("WANDB Run id : ",run.id)
            print("logging to WANDB...")

            wandb.config.update(logger.hyperparams, allow_val_change=True)

            # read the log file and log each line to wandb
            with open(logger._save_dir + "/training_logs.txt", "r") as f:
                for line in f:
                    # convert line to dict
                    wandb.log(eval(line)) #allows to delete duplicates if any in terms of epoch

            wandb.finish()


if __name__ == "__main__":
    main()
