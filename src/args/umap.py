import os

import omegaconf
from omegaconf import OmegaConf, ListConfig
from src.methods.base import BaseMethod
from src.utils.auto_resumer import AutoResumer
from src.utils.checkpointer import Checkpointer
from src.utils.misc import omegaconf_select
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


_N_CLASSES_PER_DATASET = {
    "cifar10": 10,
    "cifar100": 100,
    "stl10": 10,
    "imagenet": 1000,
    "imagenet100": 100,
    "bloodmnist": 8,
    "bbbc021": 14,
    "bbbc048": 7,
    "cyclops": 17,
    "tissuemnist": 8,
}

_SUPPORTED_DATASETS = [
    "cifar10",
    "cifar100",
    "stl10",
    "imagenet",
    "imagenet100",
    "idrcell100k",
    "bloodmnist",
    "bbbc021",
    "bbbc048",
    "cyclops",
    "tissuemnist",
    "bbbc021xbray",
    "mtbenchreg"
]


def add_and_assert_dataset_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Adds specific default values/checks for dataset config.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.

    Returns:
        omegaconf.DictConfig: same as the argument, used to avoid errors.
    """

    assert not OmegaConf.is_missing(cfg, "data.dataset")
    assert not OmegaConf.is_missing(cfg, "data.train_path")
    assert not OmegaConf.is_missing(cfg, "data.val_path")

    assert cfg.data.dataset in _SUPPORTED_DATASETS

    cfg.data.format = omegaconf_select(cfg, "data.format", "image_folder")
    cfg.data.fraction = omegaconf_select(cfg, "data.fraction", -1)
    cfg.data.img_channels = omegaconf_select(cfg, "data.img_channels", 3)
    cfg.data.sample_ratio = omegaconf_select(cfg, "data.sample_ratio", 1.0)
    cfg.data.multi_labels = omegaconf_select(cfg, "data.multi_labels", False)

    return cfg


def add_and_assert_slurm_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Adds specific default values/checks for SLURM config.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.

    Returns:
        omegaconf.DictConfig: same as the argument, used to avoid errors.
    """

    cfg.slurm = omegaconf_select(cfg, "slurm", {})
    cfg.slurm.enabled = omegaconf_select(cfg, "slurm.enabled", False)
    cfg.slurm.num_nodes = omegaconf_select(cfg, "slurm.num_nodes", 1)
    cfg.slurm.num_jobs_per_node = omegaconf_select(cfg, "slurm.num_jobs_per_node", 1)
    cfg.slurm.num_cpus_per_job = omegaconf_select(cfg, "slurm.num_cpus_per_job", 1)
    cfg.slurm.gpus_type = omegaconf_select(cfg, "slurm.gpus_type", "gpu:V100")

    return cfg


def add_and_assert_wandb_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Adds specific default values/checks for wandb config.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.

    Returns:
        omegaconf.DictConfig: same as the argument, used to avoid errors.
    """

    cfg.wandb = omegaconf_select(cfg, "wandb", {})
    cfg.wandb.enabled = omegaconf_select(cfg, "wandb.enabled", False)
    cfg.wandb.entity = omegaconf_select(cfg, "wandb.entity", None)
    cfg.wandb.project = omegaconf_select(cfg, "wandb.project", "bioFMv2")
    # set wandb offline if debug is enabled
    cfg.wandb.offline = omegaconf_select(cfg, "debug", False)

    return cfg


def add_and_assert_lightning_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Adds specific default values/checks for Pytorch Lightning config.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.

    Returns:
        omegaconf.DictConfig: same as the argument, used to avoid errors.
    """

    cfg.seed = omegaconf_select(cfg, "seed", 5)
    cfg.resume_from_checkpoint = omegaconf_select(cfg, "resume_from_checkpoint", None)
    cfg.strategy = omegaconf_select(cfg, "strategy", None)

    return cfg

def parse_cfg(cfg: omegaconf.DictConfig):
    """Parses feature extractor, dataset, pytorch lightning, linear eval specific and additional args.

    First adds an arg for the pretrained feature extractor, then adds dataset, pytorch lightning
    and linear eval specific args. If wandb is enabled, it adds checkpointer args. Finally, adds
    additional non-user given parameters.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    # Default value for SSL Validation Loss (if not specified, won't be used for efficiency purposes)
    cfg.ssl_val_loss = omegaconf_select(cfg, "ssl_val_loss", False)

    # Default value for DEBUG
    cfg.debug = omegaconf_select(cfg, "debug", False)

    # Default value for channels strategy
    cfg.channels_strategy = omegaconf_select(cfg, "channels_strategy", None)

    # default values for slurm stuff
    cfg = add_and_assert_slurm_cfg(cfg)

    # default values for checkpointer
    cfg = Checkpointer.add_and_assert_specific_cfg(cfg)

    # default values for auto_resume
    cfg = AutoResumer.add_and_assert_specific_cfg(cfg)

    # assert dataset parameters
    cfg = add_and_assert_dataset_cfg(cfg)

    # default values for wandb
    cfg = add_and_assert_wandb_cfg(cfg)

    # default values for pytorch lightning stuff
    cfg = add_and_assert_lightning_cfg(cfg)

    # backbone
    assert not omegaconf.OmegaConf.is_missing(cfg, "backbone.name")
    assert cfg.backbone.name in BaseMethod._BACKBONES

    # backbone kwargs
    cfg.backbone.kwargs = omegaconf_select(cfg, "backbone.kwargs", {})

    assert not omegaconf.OmegaConf.is_missing(cfg, "weights_init")

    cfg.method = omegaconf_select(cfg, "method", None)

    # extra training options
    cfg.auto_augment = omegaconf_select(cfg, "auto_augment", False)
    cfg.label_smoothing = omegaconf_select(cfg, "label_smoothing", 0.0)
    cfg.mixup = omegaconf_select(cfg, "mixup", 0.0)
    cfg.cutmix = omegaconf_select(cfg, "cutmix", 0.0)

    # augmentation related (crop size and custom mean/std values for normalization)
    cfg.data.augmentations = omegaconf_select(cfg, "data.augmentations", {})
    cfg.data.augmentations.crop_size = omegaconf_select(cfg, "data.augmentations.crop_size", 224)
    cfg.data.augmentations.mean = omegaconf_select(
        cfg, "data.augmentations.mean", IMAGENET_DEFAULT_MEAN
    )
    cfg.data.augmentations.std = omegaconf_select(
        cfg, "data.augmentations.std", IMAGENET_DEFAULT_STD)

    # extra processing
    if cfg.data.dataset in _N_CLASSES_PER_DATASET:
        cfg.data.num_classes = _N_CLASSES_PER_DATASET[cfg.data.dataset]
    else:
        # hack to maintain the current pipeline
        # even if the custom dataset doesn't have any labels
        cfg.data.num_classes = max(
            1,
            sum(entry.is_dir() for entry in os.scandir(cfg.data.train_path)),
        )

    # find number of big/small crops
    big_size = cfg.augmentations[0].crop_size
    num_large_crops = num_small_crops = 0
    for pipeline in cfg.augmentations:
        if big_size == pipeline.crop_size:
            num_large_crops += pipeline.num_crops
        else:
            num_small_crops += pipeline.num_crops
    cfg.data.num_large_crops = num_large_crops
    cfg.data.num_small_crops = num_small_crops

    if cfg.data.format == "dali":
        assert cfg.data.dataset in ["imagenet100", "imagenet", "custom"]

    # adjust lr according to batch size
    cfg.num_nodes = omegaconf_select(cfg, "num_nodes", 1)
    num_devices = len(cfg.devices) if isinstance(cfg.devices, ListConfig) else cfg.devices
    scale_factor = cfg.optimizer.batch_size * num_devices * cfg.num_nodes / 256
    cfg.optimizer.lr = cfg.optimizer.lr * scale_factor
    # token learner lr is only used if required in a ViTMultiChannels setting
    cfg.optimizer.token_learner_lr = omegaconf_select(cfg, "optimizer.token_learner_lr", None)
    cfg.optimizer.token_learner_lr = cfg.optimizer.token_learner_lr * scale_factor if cfg.optimizer.token_learner_lr is not None else None

    # extra optimizer kwargs
    cfg.optimizer.kwargs = omegaconf_select(cfg, "optimizer.kwargs", {})
    if cfg.optimizer.name == "sgd":
        cfg.optimizer.kwargs.momentum = omegaconf_select(cfg, "optimizer.kwargs.momentum", 0.9)
    elif cfg.optimizer.name == "lars":
        cfg.optimizer.kwargs.momentum = omegaconf_select(cfg, "optimizer.kwargs.momentum", 0.9)
        cfg.optimizer.kwargs.eta = omegaconf_select(cfg, "optimizer.kwargs.eta", 1e-3)
        cfg.optimizer.kwargs.clip_lr = omegaconf_select(cfg, "optimizer.kwargs.clip_lr", False)
        cfg.optimizer.kwargs.exclude_bias_n_norm = omegaconf_select(
            cfg,
            "optimizer.kwargs.exclude_bias_n_norm",
            False,
        )
    elif cfg.optimizer.name == "adamw":
        cfg.optimizer.kwargs.betas = omegaconf_select(cfg, "optimizer.kwargs.betas", [0.9, 0.999])

    return cfg
