import os

import omegaconf
from omegaconf import OmegaConf, ListConfig
from src.utils.auto_resumer import AutoResumer
from src.utils.checkpointer import Checkpointer
from src.utils.misc import omegaconf_select

try:
    from src.data.dali_dataloader import PretrainDALIDataModule
except ImportError:
    _dali_available = False
else:
    _dali_available = True

try:
    from src.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True

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
    "idrcell100k_3channels",
    "bloodmnist",
    "bbbc021",
    "bbbc048",
    "cyclops",
    "tissuemnist",
    "mtbenchreg",
    "bray"
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

    assert cfg.data.dataset in _SUPPORTED_DATASETS

    # if validation path is not available, assume that we want to skip eval
    cfg.data.val_path = omegaconf_select(cfg, "data.val_path", None)
    cfg.data.format = omegaconf_select(cfg, "data.format", "image_folder")
    cfg.data.no_labels = omegaconf_select(cfg, "data.no_labels", False)
    cfg.data.fraction = omegaconf_select(cfg, "data.fraction", -1)
    cfg.debug_augmentations = omegaconf_select(cfg, "debug_augmentations", False)
    cfg.data.img_channels = omegaconf_select(cfg, "data.img_channels", 3)
    cfg.data.sample_ratio = omegaconf_select(cfg, "data.sample_ratio", 1.0)

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

    cfg.seed = omegaconf_select(cfg, "seed", None)
    cfg.resume_from_checkpoint = omegaconf_select(cfg, "resume_from_checkpoint", None)
    cfg.strategy = omegaconf_select(cfg, "strategy", None)

    return cfg

def parse_cfg(cfg: omegaconf.DictConfig):
    """Parses the config file and adds default values/checks."""

    # Default value for SSL Validation Loss (if not specified, won't be used for efficiency purposes)
    cfg.ssl_val_loss = omegaconf_select(cfg, "ssl_val_loss", False)

    # Default value for DEBUG
    cfg.debug = omegaconf_select(cfg, "debug", False)

    # Default value for channels strategy
    cfg.channels_strategy = omegaconf_select(cfg, "channels_strategy", None)

    # Default value for return_all_tokens
    cfg.backbone.kwargs.return_all_tokens = omegaconf_select(cfg, "backbone.kwargs.return_all_tokens", False)

    # Default value for mixed_channels
    cfg.mixed_channels = omegaconf_select(cfg, "mixed_channels", False)

    # default values for slurm stuff
    cfg = add_and_assert_slurm_cfg(cfg)

    # default values for checkpointer
    cfg = Checkpointer.add_and_assert_specific_cfg(cfg)

    # default values for auto_resume
    cfg = AutoResumer.add_and_assert_specific_cfg(cfg)

    # default values for dali
    if _dali_available:
        cfg = PretrainDALIDataModule.add_and_assert_specific_cfg(cfg)

    # default values for auto_umap
    if _umap_available:
        cfg = AutoUMAP.add_and_assert_specific_cfg(cfg)

    # assert dataset parameters
    cfg = add_and_assert_dataset_cfg(cfg)

    # default values for wandb
    cfg = add_and_assert_wandb_cfg(cfg)

    # default values for pytorch lightning stuff
    cfg = add_and_assert_lightning_cfg(cfg)

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
    # classifier lr is only used for validation
    if cfg.data.val_path is not None:
        assert not OmegaConf.is_missing(cfg, "optimizer.classifier_lr")
        cfg.optimizer.classifier_lr = cfg.optimizer.classifier_lr * scale_factor
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
