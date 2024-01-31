import omegaconf
from src.methods.base import BaseMethod
from src.utils.misc import omegaconf_select


def parse_cfg(cfg: omegaconf.DictConfig):
    """Parses feature extractor, dataset, pytorch lightning, linear eval specific and additional args.

    First adds an arg for the pretrained feature extractor, then adds dataset, pytorch lightning
    and linear eval specific args. If wandb is enabled, it adds checkpointer args. Finally, adds
    additional non-user given parameters.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    # backbone
    assert not omegaconf.OmegaConf.is_missing(cfg, "backbone.name")
    assert cfg.backbone.name in BaseMethod._BACKBONES

    # backbone kwargs
    cfg.backbone.kwargs = omegaconf_select(cfg, "backbone.kwargs", {})

    # Default value for return_all_tokens
    cfg.backbone.kwargs.return_all_tokens = omegaconf_select(cfg, "backbone.kwargs.return_all_tokens", False)

    # Batch weights
    assert not omegaconf.OmegaConf.is_missing(cfg, "pretrained_feature_extractor")

    # Method
    cfg.pretrain_method = omegaconf_select(cfg, "pretrain_method", None)

    # Image path
    assert not omegaconf.OmegaConf.is_missing(cfg, "image_path")

    # Image size
    cfg.image_size = omegaconf_select(cfg, "image_size", 224)

    # Patch size
    cfg.patch_size = omegaconf_select(cfg, "patch_size", 16)

    # Patch size
    cfg.threshold =  omegaconf_select(cfg, "threshold", None)

    # Output dir
    cfg.output_dir = omegaconf_select(cfg, "output_dir", '.')

    # Seed
    cfg.seed = omegaconf_select(cfg, "seed", None)

    return cfg
