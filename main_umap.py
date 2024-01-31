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

import json
import os
from pathlib import Path

import inspect
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from src.args.umap import parse_cfg
from src.data.classification_dataloader import prepare_data
from src.methods import METHODS
from src.utils.auto_umap import OfflineUMAP
from src.utils.misc import omegaconf_select, seed_everything_manual


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    # hydra doesn't allow us to add new keys for "safety"
    # set_struct(..., False) disables this behavior and allows us to add more parameters
    # without making the user specify every single thing about the model
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)

    seed_everything_manual(cfg.seed)

    assert cfg.method in METHODS, f"Choose from {METHODS.keys()}"

    # Assert pretraining rule
    ckpt_path = cfg.weights_init
    assert ckpt_path is not None, "weights_init is required"

    # load imagenet weights
    if ckpt_path == "imagenet-weights" or ckpt_path == "random-weights":
        model = METHODS[cfg.method](cfg).backbone
        model.cuda()

    # load custom pretrained weights
    else:
        assert (ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".pth") or ckpt_path.endswith(".pt")), "If not loading pretrained imagenet weights on backbone, weights_init must be a .ckpt or .pth file"
        # build paths
        split_path = ckpt_path.split("/")
        # Get the name of the folder containing the checkpoint
        ckpt_dir = "/".join(split_path[:-1])
        args_path = ckpt_dir + "/args.json"

        # load arguments
        with open(args_path) as f:
            method_args = json.load(f)
        cfg_pretrained_model = OmegaConf.create(method_args)

        # FOR MODELS TRAINED BEFORE IMPLEMENTATION OF SPECIFIC PARAMS
        cfg_pretrained_model.optimizer.token_learner_lr = omegaconf_select(cfg, "optimizer.token_learner_lr", None)
        cfg_pretrained_model.ssl_val_loss = omegaconf_select(cfg, "ssl_val_loss", False)
        cfg_pretrained_model.backbone.kwargs.return_all_tokens = omegaconf_select(cfg, "backbone.kwargs.return_all_tokens", False)

        model = METHODS[cfg.method](cfg).load_from_checkpoint(ckpt_path, strict=False, cfg=cfg_pretrained_model).backbone
        model.cuda()

    # prepare data
    train_loader, val_loader = prepare_data(
        cfg.data.dataset,
        train_data_path=cfg.data.train_path,
        val_data_path=cfg.data.val_path,
        data_format=cfg.data.format,
        batch_size=cfg.optimizer.batch_size,
        num_workers=cfg.data.num_workers,
        auto_augment=False,
        channel_strategy=cfg.channels_strategy,
        sample_ratio=cfg.data.sample_ratio,
    )

    umap = OfflineUMAP()

    # move model to the gpu
    device = "cuda:0"
    model = model.to(device)

    if cfg.data.multi_labels:
        umap.plot_multi_labels(device=device, model=model, dataloader=train_loader, plot_path=f"{cfg.name}_train_umap.pdf", channels_strategy=cfg.channels_strategy, mixed_channels=cfg.mixed_channels, return_all_tokens=cfg.backbone.kwargs.return_all_tokens)
        umap.plot_multi_labels(device=device, model=model, dataloader=val_loader, plot_path=f"{cfg.name}_val_umap.pdf", channels_strategy=cfg.channels_strategy, mixed_channels=cfg.mixed_channels, return_all_tokens=cfg.backbone.kwargs.return_all_tokens)
    
    else:
        umap.plot(device=device, model=model, dataloader=train_loader, plot_path=f"{cfg.name}_train_umap.pdf", channels_strategy=cfg.channels_strategy, mixed_channels=cfg.mixed_channels, return_all_tokens=cfg.backbone.kwargs.return_all_tokens)
        umap.plot(device=device, model=model, dataloader=val_loader, plot_path=f"{cfg.name}_val_umap.pdf", channels_strategy=cfg.channels_strategy, mixed_channels=cfg.mixed_channels, return_all_tokens=cfg.backbone.kwargs.return_all_tokens)


if __name__ == "__main__":
    main()
