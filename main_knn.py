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

import csv

import hydra
from omegaconf import DictConfig, OmegaConf

import json
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.args.knn import parse_cfg
from src.data.classification_dataloader import (
    prepare_dataloaders,
    prepare_datasets,
    prepare_transforms,
)
from src.methods import METHODS
from src.utils.knn import WeightedKNNClassifier
from src.utils.misc import omegaconf_select, seed_everything_manual
from src.data.channels_strategies import modify_first_layer

@torch.no_grad()
def extract_features(loader: DataLoader, model: nn.Module, mixed_channels=False) -> Tuple[torch.Tensor]:
    """Extract features from a data loader using a model.

    Args:
        loader (DataLoader): dataloader for a dataset.
        model (nn.Module): torch module used to extract features.
        channels_strategy (str): strategy to modify the first layer of the model.

    Returns:
        Tuple(torch.Tensor): tuple containing the backbone features, projector features and labels.
    """

    model.eval()

    # Overwrite model "mixed_channels" parameter for evaluation on "normal" datasets with uniform channels size
    model.mixed_channels = mixed_channels

    backbone_features, labels = [], []
    for batch in tqdm(loader):
        feats, targets = model.extract_features(batch)
        backbone_features.append(feats.detach().cpu())
        labels.append(targets.detach().cpu())
    model.train()
    backbone_features = torch.cat(backbone_features)
    labels = torch.cat(labels)
    return backbone_features, labels

@torch.no_grad()
def run_knn(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    k: int,
    T: float,
    distance_fx: str,
) -> Tuple[float]:
    """Runs offline knn on a train and a test dataset.

    Args:
        train_features (torch.Tensor, optional): train features.
        train_targets (torch.Tensor, optional): train targets.
        test_features (torch.Tensor, optional): test features.
        test_targets (torch.Tensor, optional): test targets.
        k (int): number of neighbors.
        T (float): temperature for the exponential. Only used with cosine
            distance.
        distance_fx (str): distance function.

    Returns:
        Tuple[float]: tuple containing the the knn acc@1 and acc@5 for the model.
    """

    # build knn
    knn = WeightedKNNClassifier(
        k=k,
        T=T,
        distance_fx=distance_fx,
    )

    # add features
    knn(
        train_features=train_features,
        train_targets=train_targets,
        test_features=test_features,
        test_targets=test_targets,
    )

    # compute
    acc1, acc5 = knn.compute()

    # free up memory
    del knn

    return acc1, acc5

@torch.no_grad()
def results_to_csv(csv_filename:str, cfg:DictConfig, train_features: torch.Tensor, train_targets: torch.Tensor, test_features: torch.Tensor, test_targets: torch.Tensor):
    # Open the CSV file for writing
    with open(csv_filename, mode="w", newline="") as csv_file:
        # Create a CSV writer
        csv_writer = csv.writer(csv_file)

        # Write the header row to the CSV file
        csv_writer.writerow(["Feature Type", "Distance Function", "k", "T", "acc@1", "acc@5"])

        # run k-nn for all possible combinations of parameters
        for feat_type in cfg.knn_eval_offline.feature_type:
            print(f"\n### {feat_type.upper()} ###")
            for k in cfg.knn_eval_offline.k:
                for distance_fx in cfg.knn_eval_offline.distance_function:
                    temperatures = cfg.knn_eval_offline.temperature if distance_fx == "cosine" else [None]
                    for T in temperatures:
                        print("---")
                        print(f"Running k-NN with params: distance_fx={distance_fx}, k={k}, T={T}...")
                        acc1, acc5 = run_knn(
                            train_features=train_features[feat_type],
                            train_targets=train_targets,
                            test_features=test_features[feat_type],
                            test_targets=test_targets,
                            k=k,
                            T=T,
                            distance_fx=distance_fx,
                        )
                        print(f"Result: acc@1={acc1}, acc@5={acc5}")

                        # Write the results to the CSV file
                    csv_writer.writerow([feat_type, distance_fx, k, T, acc1, acc5])


# --------------------- main func --------------------- #
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
        model = METHODS[cfg.method](cfg)
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

        # FOR MODELS TRAINED BEFORE IMPLEMENTATION OF CERTAIN PARAMETERS
        cfg_pretrained_model.optimizer.token_learner_lr = omegaconf_select(cfg, "optimizer.token_learner_lr", None)
        cfg_pretrained_model.ssl_val_loss = omegaconf_select(cfg, "ssl_val_loss", False)
        cfg_pretrained_model.backbone.kwargs.return_all_tokens = omegaconf_select(cfg, "backbone.kwargs.return_all_tokens", False)

        model = METHODS[cfg.method](cfg).load_from_checkpoint(ckpt_path, strict=False, cfg=cfg_pretrained_model)

        # modify first layer AFTER loading pretrained weights for Standard architecture (trained on 3-Channels/RGB images)
        if not (cfg.channels_strategy == "one_channel" or cfg.channels_strategy == "multi_channels"):
            model.backbone = modify_first_layer(backbone=model.backbone, cfg=cfg, pretrained=False)
        
        model.cuda()

    # prepare data
    _, T = prepare_transforms(cfg.data.dataset)

    train_dataset, val_dataset = prepare_datasets(
        cfg.data.dataset,
        T_train=T,
        T_val=T,
        train_data_path=cfg.data.train_path,
        val_data_path=cfg.data.val_path,
        data_format=cfg.data.format,
        sample_ratio=cfg.data.sample_ratio,
    )

    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=cfg.optimizer.batch_size,
        num_workers=cfg.data.num_workers,
        channel_strategy=cfg.channels_strategy,
    )

    # extract train features
    train_features_backbone, train_targets = extract_features(train_loader, model, mixed_channels=cfg.mixed_channels)
    train_features = {"backbone": train_features_backbone}

    # extract test features
    test_features_backbone, test_targets = extract_features(val_loader, model, mixed_channels=cfg.mixed_channels)
    test_features = {"backbone": test_features_backbone}

    # Write results in .csv file
    csv_filename = f"{cfg.name}_knn_offline_eval.csv"
    results_to_csv(csv_filename=csv_filename, cfg=cfg, train_features=train_features, train_targets=train_targets, test_features=test_features, test_targets=test_targets)
    print("Results of KNN offline eval written to", csv_filename)


if __name__ == "__main__":
    main()
