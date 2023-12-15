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

import os
import logging
import csv
import time
import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

import json
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import BorderlineSMOTE

from src.args.knn import parse_cfg
from src.data.classification_dataloader import (
    prepare_dataloaders,
    prepare_datasets,
    prepare_transforms,
)
from src.methods import METHODS
from src.methods.base import BaseMethod
from src.utils.knn import WeightedKNNClassifier
from src.utils.misc import make_contiguous, omegaconf_select
from src.data.channels_strategies import modify_first_layer
from src.utils.misc import imread, check_chans
from src.utils.spherize import ZCA_corr


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
def bbbc021_nsc_classify(test_X, test_y, compounds, unbalance=False):
    """
    Description
    -----------
    Function to classify compounds in the BBBC021 dataset using the Not-Same-Compound method.

    refer to https://github.com/broadinstitute/DeepProfilerExperiments/tree/master/bbbc021
    """
    treatments = pd.DataFrame({'embeddings': list(test_X), 'labels': list(test_y), 'Compound': list(compounds)})
    model = KNeighborsClassifier(n_neighbors=1, algorithm="brute", metric="cosine")

    correct, total = 0, 0
    for i in treatments.index:
        # Leave one compound out
        mask = treatments["Compound"] != treatments.loc[i, "Compound"]
        trainFeatures = np.array(treatments['embeddings'][mask].to_list())
        trainLabels = np.array(treatments['labels'][mask].to_list())
        testFeatures = np.array(treatments['embeddings'][[i]].to_list())
        testLabelsi = np.array(treatments['labels'][[i]].to_list())

        if unbalance:
            sm = BorderlineSMOTE(random_state=42, kind="borderline-1", k_neighbors=1)
            trainFeatures, trainLabels = sm.fit_resample(trainFeatures, trainLabels)

        model.fit(trainFeatures, trainLabels)
        prediction = model.predict(testFeatures)

        # Check prediction
        if testLabelsi[0] == prediction[0]:
            correct += 1
        total += 1
    print("NSC Accuracy: {} correct out of {} = {}".format(correct, total, correct / total))
    nn_acc = correct / total
    return nn_acc

@torch.no_grad()
def bbbc021_nscb_classify(treatments, unbalance=False):
    """
    Description
    -----------
    Function to classify compounds in the BBBC021 dataset using the Not-Same-Compound-or-Batch method.

    refer to https://github.com/broadinstitute/DeepProfilerExperiments/tree/master/bbbc021
    """
    # Cholesterol-lowering and Kinase inhibitors are only in one batch
    valid_treatments = treatments[~treatments["label_names"].isin(["Cholesterol-lowering", "Kinase inhibitors"])]

    model = KNeighborsClassifier(n_neighbors=1, algorithm="brute", metric="cosine")

    correct, total = 0, 0
    for i in valid_treatments.index:
        # Leave one compound out
        mask1 = valid_treatments["Compound"] != valid_treatments.loc[i, "Compound"]
        mask2 = valid_treatments["Batch"] != valid_treatments.loc[i, "Batch"]
        mask = mask1 & mask2
        trainFeatures = np.array(valid_treatments['embeddings'][mask].to_list())
        trainLabels = np.array(valid_treatments['labels'][mask].to_list())
        testFeatures = np.array(valid_treatments['embeddings'][[i]].to_list())
        testLabelsi = np.array(valid_treatments['labels'][[i]].to_list())

        if unbalance:
            sm = BorderlineSMOTE(random_state=42, kind="borderline-1", k_neighbors=1)
            trainFeatures, trainLabels = sm.fit_resample(trainFeatures, trainLabels)

        model.fit(trainFeatures, trainLabels)
        prediction = model.predict(testFeatures)

        # Check prediction
        if testLabelsi[0] == prediction[0]:
            correct += 1
        total += 1
    print("NSCB Accuracy: {} correct out of {} = {}".format(correct, total, correct / total))
    nn_acc = correct / total
    return nn_acc

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

    # initialize logging
    logging_level = logging.INFO
    logging.basicConfig(
        level=logging_level,
        filename=f"./logs/{cfg.name}.log",
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
    )

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

        model = METHODS[cfg.method](cfg).load_from_checkpoint(ckpt_path, strict=False, cfg=cfg_pretrained_model)
        model.cuda()

    logging.info(f"Loaded {ckpt_path}")

    # ------------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------- SPECIFIC CASE for BBBC021 dataset ---------------------------------------- #
    # ------------------------------------------------------------------------------------------------------------------- #

    if cfg.data.dataset == "bbbc021":

        ###### ------------------------------- 1. Extract embeddings ------------------- ######

        # ------------------------------------ COMPOUNDS ------------------------------------ #
        comp_dataset_path = os.path.join(cfg.data.val_path, 'bbbc021_comp' + '_eval.npz')

        # prepare data
        _, T = prepare_transforms(cfg.data.dataset)
        comp_train_dataset, comp_val_dataset = prepare_datasets(
            cfg.data.dataset,
            T_train=T,
            T_val=T,
            train_data_path=comp_dataset_path,
            val_data_path=comp_dataset_path,
            data_format=cfg.data.format,
        )
        _, comp_val_loader = prepare_dataloaders(
            comp_train_dataset,
            comp_val_dataset,
            batch_size=cfg.optimizer.batch_size,
            num_workers=cfg.data.num_workers,
            channel_strategy=cfg.channels_strategy,
        )

        # extract features
        start_time = time.time()
        comp_embeddings, comp_targets = extract_features(comp_val_loader, model, mixed_channels=cfg.mixed_channels)
        comp_img_indices = comp_val_dataset.img_indices
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('\033[1;33mTotal Embed Time for Compounds: {} \033[0m'.format(total_time_str))

        # ------------------------------------ DMSO ------------------------------------ #
        dmso_dataset_path = os.path.join(cfg.data.val_path, 'bbbc021_dmso' + '_eval.npz')

        # prepare data
        _, T = prepare_transforms(cfg.data.dataset)
        dmso_train_dataset, dmso_val_dataset = prepare_datasets(
            cfg.data.dataset,
            T_train=T,
            T_val=T,
            train_data_path=dmso_dataset_path,
            val_data_path=dmso_dataset_path,
            data_format=cfg.data.format,
        )
        _, dmso_val_loader = prepare_dataloaders(
            dmso_train_dataset,
            dmso_val_dataset,
            batch_size=cfg.optimizer.batch_size,
            num_workers=cfg.data.num_workers,
            channel_strategy=cfg.channels_strategy,
        )

        # extract features
        start_time = time.time()
        dmso_embeddings, dmso_targets = extract_features(dmso_val_loader, model, mixed_channels=cfg.mixed_channels)
        dmso_img_indices = dmso_val_dataset.img_indices
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('\033[1;33mTotal Embed Time for DMSO: {} \033[0m'.format(total_time_str))


        ###### ------------------------------- 2. Prepare embeddings ------------------------------- ######
        batch_level = 'plate'

        dmso_batches = dmso_val_dataset.get_batches(dmso_img_indices, level=batch_level)
        dmso_embed_df_batches = pd.DataFrame({'embeddings':dmso_embeddings.tolist(), 'batches': dmso_batches, 'inds': dmso_img_indices})

        comp_batches = comp_val_dataset.get_batches(comp_img_indices, level=batch_level)
        batches_all = list(set(comp_batches))
        embed_df_batches = pd.DataFrame({'embeddings':comp_embeddings.tolist(), 'batches': comp_batches, 'inds': comp_img_indices})


        ##### ------------------------------- 3. Spherize embeddings ------------------------------- ######
        # refer to https://github.com/jump-cellpainting/2021_Chandrasekaran_submitted/blob/main/benchmark/old_notebooks/3.spherize_profiles.ipynb

        embeddings_batches = []
        inds_batches = []

        for batchi in batches_all:
            dmso_embeddings_batch = np.array(dmso_embed_df_batches.query('batches==@batchi').embeddings.to_list())
            spherizer = ZCA_corr()
            spherizer.fit(dmso_embeddings_batch)

            embeddings_batch = np.array(embed_df_batches.query('batches==@batchi').embeddings.to_list())
            embeddings_batch = spherizer.transform(embeddings_batch)
            embeddings_batches += list(embeddings_batch)
            inds_batch = np.array(embed_df_batches[comp_batches == batchi].inds.to_list())
            inds_batches += list(inds_batch)

        embeddings = embeddings_batches
        inds = np.array(inds_batches)

        labels = comp_val_dataset.get_labels(inds)
        compounds = comp_val_dataset.get_compounds(inds)

        ##### 4. Aggregate on treatment #####
        concentrations = comp_val_dataset.get_concentrations(inds)
        batches = comp_val_dataset.get_batches(inds)
        embed_df = pd.DataFrame({'embeddings':embeddings, 'compounds':compounds, 'concentrations':concentrations,'labels':labels, 'batches': batches})
        embed_df['treatment'] = embed_df.apply(lambda x: f"{x.compounds}-{x.concentrations}", axis=1)
        labels = embed_df.groupby(by=['treatment']).sample(n=1).labels.to_numpy()
        compounds = embed_df.groupby(by=['treatment']).sample(n=1).compounds.to_numpy()
        treatments = embed_df.groupby(by=['treatment']).sample(n=1).treatment.to_numpy()
        embeddings = np.array(embed_df.groupby(by=['treatment']).embeddings.mean().to_list())

        labelmap_path = os.path.join(cfg.data.val_path, 'bbbc021_comp_labelmap.npy')
        label_map = np.load(labelmap_path, allow_pickle=True).item()
        print('>>>> Label map:{}, Totally {} classes'.format(label_map, len(label_map)))

        ###### 5. KNN classify ######
        # classify_dir = os.path.join(output_dir, 'evaluation')
        # if not os.path.isdir(classify_dir): os.makedirs(classify_dir)

        df_results = bbbc021_nsc_classify(test_X=embeddings, test_y=labels, compounds=compounds)

        label_names = [label_map[i] for i in labels]
        treatments_df = pd.DataFrame({'embeddings': list(embeddings), 'labels': list(labels), 'label_names': label_names,
                                    'Compound': list(compounds), 'treatments': list(treatments)})
        treatments_df["Batch"] = ""
        for i in treatments_df.index:
            result = embed_df.query("treatment == '{}'".format(treatments_df.loc[i, "treatments"]))
            treatments_df.loc[i, "Batch"] = ",".join(result["batches"].unique())
        df_results = bbbc021_nscb_classify(treatments=treatments_df)


    # ------------------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------- OTHER CASES --------------------------------------------------- #
    # ------------------------------------------------------------------------------------------------------------------- #

    else:
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
