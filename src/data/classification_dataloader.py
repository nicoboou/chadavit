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
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torchvision
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import STL10, ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from src.data.channels_strategies import RandomDiscarder, one_channel_collate_fn

try:
    from src.data.custom_datasets import H5Dataset
except ImportError:
    _h5_available = False
else:
    _h5_available = True

from src.data.custom_datasets import IDRCell100K, BBBC021, BloodMNIST, BBBC048, CyclOPS, TissueMNIST, Transloc, BBBC021xBray, MTBenchReg, Bray

class AlbumentationTransform:
    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, image):
        return self.transform(image=image)["image"]

def prepare_transforms(dataset: str) -> Tuple[nn.Module, nn.Module]:
    """Prepares pre-defined train and test transformation pipelines for some datasets.

    Args:
        dataset (str): dataset name.

    Returns:
        Tuple[nn.Module, nn.Module]: training and validation transformation pipelines.
    """

    cifar_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    stl_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=96, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize((96, 96)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    imagenet_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
    }

    idrcell100k_pipeline = {
        "T_train": A.Compose(
            [
                A.augmentations.crops.transforms.RandomResizedCrop(height=224,width=224,scale=(0.08, 1.0),interpolation=cv2.INTER_CUBIC,p=1.0),
                A.HorizontalFlip(p=0.5),
                ToTensorV2(),
            ]
        ),
        "T_val": A.Compose(
            [
                A.Resize(height=256, width=256),  # resize shorter
                A.CenterCrop(height=224,width=224),  # take center crop
                ToTensorV2(),
            ]
        ),
    }

    bray_pipeline = {
        "T_train": A.Compose(
            [
                A.augmentations.crops.transforms.RandomResizedCrop(height=224,width=224,scale=(0.08, 1.0),interpolation=cv2.INTER_CUBIC,p=1.0),
                A.HorizontalFlip(p=0.5),
                ToTensorV2(),
            ]
        ),
        "T_val": A.Compose(
            [
                A.Resize(height=256, width=256),  # resize shorter
                A.CenterCrop(height=224,width=224),  # take center crop
                ToTensorV2(),
            ]
        ),
    }

    bbbc021_pipeline = {
        "T_train": A.Compose(
            [
                A.augmentations.crops.transforms.RandomResizedCrop(height=224,width=224,scale=(0.2, 1.0),interpolation=cv2.INTER_CUBIC,p=1.0),
                A.HorizontalFlip(p=0.5),
                ToTensorV2(),
            ]
        ),
        "T_val": A.Compose(
            [
                A.Resize(height=256, width=256),  # resize shorter
                A.CenterCrop(height=224,width=224),  # take center crop
                ToTensorV2(),
            ]
        ),
    }

    bbbc021_x_bray_pipeline = {
        "T_train": A.Compose(
            [
                A.Resize(height=256, width=256),  # resize shorter
                A.CenterCrop(height=224,width=224),  # take center crop
                ToTensorV2(),
            ]
        ),
        "T_val": A.Compose(
            [
                A.Resize(height=256, width=256),  # resize shorter
                A.CenterCrop(height=224,width=224),  # take center crop
                ToTensorV2(),
            ]
        ),
    }

    bloodmnist_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
            ]
        ),
    }

    tissuemnist_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
            ]
        ),
    }

    bbbc048_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
            ]
        ),
    }

    cyclops_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
            ]
        ),
    }

    transloc_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
            ]
        ),
    }

    mtbenchreg_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
            ]
        ),
    }

    pipelines = {
        "cifar10": cifar_pipeline,
        "cifar100": cifar_pipeline,
        "stl10": stl_pipeline,
        "imagenet100": imagenet_pipeline,
        "imagenet": imagenet_pipeline,
        "idrcell100k": idrcell100k_pipeline,
        "bbbc021": bbbc021_pipeline,
        "bloodmnist": bloodmnist_pipeline,
        "bbbc048": bbbc048_pipeline,
        "cyclops": cyclops_pipeline,
        "tissuemnist": tissuemnist_pipeline,
        "transloc": transloc_pipeline,
        "bbbc021xbray": bbbc021_x_bray_pipeline,
        "mtbenchreg": mtbenchreg_pipeline,
        "bray": bray_pipeline,
    }

    assert dataset in pipelines

    pipeline = pipelines[dataset]
    if dataset in ["idrcell100k", "bbbc021", "bbbc021xbray", "bray"]:
        T_train = AlbumentationTransform(transform=pipeline["T_train"])
        T_val = AlbumentationTransform(transform=pipeline["T_val"])
    else:
        T_train = pipeline["T_train"]
        T_val = pipeline["T_val"]

    return T_train, T_val

def prepare_datasets(
    dataset: str,
    T_train: Callable,
    T_val: Callable,
    train_data_path: Optional[Union[str, Path]] = None,
    val_data_path: Optional[Union[str, Path]] = None,
    data_format: Optional[str] = "image_folder",
    download: bool = True,
    data_fraction: float = -1.0,
    sample_ratio=1.0,
) -> Tuple[Dataset, Dataset]:
    """Prepares train and val datasets.

    Args:
        dataset (str): dataset name.
        T_train (Callable): pipeline of transformations for training dataset.
        T_val (Callable): pipeline of transformations for validation dataset.
        train_data_path (Optional[Union[str, Path]], optional): path where the
            training data is located. Defaults to None.
        val_data_path (Optional[Union[str, Path]], optional): path where the
            validation data is located. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder".
            Possible values are "image_folder" and "h5".
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.

    Returns:
        Tuple[Dataset, Dataset]: training dataset and validation dataset.
    """

    if train_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        train_data_path = sandbox_folder / "datasets"

    if val_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        val_data_path = sandbox_folder / "datasets"

    assert dataset in ["cifar10", "cifar100", "stl10", "imagenet", "imagenet100", "bbbc021", "idrcell100k", "bloodmnist", "bbbc048", "cyclops", "tissuemnist", "transloc", "bbbc021xbray", "mtbenchreg", "bray"]

    # ----------- Natural images datasets ----------- #
    if dataset in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = DatasetClass(
            train_data_path,
            train=True,
            download=download,
            transform=T_train,
        )

        val_dataset = DatasetClass(
            val_data_path,
            train=False,
            download=download,
            transform=T_val,
        )

    elif dataset == "stl10":
        train_dataset = STL10(
            train_data_path,
            split="train",
            download=True,
            transform=T_train,
        )
        val_dataset = STL10(
            val_data_path,
            split="test",
            download=download,
            transform=T_val,
        )

    elif dataset in ["imagenet", "imagenet100"]:
        if data_format == "h5":
            assert _h5_available
            train_dataset = H5Dataset(dataset, train_data_path, T_train)
            val_dataset = H5Dataset(dataset, val_data_path, T_val)
        else:
            train_dataset = ImageFolder(train_data_path, T_train)
            val_dataset = ImageFolder(val_data_path, T_val)

    # ----------- No Label dataset (SSL) ----------- #
    elif dataset == "idrcell100k":
        train_dataset = IDRCell100K(root_dir=train_data_path, train=True, transform=T_train, sample_ratio=sample_ratio)
        val_dataset = IDRCell100K(root_dir=val_data_path, train=False, transform=T_val, sample_ratio=sample_ratio)

    elif dataset == "bray":
        train_dataset = Bray(root_dir=train_data_path, train=True, transform=T_train, sample_ratio=sample_ratio)
        val_dataset = Bray(root_dir=val_data_path, train=False, transform=T_val, sample_ratio=sample_ratio)

    # ----------- Classification datasets ----------- #
    elif dataset == "bbbc021":
        train_dataset = BBBC021(root_dir=train_data_path, train=True, transform=T_train, sample_ratio=sample_ratio)
        val_dataset = BBBC021(root_dir=val_data_path, train=False, transform=T_val, sample_ratio=sample_ratio)

    elif dataset == "bloodmnist":
        train_dataset = BloodMNIST(root_dir=train_data_path, train=True, transform=T_train, sample_ratio=sample_ratio)
        val_dataset = BloodMNIST(root_dir=val_data_path, train=False, transform=T_val, sample_ratio=sample_ratio)

    elif dataset == "bbbc048":
        train_dataset = BBBC048(root_dir=train_data_path, train=True, transform=T_train, sample_ratio=sample_ratio)
        val_dataset = BBBC048(root_dir=val_data_path, train=False, transform=T_val, sample_ratio=sample_ratio)

    elif dataset == "cyclops":
        train_dataset = CyclOPS(root_dir=train_data_path, train=True, transform=T_train, sample_ratio=sample_ratio)
        val_dataset = CyclOPS(root_dir=val_data_path, train=False, transform=T_val, sample_ratio=sample_ratio)

    elif dataset == "tissuemnist":
        train_dataset = TissueMNIST(root_dir=train_data_path, train=True, transform=T_train, sample_ratio=sample_ratio)
        val_dataset = TissueMNIST(root_dir=val_data_path, train=False, transform=T_val, sample_ratio=sample_ratio)

    # ----------- Regression dataset ----------- #
    elif dataset == "transloc":
        train_dataset = Transloc(root_dir=train_data_path, train=True, transform=T_train, sample_ratio=sample_ratio)
        val_dataset = Transloc(root_dir=val_data_path, train=False, transform=T_val, sample_ratio=sample_ratio)

    elif dataset == "mtbenchreg":
        # No sample ratio since we want to use all the data because dataset is already quite small
        train_dataset = MTBenchReg(root_dir=train_data_path, train=True, transform=T_train)
        val_dataset = MTBenchReg(root_dir=val_data_path, train=False, transform=T_val)

    # ----------- UMAP Vizualization dataset ----------- #
    elif dataset == "bbbc021xbray":
        train_dataset = BBBC021xBray(root_dir=train_data_path, train=True, transform=T_train, sample_ratio=sample_ratio)
        val_dataset = BBBC021xBray(root_dir=val_data_path, train=False, transform=T_val, sample_ratio=sample_ratio)


    if data_fraction > 0:
        assert data_fraction < 1, "Only use data_fraction for values smaller than 1."
        data = train_dataset.samples
        files = [f for f, _ in data]
        labels = [l for _, l in data]

        from sklearn.model_selection import train_test_split

        files, _, labels, _ = train_test_split(
            files, labels, train_size=data_fraction, stratify=labels, random_state=42
        )
        train_dataset.samples = [tuple(p) for p in zip(files, labels)]

    return train_dataset, val_dataset


def prepare_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 4,
    channel_strategy: str = None,
) -> Tuple[DataLoader, DataLoader]:
    """Wraps a train and a validation dataset with a DataLoader.

    Args:
        train_dataset (Dataset): object containing training data.
        val_dataset (Dataset): object containing validation data.
        batch_size (int): batch size.
        num_workers (int): number of parallel workers.
    Returns:
        Tuple[DataLoader, DataLoader]: training dataloader and validation dataloader.
    """

    if channel_strategy == "one_channel" or channel_strategy == "multi_channels":
        collate_fn = one_channel_collate_fn
    else:
        collate_fn = None

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


# ==================== main function ==================== #
def prepare_data(
    dataset: str,
    train_data_path: Optional[Union[str, Path]] = None,
    val_data_path: Optional[Union[str, Path]] = None,
    data_format: Optional[str] = "image_folder",
    batch_size: int = 64,
    num_workers: int = 4,
    download: bool = True,
    data_fraction: float = -1.0,
    auto_augment: bool = False,
    channel_strategy: str = None,
    sample_ratio: float = 1.0,
) -> Tuple[DataLoader, DataLoader]:
    """Prepares transformations, creates dataset objects and wraps them in dataloaders.

    Args:
        dataset (str): dataset name.
        train_data_path (Optional[Union[str, Path]], optional): path where the
            training data is located. Defaults to None.
        val_data_path (Optional[Union[str, Path]], optional): path where the
            validation data is located. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder".
            Possible values are "image_folder" and "h5".
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of parallel workers. Defaults to 4.
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.
        auto_augment (bool, optional): use auto augment following timm.data.create_transform.
            Defaults to False.

    Returns:
        Tuple[DataLoader, DataLoader]: prepared training and validation dataloader.
    """

    # prepare transformations
    T_train, T_val = prepare_transforms(dataset)
    if auto_augment:
        T_train = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=None,  # don't use color jitter when doing random aug
            auto_augment="rand-m9-mstd0.5-inc1",  # auto augment string
            interpolation="bicubic",
            re_prob=0.25,  # random erase probability
            re_mode="pixel",
            re_count=1,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )

    # prepare datasets
    train_dataset, val_dataset = prepare_datasets(
        dataset,
        T_train,
        T_val,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        data_format=data_format,
        download=download,
        data_fraction=data_fraction,
        sample_ratio=sample_ratio,
    )

    # train_dataset = RandomDiscarder(train_dataset)
    # val_dataset = RandomDiscarder(val_dataset)

    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        channel_strategy=channel_strategy,
    )

    return train_loader, val_loader
