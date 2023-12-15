import torch
import random
from torch.utils.data import Dataset
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed


class RandomDiscarder(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        *buffer, image, label  = self.dataset[index][-2:]
        num_channels = image.shape[0]

        # Randomly decide the number of channels to discard (0 or 1)
        num_channels_to_discard = random.randint(0, 2)

        if num_channels_to_discard > 0 and num_channels > 1:
            # Randomly select the indices of channels to discard
            drop_channel = torch.randint(0, num_channels, (1,))
            channel_indices = torch.arange(num_channels) != drop_channel
            image = image[channel_indices, :, :]

        return image, label

    def __len__(self):
        return len(self.dataset)


def one_channel_collate_fn(batch):
    """
    Collate function in Dataloader to transform a batch of images into a batch of one-channel images.
    Can handle multicrops setting.

    Args:
        batch (list): A list of tuples of (List[image_crops], label)

    Returns:
        batched_images (List): A List of tensors of shape [(batch_size*num_channels, 1, height, width), ...]
        batched_labels (torch.tensor): A tensor of labels of shape (batch_size, )
        batched_num_channels (List): A List of integers of shape [(batch_size, ), ...]
    """

    batched_labels = []
    num_crops = len(batch[0][-2:][0]) if isinstance(batch[0][-2:][0], list) else 1
    crop_lists = [[] for _ in range(num_crops)]
    num_channels_lists = [[] for _ in range(num_crops)]

    for batch_tuple in batch:
        # some datasets will return a 3-fold tuple of (index, image, label), while some will return only a 2-fold tuple of (image, label)
        # we need to handle both cases so we extract the last two variables from the tuple
        *buffer, image_list, label = batch_tuple[-2:]

        # If image_list is a tensor, convert it to a list
        if isinstance(image_list, torch.Tensor):
            image_list = [image_list]

        # Iterate over the list of images and extract the channels
        for i, image_crop in enumerate(image_list):
            num_channels = image_crop.shape[0]
            num_channels_lists[i].append(num_channels)

            for channel in range(num_channels):
                channel_image = image_crop[channel, :, :].unsqueeze(0)
                crop_lists[i].append(channel_image)

            # # Create a new list containing the unsqueezed channels for this image
            # unsqueezed_channels = [image[channel, :, :] for channel in range(num_channels)]

            # # Add the unsqueezed channels to the list of images
            # batched_images.extend(unsqueezed_channels)

        # Shape: (batch_size, ) since always the same label for all crops of the same image
        batched_labels.append(label)

    for i in range(num_crops):
        crop_lists[i] = torch.cat(crop_lists[i], dim=0).unsqueeze(1) # Shape: (X*num_channels, 1, height, width)

    # Flatten crop_lists if len(crop_lists) == 1
    crop_lists = crop_lists[0] if len(crop_lists) == 1 else crop_lists

    batched_labels = torch.tensor(batched_labels)

    return crop_lists, batched_labels, num_channels_lists


def modify_first_layer(backbone, cfg, pretrained):
    # Check the channels
    img_channels = 1 if cfg.channels_strategy == "one_channel" or cfg.channels_strategy == "multi_channels" else int(cfg.data.max_img_channels)

    # retrieve backbone type
    backbone_type = backbone.__class__.__name__

    # ====================== resnet ====================== #
    if backbone_type == "ResNet":

        # remove fc layer
        backbone.fc = nn.Identity()

        # Do nothing if channels is RGB
        if img_channels == 3:
            return backbone

        conv_attrs = [
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "groups",
            "bias",
            "padding_mode",
        ]
        conv1_defs = {attr: getattr(backbone.conv1, attr) for attr in conv_attrs}

        # Weights duplication
        pretrained_weight = backbone.conv1.weight.data #retrieve pretrained weights
        pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels] #repeat weights for each channel

        # Modify conv1 layer
        backbone.conv1 = nn.Conv2d(img_channels, **conv1_defs)
        cifar = cfg.data.dataset in ["cifar10", "cifar100"]

        # Special case for cifar
        if cifar:
            backbone.conv1 = nn.Conv2d(
                img_channels, 64, kernel_size=3, stride=1, padding=2, bias=False
            )
            backbone.maxpool = nn.Identity()
        if pretrained:
            backbone.conv1.weight.data = pretrained_weight

    # ====================== VisionTransformer ====================== #
    elif backbone_type == "VisionTransformer":

        # Do nothing if channels is RGB
        if img_channels == 3:
            return backbone

        patch_embed_attrs = ["img_size", "patch_size", "dynamic_img_pad", "flatten", "strict_img_size", "output_fmt"]
        patch_defs = {attr: getattr(backbone.patch_embed, attr) for attr in patch_embed_attrs}
        patch_defs["embed_dim"] = backbone.embed_dim

        pretrained_weight = backbone.patch_embed.proj.weight.data
        if backbone.patch_embed.proj.bias is not None:
            pretrained_bias = backbone.patch_embed.proj.bias.data
        pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]

        backbone.patch_embed = PatchEmbed(in_chans=img_channels, **patch_defs)
        if pretrained:
            backbone.patch_embed.proj.weight.data = pretrained_weight
            if backbone.patch_embed.proj.bias is not None:
                backbone.patch_embed.proj.bias.data = pretrained_bias

    return backbone
