import logging
import os
import sys
import cv2
import random
import colorsys

import hydra
import torch
import torchvision
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import numpy as np
from PIL import Image

from omegaconf import DictConfig, OmegaConf

from src.methods.base import BaseMethod
from src.args.attn import parse_cfg
from src.data.channels_strategies import modify_first_layer
from src.utils.misc import seed_everything_manual

from src.backbones.vit.vit_attn_viz import VisionTransformerAttnViz


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return

@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    # hydra doesn't allow us to add new keys for "safety"
    # set_struct(..., False) disables this behavior and allows us to add more parameters
    # without making the user specify every single thing about the model
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_everything_manual(cfg.seed)

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

    # load custom pretrained weights
    else:
        assert (
            ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".pth") or ckpt_path.endswith(".pt")
        ), "If not loading pretrained imagenet weights on backbone, pretrained_feature_extractor must be a .ckpt or .pth file"
        pretrained = False

        # Use custom ViT model for one channel images to vizualize attention maps
        if cfg.channels_strategy == "one_channel":
            #TODO: Add support for other ViT models while waiting for a fix in timm library
            # Currently, case only supported for ViT-Tiny
            backbone = VisionTransformerAttnViz(patch_size=16, embed_dim=192, depth=12, num_heads=3, num_classes=0, dynamic_img_size=True)  # Adapt parameters as needed
            state = torch.load(ckpt_path, map_location="cpu")["state_dict"]

            # You might need to adapt the state dict keys depending on how they are saved
            backbone.load_state_dict(state, strict=False)

        else:
            backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
            backbone = modify_first_layer(backbone=backbone, cfg=cfg, pretrained=pretrained)
            state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            for k in list(state.keys()):
                if "encoder" in k:
                    state[k.replace("encoder", "backbone")] = state[k]
                    logging.warn(
                        "You are using an older checkpoint. Use a new one as some issues might arrise."
                    )
                if "backbone" in k:
                    state[k.replace("backbone.", "")] = state[k]
                del state[k]
            backbone.load_state_dict(state, strict=False)

        
    model = backbone
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    logging.info(f"Loaded {ckpt_path}")

    # open image
    if cfg.image_path is None:
        # user has not specified any image - we use our own image
        raise ValueError(f"Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
    elif os.path.isfile(cfg.image_path):
        with open(cfg.image_path, 'rb') as f:
            img = Image.open(f)
            img = np.array(img).astype(np.float32)
    else:
        print(f"Provided image path {cfg.image_path} is non valid.")
        sys.exit(1)

    transform =A.Compose(
        [
            A.Resize(height=cfg.image_size, width=cfg.image_size),  # resize shorter
            ToTensorV2(),
        ]
    )

    img = transform(image=img)["image"]

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % cfg.patch_size, img.shape[2] - img.shape[2] % cfg.patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = num_patches = img.shape[-2] // cfg.patch_size
    h_featmap = img.shape[-1] // cfg.patch_size

    # Retrieve attention maps
    attentions = model.get_last_selfattention(img.to(device))

    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    if cfg.threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - cfg.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=cfg.patch_size, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=cfg.patch_size, mode="nearest")[0].cpu().numpy()

    # save attentions heatmaps
    os.makedirs(cfg.output_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(cfg.output_dir, "img.png"))
    for j in range(nh):
        fname = os.path.join(cfg.output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")

    # save mean attention heatmap
    mean_attn = np.mean(attentions, axis=0)
    fname = os.path.join(cfg.output_dir, "attn-mean.png")
    plt.imsave(fname=fname, arr=mean_attn, format='png')
    print(f"{fname} saved.")

    if cfg.threshold is not None:
        image = skimage.io.imread(os.path.join(cfg.output_dir, "img.png"))
        for j in range(nh):
            display_instances(image, th_attn[j], fname=os.path.join(cfg.output_dir, "mask_th" + str(cfg.threshold) + "_head" + str(j) +".png"), blur=False)

if __name__ == "__main__":
    main()
