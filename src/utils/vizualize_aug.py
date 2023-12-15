import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def vizualize_aug(save_dir, raw_image, augmented_image, index_to_query):
    """
    Function to vizualize the transformation of the image and save it in a file.

    Parameters
    ----------
    root_dir : str
        Path to the root directory of the project.
    raw_image : torch.Tensor
        Raw image.
    augmented_image : torch.Tensor
        Augmented image.
    index_to_query : int
        Index of the image to vizualize.
    """

    # ========= IF YOU WANT TO VIZUALIZE THE TRANSFORMATION YOU DO ========= #

    # Pass image to array and save in file
    raw_image = np.array(raw_image)
    #raw_image = np.transpose(raw_image, (1, 2, 0))
    #normalized_raw_array = (raw_image - np.min(raw_image)) / (np.max(raw_image) - np.min(raw_image))
    #scaled_raw_array = (normalized_raw_array * 255).astype(np.uint8)

    # Pass image to array and save in file
    if isinstance(augmented_image, list): # If the image is a list of num_crops, take the last element
        augmented_image = augmented_image[-1]
    augmented_image = np.array(augmented_image)
    augmented_image = np.transpose(augmented_image, (1, 2, 0))
    #normalized_aug_array = (augmented_image - np.min(augmented_image)) / (np.max(augmented_image) - np.min(augmented_image))
    #scaled_aug_array = (normalized_aug_array * 255).astype(np.uint8)

    # Check if the directory exists
    if not os.path.exists(f'{save_dir}/aug_images'):
        os.makedirs(f'{save_dir}/aug_images')

    # Build a matplotlib figure with 2 columns per channel
    plt.rcParams['figure.figsize'] = [30, 30]
    plt.rcParams['figure.dpi'] = 500

    num_channels = raw_image.shape[-1]
    fig, axes = plt.subplots(num_channels, 2)

    if num_channels == 1:
        axes[0].imshow(raw_image[:, :, 0], cmap='gray')
        axes[1].imshow(augmented_image[:, :, 0], cmap='gray')

        axes[0].set_title('Raw image')
        axes[1].set_title('Augmented image')

    elif num_channels > 1:
        for channel in range(0, augmented_image.shape[-1]):
            axes[channel, 0].imshow(raw_image[:, :, channel], cmap='gray')
            axes[channel, 1].imshow(augmented_image[:, :, channel], cmap='gray')

        axes[0, 0].set_title('Raw image')
        axes[0, 1].set_title('Augmented image')

    fig.savefig(f'{save_dir}/aug_images/{index_to_query}.png')
