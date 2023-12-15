import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import RandomApply, InterpolationMode
import torchvision.transforms.functional as F
import albumentations as A

# ================================= Need to apply rand_apply ================================= #

class ResizeMultiChannels(nn.Module):
    """Resizes a multichannel image."""
    def __init__(self, size):
        """
        Args:
            size(int or tuple): size of
        """
        self.size = size
    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be resized.

        Returns:
            Resized image (torch.Tensor).
        """
        tensor_img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        resized_tensor = F.resize(tensor_img, self.size)
        resized_img = resized_tensor.numpy().transpose(1, 2, 0)
        return resized_img

class CenterCropMultiChannels(nn.Module):
    """Performs center cropping on a multichannel image."""
    def __init__(self, size):
        """
        Args:
            size (int or tuple): Desired output size of the crop. If size is an int,
                a square crop of size (size, size) is returned. If size is a tuple,
                it should contain (height, width).
        """
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be center cropped.

        Returns:
            Center cropped image (numpy.ndarray).
        """
        tensor_img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        cropped_tensor = F.center_crop(tensor_img, self.size)
        cropped_img = cropped_tensor.numpy().transpose(1, 2, 0)
        return cropped_img

class VerticalFlipMultiChannels(nn.Module):
    """Performs a vertical flip on a multichannel image."""
    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be vertically flipped.

        Returns:
            Vertically flipped image (numpy.ndarray).
        """
        tensor_img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        flipped_tensor = F.vflip(tensor_img)
        flipped_img = flipped_tensor.numpy().transpose(1, 2, 0)
        return flipped_img

class HorizontalFlipMultiChannels(nn.Module):
    """Performs a horizontal flip on a multichannel image."""
    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be horizontally flipped.

        Returns:
            Vertically flipped image (numpy.ndarray).
        """
        tensor_img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        flipped_tensor = F.hflip(tensor_img)
        flipped_img = flipped_tensor.numpy().transpose(1, 2, 0)
        return flipped_img

class RotationMultiChannels(nn.Module):
    """Performs a random rotation on a multichannel image."""
    def __init__(self, degrees, resample=False, expand=False):
        """
        Args:
            degrees (sequence or number): Range of degrees to select from.
                If degrees is a number, the range of degrees to select from
                will be (-degrees, +degrees).
            resample (bool, optional): If False, the default, the input image
                will be rotated using pixel area resampling. If True, the
                input image will be rotated using bilinear interpolation.
            expand (bool, optional): If True, the output image size will be
                expanded to include the whole rotated image. If False, the
                output image size will be the same as the input image.
                Expanding the size can help avoid cropping out parts of the
                image during rotation.
        """
        self.degrees = degrees
        self.resample = resample
        self.expand = expand

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be randomly rotated.

        Returns:
            Randomly rotated image (numpy.ndarray).
        """
        tensor_img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        if self.resample:
            rotated_tensor = F.rotate(img=tensor_img, angle=self.degrees, interpolation=InterpolationMode.BILINEAR, expand=self.expand)
        else:
            rotated_tensor = F.rotate(img=tensor_img, angle=self.degrees, expand=self.expand)

        rotated_img = rotated_tensor.numpy().transpose(1, 2, 0)
        return rotated_img

class CropMultiChannels(nn.Module):
    """Crop the given numpy array at a random location."""

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for random crop.

        Args:
            img (torch.Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to crop.
        """
        h, w = img.shape[-2:]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be cropped.

        Returns:
            numpy.ndarray: Cropped image.
        """
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float()

        if self.padding is not None:
            img_tensor = F.pad(img_tensor, self.padding, self.fill, self.padding_mode)

        # Pad the width if needed
        if self.pad_if_needed and img_tensor.shape[-1] < self.size[1]:
            img_tensor = F.pad(img_tensor, (0, self.size[1] - img_tensor.shape[-1]), self.fill, self.padding_mode)
        # Pad the height if needed
        if self.pad_if_needed and img_tensor.shape[-2] < self.size[0]:
            img_tensor = F.pad(img_tensor, (0, self.size[0] - img_tensor.shape[-2]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img_tensor, self.size)

        cropped_tensor = F.crop(img_tensor, i, j, h, w)
        cropped_img = cropped_tensor.numpy().transpose(1, 2, 0)
        return cropped_img

class GaussianBlurMultiChannels(nn.Module):
    """Apply random Gaussian blur to each channel of a multichannel image."""

    def __init__(self, radius_min=0.1, radius_max=2.0):
        """
        Args:
            radius_min (float): Minimum blur radius. Default is 0.1.
            radius_max (float): Maximum blur radius. Default is 2.0.
        """
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Input image with shape (height, width, channels).

        Returns:
            numpy.ndarray: Image with Gaussian blur applied to each channel.
        """
        img_tensor = torch.from_numpy(img)  # Convert numpy array to torch tensor
        img_tensor = img_tensor.permute(2, 0, 1)  # Change channel dimension to the first dimension

        n_channels = img_tensor.shape[0]

        for ind in range(n_channels):
            radius = torch.FloatTensor(1).uniform_(self.radius_min, self.radius_max)
            img_tensor[ind] = self.apply_gaussian_blur(img_tensor[ind], radius)

        img_tensor = img_tensor.permute(1, 2, 0)  # Change channel dimension back to the last dimension
        return img_tensor.numpy()  # Convert back to numpy array

    @staticmethod
    def apply_gaussian_blur(channel, radius):
        """
        Apply Gaussian blur to a single channel.

        Args:
            channel (torch.Tensor): Single channel image tensor.
            radius (float): Blur radius.

        Returns:
            torch.Tensor: Blurred channel tensor.
        """
        # Convert channel to 2D tensor (height, width)
        channel_2d = channel.unsqueeze(0)

        # Convert radius to kernel size
        kernel_size = int(2 * radius + 1)

        # Create Gaussian kernel
        kernel = GaussianBlurMultiChannels.create_gaussian_kernel(kernel_size, radius)

        # Apply Gaussian blur using torch.nn.conv2d with padding='same'
        blurred_channel_2d = nn.Conv2d(
            channel_2d.unsqueeze(0),
            kernel.unsqueeze(0),
            kernel_size=kernel_size,
            padding=radius,
            stride=1,
            groups=1,
        )

        return blurred_channel_2d.squeeze(0).squeeze(0)

    @staticmethod
    def create_gaussian_kernel(kernel_size, sigma):
        """
        Create a Gaussian kernel of the specified size and sigma.

        Args:
            kernel_size (int): Size of the kernel (both height and width).
            sigma (float): Standard deviation of the Gaussian distribution.

        Returns:
            torch.Tensor: Gaussian kernel tensor.
        """
        # Create a 1D Gaussian kernel
        kernel_1d = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-0.5 * (kernel_1d / sigma) ** 2)
        kernel_1d = kernel_1d / torch.sum(kernel_1d)

        # Expand the 1D kernel to 2D
        kernel = torch.outer(kernel_1d, kernel_1d)

        return kernel

class GaussianNoiseMultiChannels(nn.Module):
    """
    Adds Gaussian noise to a numpy array image.

    Arguments
    ---------
    mean (float): Mean of the Gaussian distribution. Default is 0.0.
    std (float): Standard deviation of the Gaussian distribution. Default is 0.05.

    Returns
    -------
    numpy.ndarray: Image with Gaussian noise added.
    """

    def __init__(self, mean=0.0, std=0.05):
        self.std = std
        self.mean = mean

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Input image with shape (height, width, channels).

        Returns:
            numpy.ndarray: Image with Gaussian noise added.
        """
        noise = np.random.normal(self.mean, self.std, size=img.shape)
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class CustomColorJitter(A.ImageOnlyTransform):
    def __init__(self, int_min_shift=-0.3, int_max_shift=0.3, gamma_min=0.5, gamma_max=1.5, always_apply=False, p=0.5):
        super(CustomColorJitter, self).__init__(always_apply=always_apply, p=p)
        self.int_min_shift = int_min_shift
        self.int_max_shift = int_max_shift
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def __call__(self, image, **kwargs):
        transformed_image = self.apply(img=image)
        return {'image': transformed_image}

    def apply(self, img, **params):
        # Get the number of channels in the image
        num_channels = img.shape[-1]

        # Generate random intensity shifts for each channel
        int_shifts = np.random.uniform(self.int_min_shift, self.int_max_shift, num_channels)

        # Generate random gamma values for each channel
        gammas = np.random.uniform(self.gamma_min, self.gamma_max, num_channels)

        # Apply intensity shifts and gamma changes to each channel
        adjusted_image = img.copy()

        # Put image as torch tensor
        adjusted_image = torch.from_numpy(adjusted_image.transpose((2, 0, 1))).float()

        for i in range(num_channels):
            channel = adjusted_image[i, ...]

            # Apply intensity shift
            channel += int_shifts[i]

            # Clip pixel values to ensure the intensity values are within [0, 1] range
            # channel = np.clip(channel, 0, 1)

            # Apply brightness change
            ratio = float(gammas[i])
            img1 = channel
            img2 = torch.zeros_like(channel)

            bound = 1.0 if img1.is_floating_point() else 255.0
            channel = (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)

            adjusted_image[i, ...] = channel

        # Convert back to numpy array
        adjusted_image = adjusted_image.numpy().transpose(1, 2, 0)

        return adjusted_image

    def update_params(self, params, **kwargs):
        pass  # No parameter updates needed

    def get_params(self):
        return {
            'int_min_shift': self.int_min_shift,
            'int_max_shift': self.int_max_shift,
            'gamma_min': self.gamma_min,
            'gamma_max': self.gamma_max,
            'always_apply': self.always_apply,
        }

    def get_transform_init_args_names(self):
        return ('int_min_shift', 'int_max_shift', 'gamma_min', 'gamma_max', 'always_apply', 'p')

def rand_apply(tranform, p=0.5):
    """
    Apply a transform with probability p

    Arguments
    ---------
    tranform (torchvision.transforms): transform to apply

    Returns
    -------
    transform (torchvision.transforms): transform with probability p
    """
    return RandomApply(torch.nn.ModuleList([tranform]), p)
