import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms


def nats_to_bits(nats):
    """Converts nats to bits."""
    return nats / np.log(2)


def act_to_func(act):
    """Maps an activation name to the corresponding function in PyTorch."""
    cond_act_map = {
        'relu': F.relu,
        'sigmoid': torch.sigmoid,
        'tanh': torch.tanh,
        'identity': lambda x: x
    }
    return cond_act_map.get(act)


def roll_channels_to_batch(tensor):
    """Switches from [B, H, W, C, D] to [B, C, H, W, D]."""
    return tensor.permute(0, 3, 1, 2, 4)


def roll_channels_from_batch(tensor):
    """Switches from [B, C, H, W, D] to [B, H, W, C, D]."""
    return tensor.permute(0, 2, 3, 1, 4)


def image_to_hist(image, num_symbols):
    """Returns a per-channel histogram of intensities.

    Args:
        image: 4-D Tensor, shape=(B, H, W, C)
        num_symbols: int
    Returns:
        hist: 3-D Tensor, shape=(B, C, num_symbols)
    """
    batch_size, height, width, channels = image.shape
    image = F.one_hot(image, num_classes=num_symbols).float()
    image = image.view(batch_size, height * width, channels, num_symbols)
    
    # Average spatially
    image = image.mean(dim=1)
    
    # Smooth
    eps = 1e-8
    image = (image + eps) / (1 + num_symbols * eps)
    return image


def get_bw_and_color(inputs, colorspace):
    """Returns grayscale and colored channels.

    Inputs are assumed to be in the RGB colorspace.

    Args:
        inputs: 4-D Tensor with 3 channels.
        colorspace: 'rgb' or 'ycbcr'
    Returns:
        grayscale: 4-D Tensor with 1 channel.
        inputs: 4-D Tensor with 3 channels.
    """
    if colorspace == 'rgb':
        grayscale = transforms.functional.rgb_to_grayscale(inputs, num_output_channels=1)
    elif colorspace == 'ycbcr':
        inputs = rgb_to_ycbcr(inputs)
        grayscale, inputs = inputs[..., :1], inputs[..., 1:]
    return grayscale, inputs


def rgb_to_ycbcr(rgb):
    """Convert RGB image to YCbCr color space."""
    rgb = rgb.float()
    r, g, b = rgb.unbind(dim=-1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128
    ycbcr = torch.stack((y, cb, cr), dim=-1)
    return ycbcr.clamp(0, 255).to(torch.int32)


def ycbcr_to_rgb(ycbcr):
    """Convert YCbCr image to RGB color space."""
    ycbcr = ycbcr.float()
    y, cb, cr = ycbcr.unbind(dim=-1)
    cb -= 128
    cr -= 128
    r = y + 1.402 * cr
    g = y - 0.34414 * cb - 0.71414 * cr
    b = y + 1.772 * cb
    rgb = torch.stack((r, g, b), dim=-1)
    return rgb.clamp(0, 255).to(torch.int32)


def convert_bits(x, n_bits_out=8, n_bits_in=8):
    """Quantize / dequantize from n_bits_in to n_bits_out."""
    if n_bits_in == n_bits_out:
        return x
    x = x.float()
    x = (x / (2 ** (n_bits_in - n_bits_out))).int()
    return x


def get_patch(upscaled, window, normalize=True):
    """Extract patch from upscaled with window size."""
    upscaled = upscaled.float()

    # Pool + quantize + normalize
    patch = F.avg_pool2d(upscaled, kernel_size=window, stride=window)
    if normalize:
        patch /= 256.0
    else:
        patch = patch.int()
    return patch


def labels_to_bins(labels, num_symbols_per_channel):
    """Maps each (R, G, B) channel triplet to a unique bin.

    Args:
        labels: 4-D Tensor, shape=(batch_size, H, W, 3).
        num_symbols_per_channel: int, number of symbols per channel.

    Returns:
        3-D Tensor, shape=(batch_size, H, W) with combined symbol bins.
    """
    labels = labels.float()
    channel_hash = torch.tensor([num_symbols_per_channel ** 2, num_symbols_per_channel, 1.0], device=labels.device)
    labels = (labels * channel_hash).sum(dim=-1)
    return labels.int()


def bins_to_labels(bins, num_symbols_per_channel):
    """Maps back from bins to (R, G, B) channel triplet.

    Args:
        bins: 3-D Tensor, shape=(batch_size, H, W) with symbol bins.
        num_symbols_per_channel: int, number of symbols per channel.

    Returns:
        4-D Tensor, shape=(batch_size, H, W, 3).
    """
    labels = []
    factor = num_symbols_per_channel ** 2

    for _ in range(3):
        channel = bins // factor
        labels.append(channel)
        bins = bins % factor
        factor //= num_symbols_per_channel

    return torch.stack(labels, dim=-1)
