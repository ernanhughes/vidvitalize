import torch
import random
from torchvision import transforms
from PIL import Image


def change_resolution(image, res, method='area'):
    """Resizes the image to a specified resolution."""
    if method == 'area':
        interpolation = Image.Resampling.BILINEAR  # Equivalent to 'area' in TensorFlow
    elif method == 'bilinear':
        interpolation = Image.Resampling.BILINEAR
    elif method == 'nearest':
        interpolation = Image.Resampling.NEAREST
    elif method == 'bicubic':
        interpolation = Image.Resampling.BICUBIC
    else:
        raise ValueError(f"Unsupported method: {method}")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((res, res), interpolation=interpolation),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.round(x * 255).int())
    ])
    
    return transform(image)


def downsample_and_upsample(x, train, downsample_res, upsample_res, method):
    """Downsample and then upsample images in the input dictionary."""
    keys = ['targets']
    if train and 'targets_slice' in x:
        keys.append('targets_slice')

    for key in keys:
        inputs = x[key]
        
        # Conditional low-resolution input
        x_down = change_resolution(inputs, res=downsample_res, method=method)
        x[f"{key}_{downsample_res}"] = x_down

        # Upsample to the original or specified resolution
        x_up = change_resolution(x_down, res=upsample_res, method=method)
        x[f"{key}_{downsample_res}_up_back"] = x_up

    return x


def random_channel_slice(x):
    """Randomly selects a single channel from the target image."""
    random_channel = random.randint(0, 2)  # Randomly select a channel (0, 1, or 2 for RGB)
    targets = x['targets']
    res = targets.shape[1]

    # Select the random channel and reshape for PyTorch compatibility
    image_slice = targets[:, :, random_channel].unsqueeze(-1)  # Keep the last dimension
    x['targets_slice'] = image_slice
    x['channel_index'] = random_channel

    return x
