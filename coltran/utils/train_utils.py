import os
import time
import torch
import torch.optim as optim
import yaml
import numpy as np
from torch.utils.data import DataLoader
from absl import logging


def step_with_strategy(step_fn, device=None):
    """Runs the step function, optionally using the specified device."""
    def _step(iterator):
        data = next(iterator)
        if device:
            data = {k: v.to(device) for k, v in data.items()}  # Move data to device
        step_fn(data)
    return _step


def write_config(config, logdir):
    """Write configuration dictionary to a directory."""
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, 'config.yaml'), 'w') as f:
        yaml.dump(config.to_dict(), f)


def wait_for_checkpoint(observe_dirs, prev_path=None, max_wait=-1):
    """Returns new checkpoint paths, or None if timing out."""
    is_single = isinstance(observe_dirs, str)
    if is_single:
        observe_dirs = [observe_dirs]
        if prev_path:
            prev_path = [prev_path]

    start_time = time.time()
    prev_path = prev_path or [None for _ in observe_dirs]
    while True:
        new_path = [torch.load(os.path.join(d, 'latest.ckpt')) if os.path.exists(d) else None for d in observe_dirs]
        if all(a != b for a, b in zip(new_path, prev_path)):
            latest_ckpt = new_path[0] if is_single else new_path
            if latest_ckpt is not None:
                return latest_ckpt
        if max_wait > 0 and (time.time() - start_time) > max_wait:
            return None
        logging.info('Sleeping 60s, waiting for checkpoint.')
        time.sleep(60)


def build_optimizer(config, model_params):
    """Builds optimizer."""
    optim_config = dict(config['optimizer'])
    optim_type = optim_config.pop('type', 'rmsprop')
    if optim_type == 'rmsprop':
        optimizer = optim.RMSprop(model_params, **optim_config)
    elif optim_type == 'adam':
        optimizer = optim.Adam(model_params, **optim_config)
    elif optim_type == 'sgd':
        optimizer = optim.SGD(model_params, **optim_config)
    else:
        raise ValueError(f'Unknown optimizer type: {optim_type}')
    return optimizer


def build_ema(model, decay):
    """Creates an EMA (Exponential Moving Average) model copy."""
    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)  # Turn off gradients for EMA model

    def update_ema(model, ema_model, decay):
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    return ema_model, lambda: update_ema(model, ema_model, decay)


def setup_strategy(devices_per_worker, mode, accelerator_type):
    """Set up strategy and data parallelism on available GPUs."""
    if accelerator_type == 'GPU':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_cores = min(devices_per_worker, torch.cuda.device_count())
        batch_size = num_cores
        logging.info('Running on %d number of GPUs with a batch size of %d.', num_cores, batch_size)
        return device, batch_size
    else:
        raise NotImplementedError(f'Accelerator type {accelerator_type} is not supported.')


def dataset_with_strategy(dataset_fn, strategy=None):
    """Load dataset with optional distribution strategy."""
    dataset = dataset_fn()
    if strategy:
        # Wrap with DistributedSampler if using a multi-GPU setup
        sampler = torch.utils.data.distributed.DistributedSampler(dataset) if torch.cuda.device_count() > 1 else None
        return DataLoader(dataset, sampler=sampler)
    return DataLoader(dataset)


def with_strategy(fn, device=None):
    """Runs the function within a device context if specified."""
    if device:
        with torch.cuda.device(device):
            return fn()
    else:
        return fn()


def create_checkpoint(model, optimizer=None, ema=None, path='checkpoint.pth'):
    """Creates a checkpoint and saves model parameters and optimizer state."""
    checkpoint = {'model_state_dict': model.state_dict()}
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if ema:
        checkpoint['ema_state_dict'] = ema.state_dict()
    torch.save(checkpoint, path)
    return checkpoint


def get_curr_step(ckpt_path):
    """Retrieve the current training step from a checkpoint path."""
    checkpoint = torch.load(ckpt_path)
    return checkpoint.get('step', None)


def restore(model, ckpt_path, optimizer=None, ema=None):
    """Restores model and optimizer state from a checkpoint."""
    logging.info(f'Restoring from checkpoint at {ckpt_path}')
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if ema and 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'])


def save_nparray_to_disk(filename, nparray):
    """Save a numpy array to disk."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.save(filename, nparray)
