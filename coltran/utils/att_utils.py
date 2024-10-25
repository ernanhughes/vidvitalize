import functools
import itertools
import operator
import numpy as np
import torch
import torch.nn.functional as F


def index_to_step(index, shape):
    """Compute step for a given n-dimensional index if we were enumerating to shape."""
    step = index[0]
    for i, s in enumerate(shape[1:]):
        step = step * s + index[i + 1]
    return step


def pad_to_multiple_nd(x, shape):
    """Pads tensor x such that specified axes are multiples of shape axes."""
    x_shape = x.shape
    num_feat_dim = len(x_shape) - len(shape) - 1
    if all(x_shape[1:len(shape) + 1]):
        pad_amount = [-dim % s for dim, s in zip(x_shape[1:len(shape) + 1], shape)]
        paddings = [(0, 0)] + [(0, p) for p in pad_amount] + [(0, 0)] * num_feat_dim
        return F.pad(x, [pad for sublist in paddings[::-1] for pad in sublist]) if any(p > 0 for _, p in paddings) else x
    else:
        tf_shape = x.shape[1:len(shape) + 1]
        paddings = [(0, -dim % s) for dim, s in zip(tf_shape, shape)]
        paddings = [(0, 0)] + paddings + [(0, 0)] * num_feat_dim
        return F.pad(x, [pad for sublist in paddings[::-1] for pad in sublist])


def divide_nd_blocks(inputs, nd_block_size, collapse=False):
    """Divides input into non-overlapping n-dimensional blocks."""
    inputs = pad_to_multiple_nd(inputs, nd_block_size)
    shape = list(inputs.shape)
    
    block_axes = shape[1:len(nd_block_size) + 1]
    num_blocks = [dim // s for dim, s in zip(block_axes, nd_block_size)]
    num_nd_axes = len(nd_block_size)
    num_feat_axes = len(shape) - num_nd_axes - 1
    features_shape = shape[-num_feat_axes:]

    # Reshape into [B, D1 // S1, S1, D2 // S2, S2, ..., Dk // Sk, Sk, ...]
    mid_shape = list(itertools.chain(*zip(num_blocks, nd_block_size)))
    cut_shape = shape[:1] + mid_shape + features_shape
    cut_inputs = inputs.view(*cut_shape)

    # Permute to [B, D1 // S1, D2 // S2, ..., Dk // Sk, S1, S2, ..., Sk, ...]
    mid_permute = list(itertools.chain(range(1, 2 * num_nd_axes, 2), range(2, 2 * num_nd_axes + 1, 2)))
    post_permute = list(range(2 * num_nd_axes + 1, 2 * num_nd_axes + 1 + num_feat_axes))
    permutation = [0] + mid_permute + post_permute
    permuted_inputs = cut_inputs.permute(*permutation)

    if not collapse:
        return permuted_inputs
    
    # Collapse to [B * D1 // S1 * D2 // S2 * ... * Dk // Sk, S1 * S2 * Sk, ...]
    block_length = functools.reduce(operator.mul, nd_block_size, 1)
    collapsed_inputs = permuted_inputs.view(-1, block_length, *features_shape)
    return collapsed_inputs


def relative_attn_bias(rel_bias, num_heads, decode_step=None):
    """Computes attention bias based on relative positions."""
    num_rel_pos = rel_bias.shape[-1]
    length = num_rel_pos // 2

    if isinstance(decode_step, int):
        # Decoding, so slice at the current decode step.
        start = (length - 1) - decode_step
        rel_bias = rel_bias.view(1, num_heads, num_rel_pos)
        return rel_bias[:, :, start:start + length]

    # Repeating and shifting the bias for relative positioning
    rel_bias = rel_bias.repeat(1, length)
    num_rel_pos -= 1
    rel_bias = rel_bias[:, :length * num_rel_pos]

    # Reshape to get relative shifts per head
    rel_bias = rel_bias.view(num_heads, length, num_rel_pos)

    # Slice overlapping elements to form the shifted pattern
    rel_bias = rel_bias[:, :, num_rel_pos - length:]
    rel_bias = rel_bias.permute(1, 0, 2)

    return rel_bias
