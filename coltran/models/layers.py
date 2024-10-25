import functools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from coltran.utils import att_utils
from coltran.utils import base_utils


def residual_dropout(inputs, output, dropout, training):
    """out = inputs + dropout(output)"""
    if training and dropout:
        output = F.dropout(output, p=dropout)
    output += inputs
    return output


class Shift(nn.Module):
    """Shifts an input tensor either down or right to preserve causal ordering."""

    def __init__(self, dimension, resolution):
        super(Shift, self).__init__()
        self.dimension = dimension
        self.resolution = resolution

    def forward(self, x):
        paddings = [(0, 0)] * len(x.shape)
        paddings[self.dimension + 1] = (1, 0)
        y = F.pad(x, paddings, mode='constant', value=0)
        return y[..., :x.shape[self.dimension + 1]]


class Cache(nn.Module):
    """PyTorch layer for caching."""

    def __init__(self, canvas_shape, num_batch_axes=1, dtype=torch.float32):
        super(Cache, self).__init__()
        self.canvas_shape = canvas_shape
        self.num_batch_axes = num_batch_axes
        self._dtype = dtype

    def reset(self):
        self.cache = torch.zeros(self.cache.shape, dtype=self._dtype)

    def forward(self, value, index):
        if self.cache.shape == value.shape:
            self.cache = value
            return value

        # Set up batch and index ranges
        batch_shape = value.shape[:self.num_batch_axes]
        index_shape = [index.shape[0]]
        indices = torch.meshgrid(*[torch.arange(s) for s in batch_shape + index_shape], indexing="ij")
        indices = torch.stack(indices, dim=-1).reshape(-1, self.num_batch_axes + len(index_shape))

        # Update cache with tensor_scatter_nd_update-like operation
        self.cache.index_add_(0, indices, value.view(-1, *self.cache.shape[2:]))
        return self.cache


class PositionEmbed(nn.Module):
    """Adds factorized positional embeddings for specified axes."""

    def __init__(self, axes, max_lengths=None):
        super(PositionEmbed, self).__init__()
        self.axes = axes if isinstance(axes, (list, tuple)) else [axes]
        self.max_lengths = max_lengths if max_lengths else None
        self.embeddings = nn.ParameterList()

    def forward(self, inputs):
        out = inputs
        for embed in self.embeddings:
            out += embed
        return out


class DenseND(nn.Module):
    """Maps a rank-m tensor to a rank-n tensor through a dense contraction."""

    def __init__(self, filters, contract_axes=1, use_bias=False, activation=None):
        super(DenseND, self).__init__()
        self.filters = filters if isinstance(filters, tuple) else (filters,)
        self.contract_axes = contract_axes
        self.use_bias = use_bias
        self.activation = activation

    def forward(self, inputs):
        result = torch.einsum('...bc,c->...b', inputs, self.filters)
        if self.use_bias:
            result += self.bias
        return F.relu(result) if self.activation else result


class RelativeAttentionBiasND(nn.Module):
    """Relative attention bias in nd factorizes over dimensions."""

    def __init__(self, lengths, num_heads):
        super(RelativeAttentionBiasND, self).__init__()
        self.num_heads = num_heads
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(num_heads, 2 * length), requires_grad=True) if length > 1 else None
            for length in lengths
        ])

    def forward(self, inputs=None):
        biases = []
        for i, bias in enumerate(self.biases):
            if bias is not None:
                expanded_bias = att_utils.relative_attn_bias(bias, self.num_heads)
                biases.append(expanded_bias)
        return sum(biases)


class SelfAttentionND(nn.Module):
    """Transforms input through an N-D self-attention layer."""

    def __init__(self, hidden_size, num_heads=1, num_channels_per_head=None, mask=None, nd_block_size=None):
        super(SelfAttentionND, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_channels_per_head = num_channels_per_head or hidden_size // num_heads
        self.mask = mask
        self.nd_block_size = nd_block_size

        self.project_q = DenseND([num_heads, self.num_channels_per_head])
        self.project_k = DenseND([num_heads, self.num_channels_per_head])
        self.project_v = DenseND([num_heads, self.num_channels_per_head])
        self.project_final = DenseND(hidden_size)

        if mask:
            total_length = functools.reduce(operator.mul, nd_block_size or [1], 1)
            self._mask = torch.triu(torch.ones(total_length, total_length) * -1e6, diagonal=1)
        else:
            self._mask = None

    def forward(self, inputs):
        q, k, v = self.project_q(inputs), self.project_k(inputs), self.project_v(inputs)
        q *= self.num_channels_per_head ** -0.5

        # Self-attention score computation
        alphas = torch.einsum('bqhc,bkhc->bhqk', q, k)
        if self._mask is not None:
            alphas += self._mask

        weights = F.softmax(alphas, dim=-1)
        output = torch.einsum('bhqk,bkhc->bqhc', weights, v)
        return self.project_final(output)


class FactorizedAttention(nn.Module):
    """Encodes image into 2-D spatial context with factorized attention layers."""

    def __init__(self, config):
        super(FactorizedAttention, self).__init__()
        self.config = config
        self.dropout = config.get('dropout', 0.0)

        ff_size = config.ff_size
        hidden_size = config.hidden_size
        num_heads = config.num_heads
        height = config.height
        width = config.width

        # Positional embedding and attention blocks
        self.pos_embed = PositionEmbed(axes=[1, 2], max_lengths=[height, width])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4 * config.num_encoder_layers)])

        self.attention_blocks = nn.ModuleList([
            SelfAttentionND(hidden_size, num_heads, nd_block_size=[1, width]) if i % 2 == 0 else nn.Sequential(
                nn.Linear(hidden_size, ff_size),
                nn.ReLU(),
                nn.Linear(ff_size, hidden_size)
            )
            for i in range(4 * config.num_encoder_layers)
        ])

    def forward(self, inputs, training=True):
        inputs = self.pos_embed(inputs)

        for i, layer in enumerate(self.attention_blocks):
            normed = self.layer_norms[i](inputs)
            out = layer(normed)
            inputs = residual_dropout(inputs, out, self.dropout, training)
        return inputs
