import torch
import torch.nn as nn
import torch.nn.functional as F
from coltran.models import layers as coltran_layers
from coltran.utils import base_utils


class ColorUpsampler(nn.Module):
    """Color Upsampler in PyTorch."""

    def __init__(self, config):
        super(ColorUpsampler, self).__init__()
        self.config = config
        self.hidden_size = self.config.get('hidden_size', 512)
        
        # Define layers
        self.bit_embedding = nn.Linear(24, self.hidden_size, bias=False)
        self.gray_embedding = nn.Linear(256, self.hidden_size, bias=False)
        self.input_dense = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.encoder = coltran_layers.FactorizedAttention(self.config)
        self.final_dense = nn.Linear(self.hidden_size, 256)

    def forward(self, inputs, inputs_slice, channel_index=None, training=True):
        """Upsamples the coarsely colored input into an RGB image."""
        grayscale = torch.mean(inputs, dim=-1, keepdim=True)
        inputs_slice = base_utils.convert_bits(inputs_slice, n_bits_in=8, n_bits_out=3)

        logits = self.upsampler(inputs_slice, grayscale, channel_index=channel_index, training=training)
        return logits, {}

    def upsampler(self, inputs, grayscale, channel_index=None, training=True):
        """Upsamples the coarse inputs to per-channel logits."""
        num_channels = inputs.shape[-1]
        logits = []

        # Embed grayscale image
        grayscale = F.one_hot(grayscale.long(), num_classes=256).float()
        gray_embed = self.gray_embedding(grayscale).squeeze(-2)

        if channel_index is not None:
            channel_index = channel_index.view(-1, 1, 1)

        for channel_ind in range(num_channels):
            channel = inputs[..., channel_ind]
            if channel_index is not None:
                channel += 8 * channel_index
            else:
                channel += 8 * channel_ind

            channel = F.one_hot(channel.long(), num_classes=24).float()
            channel = self.bit_embedding(channel).squeeze(-2)

            # Concatenate channel with grayscale embedding
            channel = torch.cat((channel, gray_embed), dim=-1)
            channel = self.input_dense(channel)

            # Encoder and final dense layer
            context = self.encoder(channel, training=training)
            channel_logits = self.final_dense(context)
            logits.append(channel_logits)
        
        logits = torch.stack(logits, dim=-2)
        return logits

    def sample(self, gray_cond, bit_cond, mode='argmax'):
        output = {}
        bit_cond_viz = base_utils.convert_bits(bit_cond, n_bits_in=3, n_bits_out=8)
        output['bit_cond'] = bit_cond_viz.byte()

        logits = self.upsampler(bit_cond, gray_cond, training=False)

        if mode == 'argmax':
            samples = torch.argmax(logits, dim=-1)
        elif mode == 'sample':
            batch_size, height, width, channels = logits.shape[:-1]
            logits = logits.view(-1, logits.size(-1))
            samples = torch.multinomial(F.softmax(logits, dim=-1), 1).view(batch_size, height, width, channels)

        output[f'bit_up_{mode}'] = samples.byte()
        return output

    def loss(self, targets, logits, train_config, training, aux_output=None):
        is_downsample = train_config.get('downsample', False)
        downsample_res = train_config.get('downsample_res', 64)
        if is_downsample and training:
            labels = targets[f'targets_slice_{downsample_res}']
        elif is_downsample:
            labels = targets[f'targets_{downsample_res}']
        elif training:
            labels = targets['targets_slice']
        else:
            labels = targets['targets']

        height, width, num_channels = labels.shape[1:]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = base_utils.nats_to_bits(loss)
        return loss / (height * width * num_channels), {}


class SpatialUpsampler(nn.Module):
    """Spatial Upsampler in PyTorch."""

    def __init__(self, config):
        super(SpatialUpsampler, self).__init__()
        self.config = config
        self.num_symbols = 256
        self.hidden_size = self.config.get('hidden_size', 512)
        self.down_res = self.config.get('down_res', 32)
        
        # Define layers
        self.channel_embedding = nn.Linear(self.num_symbols * 3, self.hidden_size, bias=False)
        self.gray_embedding = nn.Linear(self.num_symbols, self.hidden_size, bias=False)
        self.input_dense = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.encoder = coltran_layers.FactorizedAttention(self.config)
        self.final_dense = nn.Linear(self.hidden_size, self.num_symbols)

    def forward(self, inputs, inputs_slice, channel_index=None, training=True):
        """Super resolves blurry high resolution inputs into per-pixel logits."""
        grayscale = torch.mean(inputs, dim=-1, keepdim=True)
        logits = self.upsampler(inputs_slice, grayscale, channel_index=channel_index, training=training)
        return logits, {}

    def upsampler(self, inputs, grayscale, channel_index=None, training=True):
        num_channels = inputs.shape[-1]
        logits = []

        grayscale = F.one_hot(grayscale.long(), num_classes=self.num_symbols).float()
        gray_embed = self.gray_embedding(grayscale).squeeze(-2)

        if channel_index is not None:
            channel_index = channel_index.view(-1, 1, 1)

        for channel_ind in range(num_channels):
            channel = inputs[..., channel_ind]

            if channel_index is not None:
                channel += self.num_symbols * channel_index
            else:
                channel += self.num_symbols * channel_ind

            channel = F.one_hot(channel.long(), num_classes=self.num_symbols * 3).float()
            channel = self.channel_embedding(channel).squeeze(-2)

            channel = torch.cat((channel, gray_embed), dim=-1)
            channel = self.input_dense(channel)

            context = self.encoder(channel, training=training)
            channel_logits = self.final_dense(context)
            logits.append(channel_logits)
        
        logits = torch.stack(logits, dim=-2)
        return logits

    def sample(self, gray_cond, inputs, mode='argmax'):
        output = {}
        output['low_res_cond'] = inputs.byte()
        logits = self.upsampler(inputs, gray_cond, training=False)

        if mode == 'argmax':
            samples = torch.argmax(logits, dim=-1)
        elif mode == 'sample':
            batch_size, height, width, channels = logits.shape[:-1]
            logits = logits.view(-1, logits.size(-1))
            samples = torch.multinomial(F.softmax(logits, dim=-1), 1).view(batch_size, height, width, channels)

        output[f'high_res_{mode}'] = samples.byte()
        return output

    def loss(self, targets, logits, train_config, training, aux_output=None):
        if training:
            labels = targets['targets_slice']
        else:
            labels = targets['targets']

        height, width, num_channels = labels.shape[1:]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = base_utils.nats_to_bits(loss)
        return loss / (height * width * num_channels), {}
