import torch
import torch.nn as nn
import torch.nn.functional as F
from coltran.models import layers as coltran_layers
from coltran.utils import base_utils


def cond_with_context(inputs, cond_layer, context, cond_type, cond_act):
    cond_act_func = base_utils.act_to_func(cond_act)
    cond_out = cond_layer(context)
    if cond_type == 'shift':
        inputs += cond_out
    elif cond_type == 'affine':
        shift, scale = torch.split(cond_out, cond_out.shape[-1] // 2, dim=-1)
        inputs *= cond_act_func(scale)
        inputs += cond_act_func(shift)
    return inputs


def get_pos_embeddings(pos_embed, inputs_shape):
    embeddings = torch.zeros(inputs_shape)
    return pos_embed(embeddings)


class GrayScaleEncoder(nn.Module):
    """Encodes grayscale version of the image into a 2-D spatial context."""

    def __init__(self, config):
        super(GrayScaleEncoder, self).__init__()
        self.config = config
        self.embedding = nn.Linear(256, self.config.hidden_size)
        self.encoder = coltran_layers.FactorizedAttention(self.config)

    def forward(self, inputs):
        grayscale = F.one_hot(inputs.squeeze(-1), num_classes=256).float()
        h_gray = self.embedding(grayscale)
        return self.encoder(h_gray)


class OuterDecoder(nn.Module):
    """Outer Decoder with optional conditioning."""

    def __init__(self, config):
        super(OuterDecoder, self).__init__()
        self.config = config
        self.skip = self.config.get('skip', True)
        self.cond_mlp = self.config.get('cond_mlp', 'affine')
        self.cond_mlp_act = self.config.get('cond_mlp_act', 'identity')
        self.cond_ln = self.config.get('cond_ln', True)
        self.cond_ln_act = self.config.get('cond_ln_act', 'identity')
        self.cond_ln_seq = self.config.get('cond_ln_seq', 'sc')
        self.cond_ln_sp_ave = self.config.get('cond_ln_sp_ave', 'learnable')
        self.cond_ln_init = self.config.get('cond_ln_init', 'glorot_uniform')
        self.cond_att_act = self.config.get('cond_att_act', 'identity')
        self.cond_att_k = self.config.get('cond_att_k', True)
        self.cond_att_q = self.config.get('cond_att_q', True)
        self.cond_att_v = self.config.get('cond_att_v', True)
        self.cond_att_scale = self.config.get('cond_att_scale', True)
        self.cond_att = self.cond_att_v or self.cond_att_q or self.cond_att_k

        self.pos_embed = coltran_layers.PositionEmbed(axes=[1, 2], max_lengths=[config.height, config.width])

        self.residual_layers, self.layer_norms, self.cmlp_layers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        num_norms = self.config.num_outer_layers * 4

        for _ in range(num_norms):
            if self.cond_ln:
                self.layer_norms.append(coltran_layers.ConditionalLayerNorm(
                    spatial_average=self.cond_ln_sp_ave,
                    sequence=self.cond_ln_seq,
                    out_act=self.cond_ln_act))
            else:
                self.layer_norms.append(nn.LayerNorm(self.config.hidden_size))

        for layer_ind in range(self.config.num_outer_layers):
            self._build_layers(layer_ind)

        self.shift_down = coltran_layers.Shift(dimension=0, resolution=[config.height, config.width])

    def _build_layers(self, layer_ind):
        hidden_size, num_heads = self.config.hidden_size, self.config.num_heads
        ff_size = self.config.ff_size

        unmask_row = coltran_layers.SelfAttentionND(
            hidden_size=hidden_size, num_heads=num_heads,
            nd_block_size=[1, self.config.width], resolution=[self.config.height, self.config.width],
            cond_q=self.cond_att_q, cond_k=self.cond_att_k, cond_v=self.cond_att_v,
            cond_scale=self.cond_att_scale, cond_act=self.cond_att_act)
        ff_row = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.ReLU(),
            nn.Linear(ff_size, hidden_size)
        )

        mask_col = coltran_layers.SelfAttentionND(
            hidden_size=hidden_size, num_heads=num_heads, mask='future',
            nd_block_size=[self.config.height, 1], resolution=[self.config.height, self.config.width],
            cond_q=self.cond_att_q, cond_k=self.cond_att_k, cond_v=self.cond_att_v,
            cond_scale=self.cond_att_scale, cond_act=self.cond_att_act)
        ff_col = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.ReLU(),
            nn.Linear(ff_size, hidden_size)
        )

        self.residual_layers.extend([unmask_row, ff_row, mask_col, ff_col])

        if self.cond_mlp == 'shift':
            self.cmlp_layers.extend([nn.Linear(hidden_size, hidden_size), nn.Linear(hidden_size, hidden_size)])
        elif self.cond_mlp == 'affine':
            self.cmlp_layers.extend([nn.Linear(hidden_size, 2 * hidden_size), nn.Linear(hidden_size, 2 * hidden_size)])

    def forward(self, inputs, training=True):
        embeddings, channel_context = inputs
        cond_layer_ind = 0

        output = self.pos_embed(embeddings)
        if self.skip:
            output += channel_context
        inputs = output

        for layer, norm in zip(self.residual_layers, self.layer_norms):
            if hasattr(layer, 'attention') and self.cond_att:
                output = layer((inputs, channel_context))
            else:
                output = layer(inputs)

            if 'dense' in layer.__class__.__name__.lower() and self.cond_mlp:
                curr_cond_layer = self.cmlp_layers[cond_layer_ind]
                output = cond_with_context(output, curr_cond_layer, channel_context, self.cond_mlp, self.cond_mlp_act)
                cond_layer_ind += 1

            output = coltran_layers.residual_dropout(inputs, output, self.dropout, training)

            if self.cond_ln:
                inputs = norm((output, channel_context))
            else:
                inputs = norm(output)

        output = self.shift_down(inputs)
        return output


class InnerDecoder(nn.Module):
    """Inner Decoder with optional conditioning."""

    def __init__(self, config):
        super(InnerDecoder, self).__init__()
        self.config = config
        self.skip = self.config.get('skip', True)
        self.dropout = self.config.get('dropout', 0.0)
        self.cond_mlp = self.config.get('cond_mlp', 'affine')
        self.cond_mlp_act = self.config.get('cond_mlp_act', 'identity')
        self.cond_ln = self.config.get('cond_ln', True)
        self.cond_ln_act = self.config.get('cond_ln_act', 'identity')
        self.cond_ln_seq = self.config.get('cond_ln_seq', 'sc')
        self.cond_ln_sp_ave = self.config.get('cond_ln_sp_ave', 'learnable')
        self.cond_ln_init = self.config.get('cond_ln_init', 'glorot_uniform')
        self.cond_att_act = self.config.get('cond_att_act', 'identity')
        self.cond_att_k = self.config.get('cond_att_k', False)
        self.cond_att_q = self.config.get('cond_att_q', False)
        self.cond_att_v = self.config.get('cond_att_v', False)
        self.cond_att_scale = self.config.get('cond_att_scale', False)
        self.cond_att = self.cond_att_v or self.cond_att_q or self.cond_att_k

        self.pos_embed = coltran_layers.PositionEmbed(axes=[1, 2], max_lengths=[config.height, config.width])
        self.shift_right = coltran_layers.Shift(dimension=1, resolution=[config.height, config.width])

        self.residual_layers, self.layer_norms, self.cmlp_layers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        num_norms = 2 * self.config.num_inner_layers

        for _ in range(num_norms):
            if self.cond_ln:
                self.layer_norms.append(coltran_layers.ConditionalLayerNorm(
                    spatial_average=self.cond_ln_sp_ave,
                    sequence=self.cond_ln_seq,
                    out_act=self.cond_ln_act))
            else:
                self.layer_norms.append(nn.LayerNorm(self.config.hidden_size))

        for layer_ind in range(self.config.num_inner_layers):
            mask_row = coltran_layers.SelfAttentionND(
                hidden_size=self.config.hidden_size, num_heads=self.config.num_heads, mask='future',
                nd_block_size=[1, self.config.width], resolution=[self.config.height, self.config.width],
                cond_q=self.cond_att_q, cond_k=self.cond_att_k, cond_v=self.cond_att_v,
                cond_scale=self.cond_att_scale, cond_act=self.cond_att_act)
            ff_block = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.ff_size),
                nn.ReLU(),
                nn.Linear(self.config.ff_size, self.config.hidden_size)
            )

            self.residual_layers.extend([mask_row, ff_block])

            if self.cond_mlp == 'shift':
                self.cmlp_layers.append(nn.Linear(self.config.hidden_size, self.config.hidden_size))
            elif self.cond_mlp == 'affine':
                self.cmlp_layers.append(nn.Linear(self.config.hidden_size, 2 * self.config.hidden_size))

    def forward(self, inputs, row_ind=None, training=True):
        embeddings, upper_context, channel_context = inputs
        embeddings = self.shift_right(embeddings)

        if row_ind is None:
            embeddings = self.pos_embed(embeddings)
        else:
            pos_embed = get_pos_embeddings(self.pos_embed, embeddings.shape)
            pos_embed = pos_embed[:, row_ind: row_ind + 1]
            embeddings += pos_embed

        inputs = embeddings
        if self.skip:
            inputs += channel_context
            inputs += upper_context

        all_context = torch.cat((channel_context, upper_context), dim=-1)
        cond_layer_ind = 0

        for layer, norm in zip(self.residual_layers, self.layer_norms):
            if hasattr(layer, 'attention') and self.cond_att:
                output = layer((inputs, all_context))
            else:
                output = layer(inputs)

            if 'dense' in layer.__class__.__name__.lower() and self.cond_mlp:
                curr_cond_layer = self.cmlp_layers[cond_layer_ind]
                output = cond_with_context(output, curr_cond_layer, all_context, self.cond_mlp, self.cond_mlp_act)
                cond_layer_ind += 1

            output = coltran_layers.residual_dropout(inputs, output, self.dropout, training)

            if self.cond_ln:
                inputs = norm((output, channel_context))
            else:
                inputs = norm(output)

        return inputs
