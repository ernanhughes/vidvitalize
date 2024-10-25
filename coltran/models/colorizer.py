import torch
import torch.nn as nn
import torch.nn.functional as F
from coltran.models import core
from coltran.models import layers as coltran_layers
from coltran.utils import base_utils


class ColTranCore(nn.Module):
    """Colorization Transformer in PyTorch."""

    def __init__(self, config):
        super(ColTranCore, self).__init__()
        self.config = config
        self.num_symbols_per_channel = 2 ** 3
        self.num_symbols = self.num_symbols_per_channel ** 3
        self.gray_symbols, self.num_channels = 256, 1

        self.enc_cfg = config.encoder
        self.dec_cfg = config.decoder
        self.hidden_size = self.config.get('hidden_size', self.dec_cfg.hidden_size)
        
        self.stage = config.get('stage', 'decoder')
        self.is_parallel_loss = 'encoder' in self.stage

        if self.stage not in ['decoder', 'encoder_decoder']:
            raise ValueError(f"Expected stage to be 'decoder' or 'encoder_decoder', got {self.stage}")

        # Define encoder and decoders
        self.encoder = core.GrayScaleEncoder(self.enc_cfg)
        if self.is_parallel_loss:
            self.parallel_dense = nn.Linear(self.hidden_size, self.num_symbols, bias=False)

        # Decoder components
        self.pixel_embed_layer = nn.Linear(self.num_symbols, self.hidden_size, bias=False)
        self.outer_decoder = core.OuterDecoder(self.dec_cfg)
        self.inner_decoder = core.InnerDecoder(self.dec_cfg)
        self.final_dense = nn.Linear(self.hidden_size, self.num_symbols)
        self.final_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, inputs):
        gray = torch.mean(inputs, dim=-1, keepdim=True)  # RGB to grayscale
        z = self.encoder(gray)

        if self.is_parallel_loss:
            enc_logits = self.parallel_dense(z)
            enc_logits = enc_logits.unsqueeze(-2)

        dec_logits = self.decoder(inputs, z)
        if self.is_parallel_loss:
            return dec_logits, {'encoder_logits': enc_logits}
        return dec_logits, {}

    def decoder(self, inputs, z):
        labels = base_utils.convert_bits(inputs, n_bits_in=8, n_bits_out=3)
        labels = base_utils.labels_to_bins(labels, self.num_symbols_per_channel)
        labels = F.one_hot(labels, num_classes=self.num_symbols).float()
        
        h_dec = self.pixel_embed_layer(labels)
        h_upper = self.outer_decoder((h_dec, z))
        h_inner = self.inner_decoder((h_dec, h_upper, z))
        
        activations = self.final_norm(h_inner)
        logits = self.final_dense(activations)
        return logits.unsqueeze(-2)

    def image_loss(self, logits, labels):
        height, width = labels.shape[1:3]
        logits = logits.squeeze(-2)
        loss = F.cross_entropy(logits, labels, reduction='none')
        loss = base_utils.nats_to_bits(loss.sum(dim=-1).mean())
        return loss / (height * width)

    def sample(self, gray_cond, mode='argmax'):
        output = {}
        z_gray = self.encoder(gray_cond)
        if self.is_parallel_loss:
            z_logits = self.parallel_dense(z_gray)
            parallel_image = torch.argmax(z_logits, dim=-1)
            parallel_image = self.post_process_image(parallel_image)
            output['parallel'] = parallel_image

        image, proba = self.autoregressive_sample(z_gray=z_gray, mode=mode)
        output[f'auto_{mode}'] = image
        output['proba'] = proba
        return output

    def autoregressive_sample(self, z_gray, mode='sample'):
        batch_size, height, width, num_filters = z_gray.shape
        channel_cache = coltran_layers.Cache(canvas_shape=(height, width))
        init_channel = torch.zeros(batch_size, height, width, num_filters)
        channel_cache((init_channel, (0, 0)))

        upper_context = torch.zeros(batch_size, height, width, num_filters)
        row_cache = coltran_layers.Cache(canvas_shape=(1, width))
        init_row = torch.zeros(batch_size, 1, width, num_filters)
        row_cache((init_row, (0, 0)))

        pixel_samples, pixel_probas = [], []
        
        for row in range(height):
            row_cond_channel = z_gray[:, row].unsqueeze(1)
            row_cond_upper = upper_context[:, row].unsqueeze(1)
            row_cache.reset()

            gen_row, proba_row = [], []
            for col in range(width):
                inner_input = (row_cache.cache, row_cond_upper, row_cond_channel)
                activations = self.inner_decoder(inner_input, row_ind=row)

                pixel_sample, pixel_embed, pixel_proba = self.act_logit_sample_embed(activations, col, mode=mode)
                proba_row.append(pixel_proba)
                gen_row.append(pixel_sample)

                row_cache((pixel_embed, (0, col)))
                channel_cache((pixel_embed, (row, col)))

            gen_row = torch.stack(gen_row, dim=-1)
            pixel_samples.append(gen_row)
            pixel_probas.append(torch.stack(proba_row, dim=1))

            upper_context = self.outer_decoder((channel_cache.cache, z_gray))

        image = torch.stack(pixel_samples, dim=1)
        image = self.post_process_image(image)

        image_proba = torch.stack(pixel_probas, dim=1)
        return image, image_proba

    def act_logit_sample_embed(self, activations, col_ind, mode='sample'):
        batch_size = activations.shape[0]
        pixel_activation = activations[:, :, col_ind].unsqueeze(-2)
        pixel_logits = self.final_dense(self.final_norm(pixel_activation))
        pixel_logits = pixel_logits.squeeze(1).squeeze(1)
        pixel_proba = F.softmax(pixel_logits, dim=-1)

        if mode == 'sample':
            pixel_sample = torch.multinomial(pixel_proba, 1).squeeze(-1)
        elif mode == 'argmax':
            pixel_sample = torch.argmax(pixel_logits, dim=-1)

        pixel_sample_expand = pixel_sample.view(batch_size, 1, 1)
        pixel_one_hot = F.one_hot(pixel_sample_expand, num_classes=self.num_symbols).float()
        pixel_embed = self.pixel_embed_layer(pixel_one_hot)
        return pixel_sample, pixel_embed, pixel_proba

    def post_process_image(self, image):
        image = base_utils.bins_to_labels(image, num_symbols_per_channel=self.num_symbols_per_channel)
        image = base_utils.convert_bits(image, n_bits_in=3, n_bits_out=8)
        return image.type(torch.uint8)
