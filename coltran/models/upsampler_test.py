import unittest
import torch
from ml_collections import ConfigDict
from coltran.models import upsampler
from coltran.utils import base_utils


class ColorUpsamplerTest(unittest.TestCase):

    def get_config(self):
        config = ConfigDict()
        config.hidden_size = 128
        config.ff_size = 256
        config.num_heads = 2
        config.num_encoder_layers = 2
        config.num_symbols = 8
        return config

    def test_bit_upsampler_attention_num_channels_1(self):
        config = self.get_config()
        bit_upsampler = upsampler.ColorUpsampler(config=config)

        inputs = torch.randint(0, 256, (8, 32, 32, 3), dtype=torch.int32)
        inputs_slice = torch.randint(0, 256, (8, 32, 32, 1), dtype=torch.int32)
        grayscale = torch.mean(inputs.float(), dim=-1, keepdim=True).int()
        channel_index = torch.randint(0, 3, (8,), dtype=torch.int32)

        logits, _ = bit_upsampler(inputs=inputs, inputs_slice=inputs_slice, channel_index=channel_index)
        self.assertEqual(logits.shape, (8, 32, 32, 1, 256))

        inputs = base_utils.convert_bits(inputs, n_bits_in=8, n_bits_out=3)
        output = bit_upsampler.sample(gray_cond=grayscale, bit_cond=inputs)
        self.assertEqual(output['bit_up_argmax'].shape, (8, 32, 32, 3))

    def test_bit_upsampler_attention_num_channels_3(self):
        config = self.get_config()
        bit_upsampler = upsampler.ColorUpsampler(config=config)

        inputs = torch.randint(0, 256, (8, 32, 32, 3), dtype=torch.int32)
        grayscale = torch.mean(inputs.float(), dim=-1, keepdim=True).int()

        logits, _ = bit_upsampler(inputs=inputs, inputs_slice=inputs)
        self.assertEqual(logits.shape, (8, 32, 32, 3, 256))

        inputs = base_utils.convert_bits(inputs, n_bits_in=8, n_bits_out=3)
        output = bit_upsampler.sample(gray_cond=grayscale, bit_cond=inputs)
        self.assertEqual(output['bit_up_argmax'].shape, (8, 32, 32, 3))

    def test_color_upsampler_attention_num_channels_1(self):
        config = self.get_config()
        spatial_upsampler = upsampler.SpatialUpsampler(config=config)

        inputs = torch.randint(0, 256, (8, 64, 64, 3), dtype=torch.int32)
        inputs_slice = torch.randint(0, 256, (8, 64, 64, 1), dtype=torch.int32)
        grayscale = torch.mean(inputs.float(), dim=-1, keepdim=True).int()
        channel_index = torch.randint(0, 3, (8,), dtype=torch.int32)

        logits, _ = spatial_upsampler(inputs=inputs, inputs_slice=inputs_slice, channel_index=channel_index)
        self.assertEqual(logits.shape, (8, 64, 64, 1, 256))

        inputs = base_utils.convert_bits(inputs, n_bits_in=8, n_bits_out=3)
        output = spatial_upsampler.sample(gray_cond=grayscale, inputs=inputs)
        self.assertEqual(output['high_res_argmax'].shape, (8, 64, 64, 3))


if __name__ == '__main__':
    unittest.main()
