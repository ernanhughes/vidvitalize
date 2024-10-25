import itertools
import numpy as np
import torch
import torch.nn.functional as F
import unittest
from absl import logging
from absl.testing import parameterized
from coltran.models import core

def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Define hyperparameter combinations for testing
cond_hparams = itertools.product(["shift", "affine"], [True, False], [True, False], [True, False])
new_hparams = [(f"{cond_mlp}_{cond_ln}_{cond_att_q}_{cond_att_scale}", cond_mlp, cond_ln, cond_att_q, cond_att_scale) for cond_mlp, cond_ln, cond_att_q, cond_att_scale in cond_hparams]

class ColTranComponentsTest(unittest.TestCase, parameterized.TestCase):

    def get_config(self):
        config = {
            'hidden_size': 256,
            'ff_size': 256,
            'image_bit_depth': 5,
            'num_symbols': 32,
            'num_heads': 4,
            'resolution': [8, 8],
            'num_outer_layers': 1,
            'num_inner_layers': 3,
            'num_encoder_layers': 1,
            'batch_size': 2,
            'skip': True,
            'cond_mlp': 'affine_dense',
            'cond_mlp_act': 'identity',
            'cond_ln': True,
            'cond_ln_act': 'tanh',
            'cond_ln_seq': 'cs',
            'cond_ln_sp_ave': 'learnable',
            'cond_ln_init': 'glorot_uniform',
            'cond_att_act': 'identity',
            'cond_att_scale': True,
            'cond_att_k': True,
            'cond_att_q': True,
            'cond_att_v': True
        }
        return config

    def test_grayscale_encoder(self):
        config = self.get_config()
        inputs = torch.randint(0, 256, (2, 32, 32, 3), dtype=torch.int32)
        gray = torch.mean(inputs.float(), dim=-1, keepdim=True)
        encoder = core.GrayScaleEncoder(config)
        output = encoder(gray)
        self.assertEqual(output.shape, (2, 32, 32, 256))

    @parameterized.named_parameters(*new_hparams)
    def test_inner_decoder(self, cond_mlp, cond_ln, cond_att_q, cond_att_scale):
        embeddings = torch.rand((2, 8, 8, 256))
        channel_context = torch.rand((2, 8, 8, 256))
        upper_context = torch.rand((2, 8, 8, 256))
        config = self.get_config()
        config['cond_mlp'] = cond_mlp
        config['cond_ln'] = cond_ln
        config['cond_att_q'] = cond_att_q
        config['cond_att_scale'] = cond_att_scale

        model = core.InnerDecoder(config=config)
        output = model((embeddings, upper_context, channel_context))
        num_vars = get_num_parameters(model)
        logging.info(f"Number of parameters: {num_vars}")
        self.assertEqual(output.shape, (2, 8, 8, 256))

    @parameterized.named_parameters(*new_hparams)
    def test_outer_decoder(self, cond_mlp, cond_ln, cond_att_q, cond_att_scale):
        embeddings = torch.rand((2, 8, 8, 256))
        channel_context = torch.rand((2, 8, 8, 256))
        config = self.get_config()
        config['cond_mlp'] = cond_mlp
        config['cond_ln'] = cond_ln
        config['cond_att_q'] = cond_att_q
        config['cond_att_scale'] = cond_att_scale

        model = core.OuterDecoder(config=config)
        num_vars = get_num_parameters(model)
        logging.info(f"Number of parameters: {num_vars}")
        upper_context = model((embeddings, channel_context))

        # The first row slice should have zero context since both the present and future are effectively masked
        self.assertTrue(torch.allclose(upper_context[:, 0], torch.zeros_like(upper_context[:, 0])))
        self.assertEqual(upper_context.shape, (2, 8, 8, 256))

if __name__ == "__main__":
    unittest.main()
