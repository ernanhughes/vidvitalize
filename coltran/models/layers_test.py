import itertools
import unittest
import torch
import numpy as np
from ml_collections import ConfigDict
from coltran.models import layers


layer_hparams = itertools.product(["mean", "learnable"],
                                  ["sc", "cs"])
layer_hparams = [(a+s, a, s) for a, s in layer_hparams]


class LayersTest(unittest.TestCase):

    def test_cache_layer(self):
        cache = layers.Cache(canvas_shape=(2, 4))

        # Update 1
        exp_first = torch.arange(8, dtype=torch.float).view(1, 2, 2, 2)
        index = torch.tensor([0, 0])
        out = cache((exp_first, index))
        out_slice = out[:1, :2, :2, :2]
        self.assertTrue(torch.allclose(out_slice, exp_first))

        # Update 2
        exp_second = torch.arange(8, 16, dtype=torch.float).view(1, 2, 2, 2)
        index = torch.tensor([0, 2])
        out = cache((exp_second, index))
        first, second = out[:1, :2, :2, :2], out[:1, :2, 2:, :2]
        self.assertTrue(torch.allclose(first, exp_first))
        self.assertTrue(torch.allclose(second, exp_second))

        # Update 3 (special case)
        exp_third = torch.tensor([50.0, 100.0]).view(1, 1, 1, 2)
        index = torch.tensor([0, 0])
        out = cache((exp_third, index))
        self.assertTrue(torch.allclose(out[0, 0, 0, :2], exp_third[0, 0, 0, :2]))

    def test_shift_layer(self):
        # Shift down
        down_shift = layers.Shift(dimension=0, resolution=[3, 3])
        input_np = np.arange(9).reshape((1, 3, 3))
        input_t = torch.tensor(input_np, dtype=torch.float)
        input_down = down_shift(input_t).numpy()
        equality = input_np[:, :-1] == input_down[:, 1:]
        self.assertTrue(np.all(equality))

        # Shift right
        right_shift = layers.Shift(dimension=1, resolution=[3, 3])
        input_t = torch.tensor(input_np, dtype=torch.float)
        input_right = right_shift(input_t).numpy()
        equality = input_np[:, :, :-1] == input_right[:, :, 1:]
        self.assertTrue(np.all(equality))

    def test_position_embed(self):
        pos_embed = layers.PositionEmbed(axes=[1, 2], max_lengths=[64, 32])
        inputs = torch.rand((8, 64, 32, 256))
        embedded = pos_embed(inputs)
        for variable in pos_embed.parameters():
            if len(variable.shape) == 3:
                self.assertEqual(variable.shape, (64, 1, 256))
            else:
                self.assertEqual(variable.shape, (32, 256))
        self.assertEqual(embedded.shape, (8, 64, 32, 256))

    @unittest.parameterized.expand(layer_hparams)
    def test_conditional_layer_norm(self, name, spatial_average, sequence):
        cond_layer_norm = layers.ConditionalLayerNorm(
            spatial_average=spatial_average, sequence=sequence)
        x = torch.rand((8, 32, 32, 128))
        cond_inputs = torch.rand((8, 32, 32, 128))
        out = cond_layer_norm((x, cond_inputs))
        self.assertEqual(out.shape, (8, 32, 32, 128))

    def test_self_attention_nd_cond_scale(self):
        row_mask = layers.SelfAttentionND(
            hidden_size=256, num_heads=4, nd_block_size=[1, 32],
            resolution=[32, 32], cond_q=True, cond_k=True, cond_v=True,
            cond_scale=True)
        inputs = torch.rand((1, 3, 32, 32, 3))
        cond_inputs = torch.rand((1, 3, 32, 32, 3))
        output = row_mask((inputs, cond_inputs))
        self.assertEqual(output.shape, (1, 3, 32, 32, 256))

    def test_self_attention_nd_cond_scale_false(self):
        row_mask = layers.SelfAttentionND(
            hidden_size=256, num_heads=4, nd_block_size=[1, 32],
            resolution=[32, 32], cond_q=True, cond_k=True, cond_v=True,
            cond_scale=False)
        inputs = torch.rand((1, 3, 32, 32, 3))
        cond_inputs = torch.rand((1, 3, 32, 32, 3))
        output = row_mask((inputs, cond_inputs))
        self.assertEqual(output.shape, (1, 3, 32, 32, 256))

    def test_row_attention(self):
        row = layers.SelfAttentionND(
            hidden_size=256, num_heads=4, nd_block_size=[1, 32],
            resolution=[32, 32])
        x = torch.rand((4, 2, 32, 3))
        output = row(x)
        self.assertEqual(row.attention_dim_q, -3)
        self.assertEqual(row.attention_dim_k, -3)
        self.assertEqual(output.shape, (4, 2, 32, 256))

    def test_column_attention(self):
        column = layers.SelfAttentionND(
            hidden_size=256, num_heads=4, nd_block_size=[32, 1],
            resolution=[32, 32])
        x = torch.rand((4, 32, 2, 3))
        output = column(x)
        self.assertEqual(output.shape, (4, 32, 2, 256))

    def test_row_attention_mask(self):
        row_mask = layers.SelfAttentionND(
            hidden_size=256, num_heads=4, nd_block_size=[1, 32],
            resolution=[32, 32], mask="future")
        x = torch.rand((4, 2, 32, 3))
        output = row_mask(x)
        self.assertEqual(row_mask.attention_dim_k, -3)
        self.assertEqual(row_mask.attention_dim_q, -3)
        self.assertEqual(output.shape, (4, 2, 32, 256))

    def test_col_attention_mask(self):
        col_mask = layers.SelfAttentionND(
            hidden_size=256, num_heads=8, nd_block_size=[4, 1],
            resolution=[4, 4], mask="future")
        x = torch.rand((4, 4, 2, 3))
        output = col_mask(x)
        self.assertEqual(output.shape, (4, 4, 2, 256))
        self.assertEqual(col_mask.attention_dim_k, -4)
        self.assertEqual(col_mask.attention_dim_q, -4)

    def test_factorized_attention(self):
        config = ConfigDict()
        config.hidden_size = 256
        config.ff_size = 256
        config.num_encoder_layers = 2
        config.num_heads = 2
        config.height = 8
        config.width = 8
        fact = layers.FactorizedAttention(config)
        inputs = torch.rand((8, 8, 8, 256))
        output = fact(inputs)
        self.assertEqual(output.shape, (8, 8, 8, 256))


if __name__ == "__main__":
    unittest.main()
