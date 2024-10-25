import unittest
import torch
import torch.nn.functional as F
import numpy as np
from ml_collections import ConfigDict
from coltran.models import colorizer

class ColTranCoreTest(unittest.TestCase):
    def get_config(self, encoder_net='attention'):
        config = ConfigDict()
        config.image_bit_depth = 3
        config.encoder_1x1 = True
        config.resolution = [64, 64]
        config.batch_size = 2
        config.encoder_net = encoder_net
        config.hidden_size = 128
        config.stage = 'decoder'

        config.encoder = ConfigDict()
        config.encoder.dropout = 0.0
        config.encoder.ff_size = 128
        config.encoder.hidden_size = 128
        config.encoder.num_heads = 1
        config.encoder.num_encoder_layers = 1

        config.decoder = ConfigDict()
        config.decoder.ff_size = 128
        config.decoder.hidden_size = 128
        config.decoder.num_heads = 1
        config.decoder.num_outer_layers = 1
        config.decoder.num_inner_layers = 1
        config.decoder.resolution = [64, 64]
        config.decoder.dropout = 0.1
        config.decoder.cond_ln = True
        config.decoder.cond_q = True
        config.decoder.cond_k = True
        config.decoder.cond_v = True
        config.decoder.cond_q = True
        config.decoder.cond_scale = True
        config.decoder.cond_mlp = 'affine'
        return config

    def test_transformer_attention_encoder(self):
        config = self.get_config(encoder_net='attention')
        config.stage = 'encoder_decoder'
        transformer = colorizer.ColTranCore(config=config)
        images = torch.randint(0, 256, (2, 2, 2, 3), dtype=torch.int32)

        # Forward pass to get logits
        logits, _ = transformer(images)
        self.assertEqual(logits.shape, (2, 2, 2, 1, 512))

        # Grayscale conversion for sampling
        gray = torch.mean(images, dim=-1, keepdim=True)
        output = transformer.sample(gray, mode='argmax')
        output_np = output['auto_argmax'].numpy()
        proba_np = output['proba'].numpy()

        # Assertions for batch-size = 2 output shapes
        self.assertEqual(output_np.shape, (2, 2, 2, 3))
        self.assertEqual(proba_np.shape, (2, 2, 2, 512))

        # Checking output consistency with batch-size = 1
        output_np_bs_1, proba_np_bs_1 = [], []
        for batch_ind in [0, 1]:
            curr_gray = gray[batch_ind:batch_ind+1]
            curr_out = transformer.sample(curr_gray, mode='argmax')
            output_np_bs_1.append(curr_out['auto_argmax'].numpy())
            proba_np_bs_1.append(curr_out['proba'].numpy())

        output_np_bs_1 = np.concatenate(output_np_bs_1, axis=0)
        proba_np_bs_1 = np.concatenate(proba_np_bs_1, axis=0)
        self.assertTrue(np.allclose(output_np, output_np_bs_1))
        self.assertTrue(np.allclose(proba_np, proba_np_bs_1))

    def test_transformer_encoder_decoder(self):
        config = self.get_config()
        config.stage = 'encoder_decoder'

        transformer = colorizer.ColTranCore(config=config)
        images = torch.randint(0, 256, (1, 64, 64, 3), dtype=torch.int32)
        
        # Forward pass to get both encoder and decoder logits
        logits, aux_logits = transformer(images)
        enc_logits = aux_logits['encoder_logits']

        # Assertions for encoder and decoder logits shapes
        self.assertEqual(enc_logits.shape, (1, 64, 64, 1, 512))
        self.assertEqual(logits.shape, (1, 64, 64, 1, 512))


if __name__ == '__main__':
    unittest.main()
