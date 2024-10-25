import itertools
import numpy as np
import torch
import unittest
from coltran.utils import base_utils


class UtilsTest(unittest.TestCase):

    def test_quantize(self):
        x = torch.arange(0, 256, dtype=torch.int32)
        actual = base_utils.convert_bits(x, n_bits_in=8, n_bits_out=5).numpy()
        expected = np.repeat(np.arange(0, 32), 8)
        self.assertTrue(np.allclose(expected, actual))

    def test_dequantize(self):
        x = torch.arange(0, 32, dtype=torch.int32)
        actual = base_utils.convert_bits(x, n_bits_in=5, n_bits_out=8).numpy()
        expected = np.arange(0, 256, 8)
        self.assertTrue(np.allclose(expected, actual))

    def test_rgb_to_ycbcr(self):
        x = torch.randint(0, 256, (2, 32, 32, 3), dtype=torch.float32)
        ycbcr = base_utils.rgb_to_ycbcr(x)
        self.assertEqual(ycbcr.shape, (2, 32, 32, 3))

    def test_image_hist_to_bit(self):
        x = torch.randint(0, 256, (2, 32, 32, 3), dtype=torch.int32)
        hist = base_utils.image_to_hist(x, num_symbols=256)
        self.assertEqual(hist.shape, (2, 3, 256))

    def test_labels_to_bins(self):
        n_bits = 3
        bins = np.arange(2**n_bits)
        triplets = itertools.product(bins, bins, bins)

        labels = np.array(list(triplets))
        labels_t = torch.tensor(labels, dtype=torch.float32)
        bins_t = base_utils.labels_to_bins(labels_t, num_symbols_per_channel=8)
        bins_np = bins_t.numpy()
        self.assertTrue(np.allclose(bins_np, np.arange(512)))

        inv_labels_t = base_utils.bins_to_labels(bins_t, num_symbols_per_channel=8)
        inv_labels_np = inv_labels_t.numpy()
        self.assertTrue(np.allclose(labels, inv_labels_np))

    def test_bins_to_labels_random(self):
        labels_t = torch.randint(0, 8, (1, 64, 64, 3), dtype=torch.int32)
        labels_np = labels_t.numpy()
        bins_t = base_utils.labels_to_bins(labels_t, num_symbols_per_channel=8)

        inv_labels_t = base_utils.bins_to_labels(bins_t, num_symbols_per_channel=8)
        inv_labels_np = inv_labels_t.numpy()
        self.assertTrue(np.allclose(inv_labels_np, labels_np))


if __name__ == '__main__':
    unittest.main()
