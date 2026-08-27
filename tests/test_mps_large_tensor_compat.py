"""Regression coverage for MPS large-tensor VAE compatibility helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.video_vae_v3.modules import causal_inflation_lib, context_parallel_lib, mps_compat


class MPSLargeTensorCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tensor = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 2, 2)

    def test_repeat_first_frame_matches_torch_tile_on_cpu(self) -> None:
        expected = torch.tile(self.tensor[:, :, :1], (1, 1, 4, 1, 1))

        actual = mps_compat.repeat_first_frame(self.tensor, times=4)

        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(actual.is_contiguous())

    def test_safe_cat_matches_torch_cat_on_cpu(self) -> None:
        left = self.tensor[:, :, :1]
        right = self.tensor[:, :, 1:]
        expected = torch.cat((left, right), dim=2)

        actual = mps_compat.mps_safe_cat((left, right), dim=2)

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.device.type, "cpu")

    def test_vae_modules_bind_the_local_compatibility_helpers(self) -> None:
        self.assertIs(context_parallel_lib.repeat_first_frame, mps_compat.repeat_first_frame)
        self.assertIs(causal_inflation_lib.repeat_first_frame, mps_compat.repeat_first_frame)
        self.assertIs(causal_inflation_lib.mps_safe_cat, mps_compat.mps_safe_cat)

    def test_cache_send_recv_preserves_cpu_values(self) -> None:
        expected = torch.tile(self.tensor[:, :, :1], (1, 1, 2, 1, 1))

        actual = context_parallel_lib.cache_send_recv([self.tensor], cache_size=0, times=2)

        self.assertTrue(torch.equal(actual, expected))

    def test_extend_head_preserves_cpu_values(self) -> None:
        expected = torch.cat(
            (torch.tile(self.tensor[:, :, :1], (1, 1, 2, 1, 1)), self.tensor), dim=2
        )

        actual = causal_inflation_lib.extend_head(self.tensor, times=2)

        self.assertTrue(torch.equal(actual, expected))

    def test_split_causal_conv_routes_final_aggregation_through_safe_cat(self) -> None:
        conv = causal_inflation_lib.InflatedCausalConv3d(
            1,
            1,
            kernel_size=1,
            inflation_mode="tail",
            bias=False,
        )
        conv.set_memory_limit(1e-8)
        input_tensor = torch.ones((1, 1, 1, 2, 2), dtype=torch.float32)

        with mock.patch.object(
            causal_inflation_lib, "mps_safe_cat", wraps=mps_compat.mps_safe_cat
        ) as safe_cat:
            actual = conv.memory_limit_conv(input_tensor)

        self.assertEqual(actual.shape, input_tensor.shape)
        safe_cat.assert_called_once()

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_mps_repeat_first_frame_uses_cpu_materialization_for_large_backing_tensor(self) -> None:
        tensor = self.tensor.to("mps")
        expected = torch.tile(self.tensor[:, :, :1], (1, 1, 1, 1, 1)).to("mps")

        with (
            mock.patch.object(mps_compat, "_MPS_INT_MAX_ELEMENTS", 10),
            mock.patch.object(mps_compat, "_tile_on_cpu", wraps=mps_compat._tile_on_cpu) as cpu_tile,
        ):
            actual = mps_compat.repeat_first_frame(tensor, times=1)

        self.assertEqual(actual.device.type, "mps")
        self.assertTrue(torch.equal(actual.cpu(), expected.cpu()))
        self.assertTrue(actual.is_contiguous())
        cpu_tile.assert_called_once()

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_mps_safe_cat_uses_cpu_aggregation_when_forced(self) -> None:
        left = self.tensor[:, :, :1].to("mps")
        right = self.tensor[:, :, 1:].to("mps")
        expected = torch.cat((left.cpu(), right.cpu()), dim=2)

        with mock.patch.object(mps_compat, "_MPS_INT_MAX_ELEMENTS", 1):
            actual = mps_compat.mps_safe_cat((left, right), dim=2)

        self.assertEqual(actual.device.type, "mps")
        self.assertTrue(torch.equal(actual.cpu(), expected))


if __name__ == "__main__":
    unittest.main()
