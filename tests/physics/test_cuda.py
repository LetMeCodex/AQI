import pytest
import torch

from tests.physics.cases import puff_run

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU"),
]


def test_cuda_float32_matches_cpu_float64():
    gpu, _, _, _ = puff_run(100, device="cuda", dtype=torch.float32)
    cpu, _, _, _ = puff_run(100)
    diff = (gpu.double().cpu() - cpu).abs().max() / cpu.abs().max()
    # Measured on the RTX 4050: 4.5e-7 (float32 round-off over 173 steps).
    assert diff.item() < 1e-5
