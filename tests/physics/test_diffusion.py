import math

import torch

from dhundh.physics.diffusion import diffusive_divergence


def test_uniform_field_does_not_diffuse():
    C = torch.full((1, 6, 6), 7.0)
    K = torch.full((6, 6), 300.0)
    div = diffusive_divergence(C, K, 1000.0, 1000.0, "periodic")
    assert torch.allclose(div, torch.zeros_like(div), atol=1e-14)


def test_cosine_mode_decays_at_exact_discrete_rate():
    nx, dx, K0 = 32, 500.0, 250.0
    k = 2 * math.pi / (nx * dx)
    x = (torch.arange(nx) + 0.5) * dx
    C = torch.cos(k * x).repeat(4, 1).unsqueeze(0)
    K = torch.full((4, nx), K0)
    div = diffusive_divergence(C, K, dx, dx, "periodic")
    rate = K0 * (2 - 2 * math.cos(k * dx)) / dx**2
    assert torch.allclose(div, rate * C, atol=1e-14)


def test_variable_diffusivity_conserves_mass():
    torch.manual_seed(2)
    C = torch.rand(2, 12, 14)
    K = torch.rand(12, 14) * 800
    div = diffusive_divergence(C, K, 600.0, 800.0, "periodic")
    assert torch.allclose(div.sum(dim=(-1, -2)), torch.zeros(2), atol=1e-12)
