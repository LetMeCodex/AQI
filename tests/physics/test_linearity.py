import math

import pytest
import torch

from dhundh.physics.grid import Grid
from dhundh.physics.solver import Forcing, integrate

N, DX, T_END = 80, 1000.0, 4 * 3600.0


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    grid = Grid(nx=N, ny=N, dx=DX, dy=DX)
    X, Y = grid.centers()
    span = N * DX
    u = 4 + 3 * torch.sin(2 * math.pi * Y / span)
    v = 2 * torch.cos(2 * math.pi * X / span)
    K = torch.full((N, N), 200.0)
    h = 300 + 200 * torch.rand(N, N)
    E1 = torch.zeros(1, N, N)
    E1[0, 10:14, 10:14] = 5.0
    E2 = torch.zeros(1, N, N)
    E2[0, 50:53, 30:33] = 8.0

    def run(E):
        f = Forcing(u=u, v=v, K=K, h=h, E=E, vd=0.002)
        return integrate(torch.zeros_like(E), f, grid, T_END, "zero_gradient")

    return run, E1, E2


def test_scaling_a_source_scales_its_field(setup):
    run, E1, _ = setup
    base = run(E1)
    scaled = run(0.37 * E1)
    assert ((scaled - 0.37 * base).abs().max() / base.abs().max()).item() < 1e-12


def test_sources_add_up(setup):
    # The van Leer limiter is not exactly additive; measured deviation is 8.6e-10.
    run, E1, E2 = setup
    a, b, both = run(E1), run(E2), run(E1 + E2)
    assert ((both - a - b).abs().max() / both.abs().max()).item() < 1e-6


def test_batched_tracers_equal_separate_runs(setup):
    run, E1, E2 = setup
    batched = run(torch.cat([E1, E2]))
    assert torch.equal(batched[0], run(E1)[0])
    assert torch.equal(batched[1], run(E2)[0])


def test_fields_stay_non_negative(setup):
    run, E1, E2 = setup
    assert run(torch.cat([E1, E2])).min().item() > -1e-12
