import math

import torch

from dhundh.physics.grid import Grid
from dhundh.physics.solver import Forcing, integrate, stable_dt
from tests.physics.cases import puff_run


def rel_l2(a, b):
    return (torch.linalg.norm(a - b) / torch.linalg.norm(b)).item()


def test_puff_matches_exact_solution_with_second_order_convergence():
    C100, exact100, _, _ = puff_run(100)
    C200, exact200, _, _ = puff_run(200)
    e100, e200 = rel_l2(C100, exact100), rel_l2(C200, exact200)
    # Measured in prototyping: 0.0706 and 0.0180 (ratio 3.9).
    assert e100 < 0.08
    assert e200 < 0.025
    assert e100 / e200 > 3.0


def test_mass_is_conserved_to_machine_precision():
    C, _, grid, C0 = puff_run(100)
    m0 = C0.sum() * grid.cell_area
    m1 = C.sum() * grid.cell_area
    assert abs((m1 - m0) / m0).item() < 1e-12


def test_sharp_front_stays_positive_and_bounded():
    n = 100
    grid = Grid(nx=n, ny=n, dx=1000.0, dy=1000.0)
    C0 = torch.zeros(1, n, n)
    C0[:, 40:60, 40:60] = 100.0
    full = lambda value: torch.full((n, n), value)
    f = Forcing(u=full(7.0), v=full(-3.0), K=full(0.0), h=full(500.0),
                E=torch.zeros(1, n, n), vd=0.0)
    C = integrate(C0, f, grid, 6 * 3600.0, "periodic")
    assert C.min().item() > -1e-12
    assert C.max().item() <= 100.0 + 1e-9


def test_uniform_source_and_deposition_reach_exact_steady_approach():
    # dC/dt = E/h − (vd/h)C  ⇒  C(t) = (E/vd)(1 − exp(−vd t / h))
    n, E0, h0, vd, t = 4, 2.0, 500.0, 0.002, 6 * 3600.0
    grid = Grid(nx=n, ny=n, dx=1000.0, dy=1000.0)
    full = lambda value: torch.full((n, n), value)
    f = Forcing(u=full(0.0), v=full(0.0), K=full(0.0), h=full(h0),
                E=torch.full((1, n, n), E0), vd=vd)
    C = integrate(torch.zeros(1, n, n), f, grid, t, "periodic")
    exact = (E0 / vd) * (1 - math.exp(-vd * t / h0))
    assert torch.allclose(C, torch.full_like(C, exact), rtol=1e-6)


def test_stable_dt_respects_every_limit():
    n = 4
    grid = Grid(nx=n, ny=n, dx=1000.0, dy=1000.0)
    full = lambda value: torch.full((n, n), value)
    f = Forcing(u=full(10.0), v=full(10.0), K=full(0.0), h=full(1000.0),
                E=torch.zeros(1, n, n), vd=0.0)
    assert stable_dt(f, grid) == 0.5 / (10 / 1000 + 10 / 1000)
    calm = Forcing(u=full(0.0), v=full(0.0), K=full(0.0), h=full(1000.0),
                   E=torch.zeros(1, n, n), vd=0.0)
    assert stable_dt(calm, grid) == 300.0
