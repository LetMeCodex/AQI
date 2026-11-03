import math

import pytest
import torch

from dhundh.physics.grid import Grid
from dhundh.physics.solver import Forcing, integrate, stable_dt, tendency
from tests.physics.cases import puff_run


def rel_l2(a, b):
    return (torch.linalg.norm(a - b) / torch.linalg.norm(b)).item()


def test_puff_matches_exact_solution_with_second_order_convergence():
    C100, exact100, _, _ = puff_run(100)
    C200, exact200, _, _ = puff_run(200)
    e100, e200 = rel_l2(C100, exact100), rel_l2(C200, exact200)
    # Measured: e100 = 0.0699, e200 = 0.0176 (ratio 3.98).
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


def test_stable_dt_sums_the_rates_of_all_processes():
    n = 4
    grid = Grid(nx=n, ny=n, dx=1000.0, dy=1000.0)
    full = lambda value: torch.full((n, n), value)
    f = Forcing(u=full(2.5), v=full(2.5), K=full(1250.0), h=full(500.0),
                E=torch.zeros(1, n, n), vd=2.5)
    adv, diff, dep = 2.5 / 1000 + 2.5 / 1000, 2 * 1250 * (2 / 1000**2), 2.5 / 500
    assert stable_dt(f, grid) == pytest.approx(0.5 / (adv + diff + dep))


def test_all_processes_binding_stays_stable_and_positive():
    n = 60
    grid = Grid(nx=n, ny=n, dx=1000.0, dy=1000.0)
    torch.manual_seed(3)
    C0 = torch.rand(1, n, n) * 100
    full = lambda value: torch.full((n, n), value)
    f = Forcing(u=full(2.5), v=full(2.5), K=full(1250.0), h=full(500.0),
                E=torch.zeros(1, n, n), vd=2.5)
    C = integrate(C0, f, grid, 3000.0, "periodic")
    assert C.min().item() > -1e-12
    assert C.max().item() <= C0.max().item() + 1e-9


def test_step_keeps_euler_stage_positive_under_alternating_wind():
    # Rows alternate (u=5, v=0) and (u=0, v=5); a per-cell rate bound lets one stage go negative.
    n = 40
    grid = Grid(nx=n, ny=n, dx=1000.0, dy=1000.0)
    rows = torch.arange(n).unsqueeze(1).expand(n, n)
    zero = torch.zeros(n, n)
    f = Forcing(u=torch.where(rows % 2 == 0, 5.0, 0.0), v=torch.where(rows % 2 == 1, 5.0, 0.0),
                K=zero, h=zero + 500.0, E=torch.zeros(1, n, n), vd=0.0)
    C = torch.zeros(1, n, n)
    C[0, 10, 10], C[0, 10, 11], C[0, 11, 10] = 1.0, 1e6, 1e6
    dt = stable_dt(f, grid)
    stage = C + dt * tendency(C, f, grid, "periodic")
    assert stage.min().item() >= -1e-12


def test_integrate_rejects_negative_time():
    n = 4
    grid = Grid(nx=n, ny=n, dx=1000.0, dy=1000.0)
    zero = torch.zeros(n, n)
    f = Forcing(u=zero, v=zero, K=zero, h=zero + 500.0, E=torch.zeros(1, n, n), vd=0.0)
    with pytest.raises(ValueError):
        integrate(torch.zeros(1, n, n), f, grid, -1.0, "periodic")


def test_integrate_zero_time_returns_input():
    n = 4
    grid = Grid(nx=n, ny=n, dx=1000.0, dy=1000.0)
    zero = torch.zeros(n, n)
    f = Forcing(u=zero + 3.0, v=zero, K=zero, h=zero + 500.0, E=torch.zeros(1, n, n), vd=0.0)
    C0 = torch.rand(1, n, n)
    assert torch.equal(integrate(C0, f, grid, 0.0, "periodic"), C0)
