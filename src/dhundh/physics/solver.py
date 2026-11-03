"""Column advection–diffusion–source–deposition solver for k tagged tracers.

    ∂C/∂t + ∇·(uC) = ∇·(K∇C) + E/h − (vd/h)·C

Time stepping is SSP-RK2 (Heun), which keeps the monotone spatial scheme
positivity-preserving under the step limits in stable_dt.
"""
import math
from dataclasses import dataclass

from torch import Tensor

from dhundh.physics.advection import advective_divergence
from dhundh.physics.boundary import BC
from dhundh.physics.diffusion import diffusive_divergence
from dhundh.physics.grid import Grid


@dataclass(frozen=True)
class Forcing:
    u: Tensor  # (ny, nx) m/s, eastward
    v: Tensor  # (ny, nx) m/s, northward
    K: Tensor  # (ny, nx) m²/s
    h: Tensor  # (ny, nx) m, boundary-layer height
    E: Tensor  # (k, ny, nx) µg/m²/s
    vd: float  # m/s, dry deposition velocity


def tendency(C: Tensor, f: Forcing, grid: Grid, bc: BC) -> Tensor:
    return (
        -advective_divergence(C, f.u, f.v, grid.dx, grid.dy, bc)
        - diffusive_divergence(C, f.K, grid.dx, grid.dy, bc)
        + f.E / f.h
        - (f.vd / f.h) * C
    )


def stable_dt(f: Forcing, grid: Grid, cfl: float = 0.5, max_dt: float = 300.0) -> float:
    limits = [max_dt]
    adv = (f.u.abs() / grid.dx + f.v.abs() / grid.dy).max().item()
    if adv > 0:
        limits.append(cfl / adv)
    k_max = f.K.max().item()
    if k_max > 0:
        limits.append(cfl / (2 * k_max * (1 / grid.dx**2 + 1 / grid.dy**2)))
    if f.vd > 0:
        limits.append(cfl * f.h.min().item() / f.vd)
    return min(limits)


def integrate(C0: Tensor, f: Forcing, grid: Grid, t_end: float, bc: BC) -> Tensor:
    steps = math.ceil(t_end / stable_dt(f, grid))
    dt = t_end / steps
    C = C0
    for _ in range(steps):
        C1 = C + dt * tendency(C, f, grid, bc)
        C = 0.5 * C + 0.5 * (C1 + dt * tendency(C1, f, grid, bc))
    return C
