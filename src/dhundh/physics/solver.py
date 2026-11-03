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
    """A step sufficient for each SSP-RK2 stage to be positivity-preserving: per-axis
    maximum advective rates, the explicit-diffusion rate and the deposition rate are
    summed, so each stage is a convex combination of individually stable updates."""
    rate = f.u.abs().max().item() / grid.dx + f.v.abs().max().item() / grid.dy
    rate += 2 * f.K.max().item() * (1 / grid.dx**2 + 1 / grid.dy**2)
    if f.vd > 0:
        rate += f.vd / f.h.min().item()
    return min(max_dt, cfl / rate) if rate > 0 else max_dt


def integrate(C0: Tensor, f: Forcing, grid: Grid, t_end: float, bc: BC) -> Tensor:
    if t_end < 0:
        raise ValueError(f"t_end must be >= 0, got {t_end}")
    if t_end == 0:
        return C0
    steps = math.ceil(t_end / stable_dt(f, grid))
    dt = t_end / steps
    C = C0
    for _ in range(steps):
        C1 = C + dt * tendency(C, f, grid, bc)
        C = 0.5 * C + 0.5 * (C1 + dt * tendency(C1, f, grid, bc))
    return C
