"""Conservative flux-form advection, MUSCL reconstruction with the van Leer limiter.

Flux form conserves mass exactly. With the step bound in solver.stable_dt, the
limiter keeps each stage positivity-preserving; in divergence-free flow it also
creates no new extrema (convergent flow legitimately concentrates mass).
"""
import torch
from torch import Tensor

from dhundh.physics.boundary import BC, pad_last


def van_leer_slopes(P: Tensor) -> Tensor:
    dm = P[..., 1:-1] - P[..., :-2]
    dp = P[..., 2:] - P[..., 1:-1]
    den = dm + dp
    safe = torch.where(den == 0, torch.ones_like(den), den)
    return torch.where(dm * dp > 0, 2 * dm * dp / safe, torch.zeros_like(dm))


def advective_divergence_x(C: Tensor, u: Tensor, dx: float, bc: BC) -> Tensor:
    nx = C.shape[-1]
    P = pad_last(C, 2, bc)
    s = van_leer_slopes(P)
    c_left = P[..., 1 : nx + 2] + 0.5 * s[..., 0 : nx + 1]
    c_right = P[..., 2 : nx + 3] - 0.5 * s[..., 1 : nx + 2]
    U = pad_last(u, 1, bc)
    u_face = 0.5 * (U[..., :-1] + U[..., 1:])
    flux = torch.where(u_face > 0, u_face * c_left, u_face * c_right)
    return (flux[..., 1:] - flux[..., :-1]) / dx


def advective_divergence(C: Tensor, u: Tensor, v: Tensor, dx: float, dy: float, bc: BC) -> Tensor:
    t = lambda a: a.transpose(-1, -2)
    return advective_divergence_x(C, u, dx, bc) + t(advective_divergence_x(t(C), t(v), dy, bc))
