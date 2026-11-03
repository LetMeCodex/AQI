"""Flux-form horizontal diffusion with face-averaged diffusivity (mass-conserving)."""
from torch import Tensor

from dhundh.physics.boundary import BC, pad_last


def diffusive_divergence_x(C: Tensor, K: Tensor, dx: float, bc: BC) -> Tensor:
    P = pad_last(C, 1, bc)
    Kp = pad_last(K, 1, bc)
    k_face = 0.5 * (Kp[..., :-1] + Kp[..., 1:])
    flux = -k_face * (P[..., 1:] - P[..., :-1]) / dx
    return (flux[..., 1:] - flux[..., :-1]) / dx


def diffusive_divergence(C: Tensor, K: Tensor, dx: float, dy: float, bc: BC) -> Tensor:
    t = lambda a: a.transpose(-1, -2)
    return diffusive_divergence_x(C, K, dx, bc) + t(diffusive_divergence_x(t(C), t(K), dy, bc))
