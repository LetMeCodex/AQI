"""Exact solutions used to verify the solver."""
import math

import torch
from torch import Tensor


def gaussian_puff(
    X: Tensor, Y: Tensor, t: float, x0: float, y0: float,
    sigma0: float, u: float, v: float, K: float, mass: float,
) -> Tensor:
    """Column puff under constant wind (u, v) and diffusivity K: variance grows as sigma0² + 2Kt."""
    s2 = sigma0**2 + 2 * K * t
    r2 = (X - (x0 + u * t)) ** 2 + (Y - (y0 + v * t)) ** 2
    return mass / (2 * math.pi * s2) * torch.exp(-r2 / (2 * s2))
