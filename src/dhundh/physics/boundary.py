from typing import Literal

import torch.nn.functional as F
from torch import Tensor

BC = Literal["periodic", "zero_gradient"]

_MODES = {"periodic": "circular", "zero_gradient": "replicate"}


def pad_last(x: Tensor, n: int, bc: BC) -> Tensor:
    """Pad the last axis with n ghost cells per side."""
    if bc not in _MODES:
        raise ValueError(f"unknown boundary condition {bc!r}")
    squeeze = x.dim() == 2
    if squeeze:
        x = x.unsqueeze(0)
    y = F.pad(x, (n, n), mode=_MODES[bc])
    return y.squeeze(0) if squeeze else y
