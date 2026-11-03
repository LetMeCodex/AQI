from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Grid:
    """Uniform rectilinear grid. Axis -1 is x (east), axis -2 is y (north)."""

    nx: int
    ny: int
    dx: float
    dy: float

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy

    def centers(
        self,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[Tensor, Tensor]:
        dtype = dtype or torch.get_default_dtype()
        x = (torch.arange(self.nx, dtype=dtype, device=device) + 0.5) * self.dx
        y = (torch.arange(self.ny, dtype=dtype, device=device) + 0.5) * self.dy
        Y, X = torch.meshgrid(y, x, indexing="ij")
        return X, Y
