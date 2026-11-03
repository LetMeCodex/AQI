import torch

from dhundh.physics.analytic import gaussian_puff
from dhundh.physics.grid import Grid
from dhundh.physics.solver import Forcing, integrate

# 200 km periodic square, 10 km puff, wind (10, 5) m/s, K = 500 m²/s, 3 hours.
L, SIGMA0, X0, Y0 = 200e3, 10e3, 50e3, 70e3
U, V, K, T_END = 10.0, 5.0, 500.0, 3 * 3600.0


def puff_run(n: int, device="cpu", dtype=torch.float64):
    grid = Grid(nx=n, ny=n, dx=L / n, dy=L / n)
    X, Y = grid.centers(dtype=dtype, device=device)
    C0 = gaussian_puff(X, Y, 0.0, X0, Y0, SIGMA0, U, V, K, 1.0).unsqueeze(0)
    full = lambda value: torch.full((n, n), value, dtype=dtype, device=device)
    f = Forcing(u=full(U), v=full(V), K=full(K), h=full(1000.0),
                E=torch.zeros(1, n, n, dtype=dtype, device=device), vd=0.0)
    C = integrate(C0, f, grid, T_END, "periodic")
    exact = gaussian_puff(X, Y, T_END, X0, Y0, SIGMA0, U, V, K, 1.0).unsqueeze(0)
    return C, exact, grid, C0
