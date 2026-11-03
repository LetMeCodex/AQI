# Dhundh Plan 1 — Foundation, Physics Engine, Health Math

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A tested Python package `dhundh` containing a verified, mass-conserving, positivity-preserving 2D advection–diffusion–source–deposition solver for N tagged tracers (CPU and CUDA), plus the pure health functions (India AQI, AQLI life-years, cigarette equivalents, GEMM hazard ratio).

**Architecture:** Finite-volume solver in PyTorch, tensors shaped `(k, ny, nx)` for k tracers. Advection uses MUSCL reconstruction with the van Leer limiter in conservative flux form; diffusion uses face-averaged diffusivity; time stepping uses SSP-RK2 (Heun) with a CFL/diffusion/deposition-limited timestep. Health functions are dependency-free pure Python. Every module is proven against analytic or published reference values.

**Tech Stack:** Python 3.11, PyTorch 2.5.1 (+cu121, already installed system-wide), NumPy, pytest 8.

**Spec:** `docs/superpowers/specs/2026-11-03-dhundh-design.md` (§4 physics, §6 health). This is Plan 1 of 6; later plans: 2 data ingestion, 3 physics on the real Delhi domain (nesting + its §4.5 test 5, Smagorinsky K, emissions, fires), 4 ML corrector (plus the short-term daily-risk coefficient, which first needs its primary paper verified), 5 hourly pipeline, 6 frontend (including "best time outside", which needs forecasts).

## Global Constraints

- Python `>=3.11`; PyTorch `>=2.5`. The venv is created with `--system-site-packages` so the existing CUDA build of torch is reused (no 2.5 GB download).
- Units are SI throughout: metres, seconds, µg/m³ for concentration, µg/m²/s for emission flux, m/s for wind and deposition velocity, m²/s for diffusivity. CO breakpoints are in mg/m³ (CPCB convention).
- Tensor shape convention: concentration and emissions `(k, ny, nx)`; 2D forcing fields `(ny, nx)`. Axis -1 is x (east), axis -2 is y (north).
- Boundary condition names are exactly `"periodic"` and `"zero_gradient"`.
- Advection is solved in conservative flux form `∇·(uC)`.
- Verification tests run in float64; the CUDA parity test runs float32.
- `sub_index` returns floats; the reported AQI is an `int` rounded half-up.
- No network access, API keys, or secrets in this plan.
- Commands are written for Windows PowerShell from the repo root.

---

## File structure

```
pyproject.toml                       package + pytest config
.gitignore
src/dhundh/__init__.py               version
src/dhundh/physics/__init__.py
src/dhundh/physics/grid.py           Grid dataclass (shape, spacing, cell centres)
src/dhundh/physics/boundary.py       ghost-cell padding for bc names
src/dhundh/physics/advection.py      MUSCL/van Leer flux-form advection divergence
src/dhundh/physics/diffusion.py      flux-form diffusion divergence
src/dhundh/physics/solver.py         Forcing, tendency, stable_dt, integrate (SSP-RK2)
src/dhundh/physics/analytic.py       exact Gaussian puff solution
src/dhundh/health/__init__.py
src/dhundh/health/aqi.py             CPCB National AQI
src/dhundh/health/exposure.py        AQLI life-years, cigarette equivalents
src/dhundh/health/gemm.py            GEMM hazard ratio
tests/__init__.py
tests/conftest.py                    float64 default dtype
tests/test_smoke.py
tests/physics/__init__.py
tests/physics/cases.py               shared puff case (CPU + CUDA)
tests/physics/test_grid_boundary.py
tests/physics/test_advection.py
tests/physics/test_diffusion.py
tests/physics/test_solver.py
tests/physics/test_linearity.py
tests/physics/test_cuda.py
tests/health/__init__.py
tests/health/test_aqi.py
tests/health/test_exposure.py
tests/health/test_gemm.py
```

---

### Task 1: Repository scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/dhundh/__init__.py`, `src/dhundh/physics/__init__.py`, `src/dhundh/health/__init__.py`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_smoke.py`, `tests/physics/__init__.py`, `tests/health/__init__.py`

**Interfaces:**
- Produces: importable package `dhundh` with `dhundh.__version__ == "0.1.0"`; test suite defaulting to float64.

- [ ] **Step 1: Initialise git**

Run: `git init -b main`
Expected: `Initialized empty Git repository`

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "dhundh"
version = "0.1.0"
description = "Physics-grounded air-quality forecasting for Delhi NCR"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "torch>=2.5"]

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["cuda: requires a CUDA GPU"]
```

- [ ] **Step 3: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
data/
*.onnx
```

- [ ] **Step 4: Write package and test package files**

`src/dhundh/__init__.py`:
```python
__version__ = "0.1.0"
```

`src/dhundh/physics/__init__.py`, `src/dhundh/health/__init__.py`, `tests/__init__.py`, `tests/physics/__init__.py`, `tests/health/__init__.py`: empty files.

`tests/conftest.py`:
```python
import torch

torch.set_default_dtype(torch.float64)
```

- [ ] **Step 5: Write the smoke test**

`tests/test_smoke.py`:
```python
import dhundh


def test_version():
    assert dhundh.__version__ == "0.1.0"
```

- [ ] **Step 6: Create the venv and install**

Run:
```powershell
python -m venv .venv --system-site-packages
.venv\Scripts\python -m pip install -e ".[dev]"
```
Expected: `Successfully installed dhundh-0.1.0` (torch reported as already satisfied).

- [ ] **Step 7: Run the smoke test**

Run: `.venv\Scripts\python -m pytest tests/test_smoke.py -v`
Expected: `1 passed`

- [ ] **Step 8: Commit**

```powershell
git add pyproject.toml .gitignore src tests docs
git commit -m "chore: scaffold dhundh package with spec and plan"
```

---

### Task 2: Grid and boundary padding

**Files:**
- Create: `src/dhundh/physics/grid.py`, `src/dhundh/physics/boundary.py`
- Test: `tests/physics/test_grid_boundary.py`

**Interfaces:**
- Produces:
  - `Grid(nx: int, ny: int, dx: float, dy: float)` frozen dataclass; `Grid.centers(dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> tuple[Tensor, Tensor]` returning `(X, Y)`, each `(ny, nx)`, cell-centre coordinates in metres; property `Grid.cell_area -> float`.
  - `BC = Literal["periodic", "zero_gradient"]`
  - `pad_last(x: Tensor, n: int, bc: BC) -> Tensor` — pads the last axis with `n` ghost cells on each side; accepts 2D or 3D tensors.

- [ ] **Step 1: Write the failing tests**

`tests/physics/test_grid_boundary.py`:
```python
import pytest
import torch

from dhundh.physics.boundary import pad_last
from dhundh.physics.grid import Grid


def test_centers_are_cell_midpoints():
    g = Grid(nx=4, ny=3, dx=1000.0, dy=500.0)
    X, Y = g.centers()
    assert X.shape == (3, 4) and Y.shape == (3, 4)
    assert X[0].tolist() == [500.0, 1500.0, 2500.0, 3500.0]
    assert Y[:, 0].tolist() == [250.0, 750.0, 1250.0]
    assert g.cell_area == 500_000.0


def test_pad_periodic_wraps():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    assert pad_last(x, 2, "periodic").tolist() == [[2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]]


def test_pad_zero_gradient_replicates_edges():
    x = torch.tensor([[[1.0, 2.0, 3.0]]])
    assert pad_last(x, 1, "zero_gradient").tolist() == [[[1.0, 1.0, 2.0, 3.0, 3.0]]]


def test_pad_rejects_unknown_bc():
    with pytest.raises(ValueError):
        pad_last(torch.zeros(2, 2), 1, "reflect")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/physics/test_grid_boundary.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhundh.physics.boundary'`

- [ ] **Step 3: Implement**

`src/dhundh/physics/grid.py`:
```python
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

    def centers(self, dtype: torch.dtype | None = None, device=None) -> tuple[Tensor, Tensor]:
        dtype = dtype or torch.get_default_dtype()
        x = (torch.arange(self.nx, dtype=dtype, device=device) + 0.5) * self.dx
        y = (torch.arange(self.ny, dtype=dtype, device=device) + 0.5) * self.dy
        Y, X = torch.meshgrid(y, x, indexing="ij")
        return X, Y
```

`src/dhundh/physics/boundary.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/physics/test_grid_boundary.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```powershell
git add src/dhundh/physics/grid.py src/dhundh/physics/boundary.py tests/physics/test_grid_boundary.py
git commit -m "feat(physics): add grid and ghost-cell boundary padding"
```

---

### Task 3: Flux-form advection (MUSCL + van Leer)

**Files:**
- Create: `src/dhundh/physics/advection.py`
- Test: `tests/physics/test_advection.py`

**Interfaces:**
- Consumes: `pad_last`, `BC` from Task 2.
- Produces:
  - `van_leer_slopes(P: Tensor) -> Tensor` — limited slopes for interior points of a padded last axis (output length = input length − 2).
  - `advective_divergence_x(C: Tensor, u: Tensor, dx: float, bc: BC) -> Tensor` — `∂(uC)/∂x`, same shape as `C`.
  - `advective_divergence(C: Tensor, u: Tensor, v: Tensor, dx: float, dy: float, bc: BC) -> Tensor` — `∇·(uC)`.

- [ ] **Step 1: Write the failing tests**

`tests/physics/test_advection.py`:
```python
import torch

from dhundh.physics.advection import advective_divergence, van_leer_slopes


def test_slopes_follow_a_linear_ramp():
    P = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    assert torch.allclose(van_leer_slopes(P), torch.tensor([1.0, 1.0, 1.0]))


def test_slopes_vanish_at_extrema():
    P = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
    assert torch.equal(van_leer_slopes(P), torch.zeros(3))


def test_uniform_field_has_zero_divergence_under_uniform_wind():
    C = torch.full((2, 8, 10), 42.0)
    u = torch.full((8, 10), 3.0)
    v = torch.full((8, 10), -1.5)
    div = advective_divergence(C, u, v, 1000.0, 1000.0, "periodic")
    assert torch.allclose(div, torch.zeros_like(div), atol=1e-12)


def test_periodic_flux_form_conserves_mass_exactly():
    torch.manual_seed(1)
    C = torch.rand(3, 16, 20) * 100
    u = torch.randn(16, 20) * 5
    v = torch.randn(16, 20) * 5
    div = advective_divergence(C, u, v, 750.0, 900.0, "periodic")
    total = div.sum(dim=(-1, -2))
    assert torch.allclose(total, torch.zeros(3), atol=1e-10)


def test_positive_wind_moves_mass_downstream():
    # 4 rows: periodic padding needs at least 2 cells per axis.
    C = torch.zeros(1, 4, 10)
    C[0, :, 4] = 1.0
    u = torch.full((4, 10), 1.0)
    v = torch.zeros(4, 10)
    tend = -advective_divergence(C, u, v, 1.0, 1.0, "periodic")
    assert tend[0, 0, 4] < 0
    assert tend[0, 0, 5] > 0
    assert tend[0, 0, 3] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/physics/test_advection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhundh.physics.advection'`

- [ ] **Step 3: Implement**

`src/dhundh/physics/advection.py`:
```python
"""Conservative flux-form advection, MUSCL reconstruction with the van Leer limiter.

Flux form conserves mass exactly; the limiter is monotone, so no new extrema
and no negative concentrations appear at sharp fronts.
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/physics/test_advection.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```powershell
git add src/dhundh/physics/advection.py tests/physics/test_advection.py
git commit -m "feat(physics): add MUSCL van Leer flux-form advection"
```

---

### Task 4: Flux-form diffusion

**Files:**
- Create: `src/dhundh/physics/diffusion.py`
- Test: `tests/physics/test_diffusion.py`

**Interfaces:**
- Consumes: `pad_last`, `BC` from Task 2.
- Produces:
  - `diffusive_divergence_x(C: Tensor, K: Tensor, dx: float, bc: BC) -> Tensor` — `∂/∂x(−K ∂C/∂x)`.
  - `diffusive_divergence(C: Tensor, K: Tensor, dx: float, dy: float, bc: BC) -> Tensor` — `∇·(−K∇C)`. The solver subtracts this.

- [ ] **Step 1: Write the failing tests**

`tests/physics/test_diffusion.py`:
```python
import math

import torch

from dhundh.physics.diffusion import diffusive_divergence


def test_uniform_field_does_not_diffuse():
    C = torch.full((1, 6, 6), 7.0)
    K = torch.full((6, 6), 300.0)
    div = diffusive_divergence(C, K, 1000.0, 1000.0, "periodic")
    assert torch.allclose(div, torch.zeros_like(div), atol=1e-14)


def test_cosine_mode_decays_at_exact_discrete_rate():
    nx, dx, K0 = 32, 500.0, 250.0
    k = 2 * math.pi / (nx * dx)
    x = (torch.arange(nx) + 0.5) * dx
    C = torch.cos(k * x).repeat(4, 1).unsqueeze(0)
    K = torch.full((4, nx), K0)
    div = diffusive_divergence(C, K, dx, dx, "periodic")
    rate = K0 * (2 - 2 * math.cos(k * dx)) / dx**2
    assert torch.allclose(div, rate * C, atol=1e-14)


def test_variable_diffusivity_conserves_mass():
    torch.manual_seed(2)
    C = torch.rand(2, 12, 14)
    K = torch.rand(12, 14) * 800
    div = diffusive_divergence(C, K, 600.0, 800.0, "periodic")
    assert torch.allclose(div.sum(dim=(-1, -2)), torch.zeros(2), atol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/physics/test_diffusion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhundh.physics.diffusion'`

- [ ] **Step 3: Implement**

`src/dhundh/physics/diffusion.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/physics/test_diffusion.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```powershell
git add src/dhundh/physics/diffusion.py tests/physics/test_diffusion.py
git commit -m "feat(physics): add flux-form diffusion"
```

---

### Task 5: Solver (SSP-RK2) verified against the exact Gaussian puff

**Files:**
- Create: `src/dhundh/physics/solver.py`, `src/dhundh/physics/analytic.py`, `tests/physics/cases.py`
- Test: `tests/physics/test_solver.py`

**Interfaces:**
- Consumes: `Grid` (Task 2), `advective_divergence` (Task 3), `diffusive_divergence` (Task 4), `BC`.
- Produces:
  - `Forcing(u, v, K, h, E, vd)` frozen dataclass — `u, v, K, h`: `(ny, nx)` tensors; `E`: `(k, ny, nx)` tensor in µg/m²/s; `vd`: float m/s.
  - `tendency(C: Tensor, f: Forcing, grid: Grid, bc: BC) -> Tensor`
  - `stable_dt(f: Forcing, grid: Grid, cfl: float = 0.5, max_dt: float = 300.0) -> float`
  - `integrate(C0: Tensor, f: Forcing, grid: Grid, t_end: float, bc: BC) -> Tensor`
  - `gaussian_puff(X, Y, t, x0, y0, sigma0, u, v, K, mass) -> Tensor` — exact solution for constant wind and diffusivity.
  - `tests.physics.cases.puff_run(n: int, device="cpu", dtype=torch.float64) -> tuple[Tensor, Tensor, Grid, Tensor]` returning `(C_numeric, C_exact, grid, C0)`.

- [ ] **Step 1: Write the analytic solution and shared case**

`src/dhundh/physics/analytic.py`:
```python
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
```

`tests/physics/cases.py`:
```python
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
```

- [ ] **Step 2: Write the failing tests**

`tests/physics/test_solver.py`:
```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/physics/test_solver.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhundh.physics.solver'`

- [ ] **Step 4: Implement**

`src/dhundh/physics/solver.py`:
```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/physics/test_solver.py -v`
Expected: `5 passed`

- [ ] **Step 6: Commit**

```powershell
git add src/dhundh/physics/solver.py src/dhundh/physics/analytic.py tests/physics/cases.py tests/physics/test_solver.py
git commit -m "feat(physics): add SSP-RK2 solver verified against exact puff"
```

---

### Task 6: Tagged-tracer linearity and CUDA parity

These tests prove the properties the product depends on: what-if scaling (spec §4.4), attribution summing to the total, batched tracers matching separate runs, and the GPU giving the CPU's answer.

**Files:**
- Test: `tests/physics/test_linearity.py`, `tests/physics/test_cuda.py`

**Interfaces:**
- Consumes: `Grid`, `Forcing`, `integrate` (Task 5); `puff_run` (Task 5).
- Produces: no new code; regression guarantees.

- [ ] **Step 1: Write the tests**

`tests/physics/test_linearity.py`:
```python
import math

import pytest
import torch

from dhundh.physics.grid import Grid
from dhundh.physics.solver import Forcing, integrate

N, DX, T_END = 80, 1000.0, 4 * 3600.0


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    grid = Grid(nx=N, ny=N, dx=DX, dy=DX)
    X, Y = grid.centers()
    span = N * DX
    u = 4 + 3 * torch.sin(2 * math.pi * Y / span)
    v = 2 * torch.cos(2 * math.pi * X / span)
    K = torch.full((N, N), 200.0)
    h = 300 + 200 * torch.rand(N, N)
    E1 = torch.zeros(1, N, N)
    E1[0, 10:14, 10:14] = 5.0
    E2 = torch.zeros(1, N, N)
    E2[0, 50:53, 30:33] = 8.0

    def run(E):
        f = Forcing(u=u, v=v, K=K, h=h, E=E, vd=0.002)
        return integrate(torch.zeros_like(E), f, grid, T_END, "zero_gradient")

    return run, E1, E2


def test_scaling_a_source_scales_its_field(setup):
    run, E1, _ = setup
    base = run(E1)
    scaled = run(0.37 * E1)
    assert ((scaled - 0.37 * base).abs().max() / base.abs().max()).item() < 1e-12


def test_sources_add_up(setup):
    # The van Leer limiter is not exactly additive; measured deviation is 8.6e-10.
    run, E1, E2 = setup
    a, b, both = run(E1), run(E2), run(E1 + E2)
    assert ((both - a - b).abs().max() / both.abs().max()).item() < 1e-6


def test_batched_tracers_equal_separate_runs(setup):
    run, E1, E2 = setup
    batched = run(torch.cat([E1, E2]))
    assert torch.equal(batched[0], run(E1)[0])
    assert torch.equal(batched[1], run(E2)[0])


def test_fields_stay_non_negative(setup):
    run, E1, E2 = setup
    assert run(torch.cat([E1, E2])).min().item() > -1e-12
```

`tests/physics/test_cuda.py`:
```python
import pytest
import torch

from tests.physics.cases import puff_run

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU"),
]


def test_cuda_float32_matches_cpu_float64():
    gpu, _, _, _ = puff_run(100, device="cuda", dtype=torch.float32)
    cpu, _, _, _ = puff_run(100)
    diff = (gpu.double().cpu() - cpu).abs().max() / cpu.abs().max()
    # Measured on the RTX 4050 in prototyping: 3.2e-4 (float32 round-off over 162 steps).
    assert diff.item() < 1e-3
```

- [ ] **Step 2: Run the tests**

Run: `.venv\Scripts\python -m pytest tests/physics/test_linearity.py tests/physics/test_cuda.py -v`
Expected: `5 passed` (on the RTX 4050; the CUDA test shows `skipped` on CPU-only machines such as GitHub Actions).

If `test_sources_add_up` fails, do not loosen the bound blindly: print the measured deviation and confirm it is a limiter effect by re-running with `K` doubled (deviation must shrink as fields smooth).

- [ ] **Step 3: Commit**

```powershell
git add tests/physics/test_linearity.py tests/physics/test_cuda.py
git commit -m "test(physics): prove tracer scaling, additivity, batching, CUDA parity"
```

---

### Task 7: India National AQI (CPCB)

**Files:**
- Create: `src/dhundh/health/aqi.py`
- Test: `tests/health/test_aqi.py`

**Interfaces:**
- Produces:
  - `Pollutant = Literal["pm25", "pm10", "no2", "so2", "co", "o3", "nh3", "pb"]`
  - `BREAKPOINTS: dict[str, tuple[float, ...]]`, `AQI_POINTS = (0, 50, 100, 200, 300, 400, 500)`
  - `sub_index(pollutant: Pollutant, conc: float) -> float`
  - `category(aqi: float) -> str` — one of `"Good"`, `"Satisfactory"`, `"Moderate"`, `"Poor"`, `"Very poor"`, `"Severe"`.
  - `daily_concentration(pollutant: Pollutant, hourly: Sequence[float | None]) -> float | None`
  - `AqiResult(aqi: int, category: str, dominant: Pollutant, sub_indices: dict[str, float], capped: bool)` frozen dataclass.
  - `daily_aqi(hourly: Mapping[Pollutant, Sequence[float | None]]) -> AqiResult | None`

- [ ] **Step 1: Verify the breakpoints against the primary source**

Open the CPCB National AQI report (https://theasthmafiles.org/sites/default/files/artifacts/media/pdf/final_report_of_cpcb_on_air_quality_index_2014-15.pdf), breakpoint table. Confirm the band edges below for all eight pollutants. The Severe band's upper edge is not printed in the table: CPCB's calculator closes it with the same width as Very Poor (e.g. PM2.5 250 + 130 = 380), and that rule is encoded below. If any printed edge differs, correct `BREAKPOINTS` and the expected values in Step 2 before continuing.

- [ ] **Step 2: Write the failing tests**

`tests/health/test_aqi.py`:
```python
import pytest

from dhundh.health.aqi import category, daily_aqi, daily_concentration, sub_index


@pytest.mark.parametrize(
    "pollutant, conc, expected",
    [
        ("pm25", 0.0, 0.0),
        ("pm25", 30.0, 50.0),
        ("pm25", 45.0, 75.0),
        ("pm25", 100.0, 200 + 10 * 100 / 30),
        ("pm25", 300.0, 400 + 50 * 100 / 130),
        ("pm10", 200.0, 100 + 100 * 100 / 150),
        ("no2", 100.0, 120.0),
        ("co", 1.5, 75.0),
        ("o3", 180.0, 200 + 12 * 100 / 40),
    ],
)
def test_sub_index_interpolates_within_bands(pollutant, conc, expected):
    assert sub_index(pollutant, conc) == pytest.approx(expected)


def test_sub_index_caps_at_500():
    assert sub_index("pm25", 380.0) == 500.0
    assert sub_index("pm25", 900.0) == 500.0


def test_sub_index_rejects_negative():
    with pytest.raises(ValueError):
        sub_index("pm25", -1.0)


@pytest.mark.parametrize(
    "aqi, name",
    [(50, "Good"), (51, "Satisfactory"), (150, "Moderate"), (250, "Poor"), (350, "Very poor"), (450, "Severe")],
)
def test_category(aqi, name):
    assert category(aqi) == name


def test_daily_concentration_requires_16_hours():
    assert daily_concentration("pm25", [100.0] * 15 + [None] * 9) is None
    assert daily_concentration("pm25", [100.0] * 16 + [None] * 8) == 100.0


def test_eight_hour_pollutants_use_max_rolling_mean():
    hourly = [1.0] * 16 + [5.0] * 8
    assert daily_concentration("co", hourly) == pytest.approx(5.0)


def test_daily_aqi_takes_worst_sub_index():
    result = daily_aqi({"pm25": [100.0] * 24, "pm10": [200.0] * 24, "no2": [50.0] * 24})
    assert result.aqi == 233
    assert result.dominant == "pm25"
    assert result.category == "Poor"
    assert result.capped is False


def test_daily_aqi_needs_three_pollutants_including_pm():
    assert daily_aqi({"pm25": [100.0] * 24, "no2": [50.0] * 24}) is None
    assert daily_aqi({"no2": [50.0] * 24, "so2": [10.0] * 24, "co": [1.0] * 24}) is None


def test_daily_aqi_flags_capped_values():
    result = daily_aqi({"pm25": [900.0] * 24, "pm10": [200.0] * 24, "no2": [50.0] * 24})
    assert result.aqi == 500
    assert result.capped is True
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/health/test_aqi.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhundh.health.aqi'`

- [ ] **Step 4: Implement**

`src/dhundh/health/aqi.py`:
```python
"""India National Air Quality Index (CPCB, 2014).

Sub-indices interpolate linearly between breakpoints; the AQI is the worst
sub-index. PM/NO2/SO2/NH3/Pb use 24-hour means, CO/O3 the maximum 8-hour
mean. A valid AQI needs at least 3 pollutants, one of them PM2.5 or PM10,
each with at least 16 valid hours.
"""
import math
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

Pollutant = Literal["pm25", "pm10", "no2", "so2", "co", "o3", "nh3", "pb"]

AQI_POINTS = (0, 50, 100, 200, 300, 400, 500)

# µg/m³ (CO in mg/m³). The last edge closes Severe with the width of Very Poor.
BREAKPOINTS: dict[str, tuple[float, ...]] = {
    "pm25": (0, 30, 60, 90, 120, 250, 380),
    "pm10": (0, 50, 100, 250, 350, 430, 510),
    "no2": (0, 40, 80, 180, 280, 400, 520),
    "so2": (0, 40, 80, 380, 800, 1600, 2400),
    "co": (0, 1.0, 2.0, 10, 17, 34, 51),
    "o3": (0, 50, 100, 168, 208, 748, 1288),
    "nh3": (0, 200, 400, 800, 1200, 1800, 2400),
    "pb": (0, 0.5, 1.0, 2.0, 3.0, 3.5, 4.0),
}

EIGHT_HOUR = frozenset({"co", "o3"})
MIN_VALID_HOURS = 16
MIN_WINDOW_VALUES = 6
_CATEGORIES = ((50, "Good"), (100, "Satisfactory"), (200, "Moderate"), (300, "Poor"), (400, "Very poor"))


def sub_index(pollutant: Pollutant, conc: float) -> float:
    if conc < 0:
        raise ValueError(f"negative concentration {conc}")
    bp = BREAKPOINTS[pollutant]
    if conc >= bp[-1]:
        return 500.0
    for i in range(1, len(bp)):
        if conc <= bp[i]:
            lo, hi = bp[i - 1], bp[i]
            ilo, ihi = AQI_POINTS[i - 1], AQI_POINTS[i]
            return ilo + (ihi - ilo) * (conc - lo) / (hi - lo)
    raise AssertionError("unreachable")


def category(aqi: float) -> str:
    for upper, name in _CATEGORIES:
        if aqi <= upper:
            return name
    return "Severe"


def daily_concentration(pollutant: Pollutant, hourly: Sequence[float | None]) -> float | None:
    valid = [x for x in hourly if x is not None]
    if len(valid) < MIN_VALID_HOURS:
        return None
    if pollutant not in EIGHT_HOUR:
        return sum(valid) / len(valid)
    means = []
    for start in range(len(hourly) - 7):
        window = [x for x in hourly[start : start + 8] if x is not None]
        if len(window) >= MIN_WINDOW_VALUES:
            means.append(sum(window) / len(window))
    return max(means) if means else None


@dataclass(frozen=True)
class AqiResult:
    aqi: int
    category: str
    dominant: Pollutant
    sub_indices: dict[str, float]
    capped: bool


def daily_aqi(hourly: Mapping[Pollutant, Sequence[float | None]]) -> AqiResult | None:
    subs: dict[str, float] = {}
    for pollutant, series in hourly.items():
        conc = daily_concentration(pollutant, series)
        if conc is not None:
            subs[pollutant] = sub_index(pollutant, conc)
    if len(subs) < 3 or not ({"pm25", "pm10"} & subs.keys()):
        return None
    dominant = max(subs, key=subs.get)
    value = subs[dominant]
    aqi = math.floor(value + 0.5)
    return AqiResult(aqi=aqi, category=category(aqi), dominant=dominant, sub_indices=subs, capped=value >= 500.0)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/health/test_aqi.py -v`
Expected: `22 passed`

- [ ] **Step 6: Cross-check one real bulletin**

Take one station-day from CPCB's published daily AQI bulletin (cpcb.nic.in, AQI Bulletin PDF) for a Delhi station whose hourly data you also have from the OpenCity CSV (https://data.opencity.in/dataset/delhi-hourly-air-quality-reports). Feed its 24 hourly values for each pollutant to `daily_aqi` in a Python shell and confirm the AQI matches the bulletin within ±1 (the bulletin uses the 24 hours ending 4 pm). Record the station, date, and both values in the commit message. If it does not match, stop and investigate before committing: the mismatch is a finding, not noise.

- [ ] **Step 7: Commit**

```powershell
git add src/dhundh/health/aqi.py tests/health/test_aqi.py
git commit -m "feat(health): add CPCB national AQI with validity rules"
```

---

### Task 8: Exposure translations and GEMM

**Files:**
- Create: `src/dhundh/health/exposure.py`, `src/dhundh/health/gemm.py`
- Test: `tests/health/test_exposure.py`, `tests/health/test_gemm.py`

**Interfaces:**
- Produces:
  - `WHO_GUIDELINE = 5.0`, `NAAQS_ANNUAL = 40.0`, `AQLI_YEARS_PER_UG = 0.098`, `CIGARETTE_UG = 22.0`
  - `life_years_lost(annual_pm25: float, reference: float = WHO_GUIDELINE) -> float`
  - `cigarettes_per_day(pm25_24h_mean: float) -> float`
  - `gemm_hazard_ratio(pm25: float) -> float` — GEMM NCD+LRI, long-term exposure.

- [ ] **Step 1: Verify the GEMM coefficients against the primary source**

Open Burnett et al. 2018, PNAS 115:9592 (https://pmc.ncbi.nlm.nih.gov/articles/PMC6156628), SI Appendix, GEMM NCD+LRI parameter table (the model including the Chinese male cohort). Confirm θ = 0.1430, α = 1.6, μ = 15.5, ν = 36.8 and the counterfactual concentration 2.4 µg/m³. If any differ, update the constants in Step 4 and recompute the expected values in Step 2 with:
`.venv\Scripts\python -c "from dhundh.health.gemm import gemm_hazard_ratio as h; print([round(h(p), 6) for p in (10, 35, 88.4, 150)])"` (after Step 4).

- [ ] **Step 2: Write the failing tests**

`tests/health/test_exposure.py`:
```python
import pytest

from dhundh.health.exposure import NAAQS_ANNUAL, cigarettes_per_day, life_years_lost


def test_reproduces_aqli_2025_delhi_headline():
    # AQLI 2025: Delhi 2023 annual PM2.5 88.4 µg/m³ → 8.2 years vs WHO, 4.7 vs NAAQS.
    assert life_years_lost(88.4) == pytest.approx(8.1732)
    assert round(life_years_lost(88.4), 1) == 8.2
    assert round(life_years_lost(88.4, reference=NAAQS_ANNUAL), 1) == 4.7


def test_clean_air_costs_nothing():
    assert life_years_lost(4.0) == 0.0


def test_cigarette_equivalent():
    assert cigarettes_per_day(22.0) == pytest.approx(1.0)
    assert cigarettes_per_day(88.0) == pytest.approx(4.0)
```

`tests/health/test_gemm.py`:
```python
import pytest

from dhundh.health.gemm import gemm_hazard_ratio


def test_no_excess_risk_at_or_below_counterfactual():
    assert gemm_hazard_ratio(2.4) == 1.0
    assert gemm_hazard_ratio(0.0) == 1.0


@pytest.mark.parametrize("pm25, expected", [(10, 1.118171), (35, 1.308554), (88.4, 1.646972), (150, 1.879718)])
def test_reference_values(pm25, expected):
    assert gemm_hazard_ratio(pm25) == pytest.approx(expected, abs=1e-6)


def test_risk_rises_monotonically():
    values = [gemm_hazard_ratio(p) for p in range(0, 300, 5)]
    assert all(b >= a for a, b in zip(values, values[1:]))
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/health -k "exposure or gemm" -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement**

`src/dhundh/health/exposure.py`:
```python
"""Plain-language exposure translations.

Life expectancy: AQLI (EPIC, U. Chicago) — 0.098 years lost per µg/m³ of
sustained PM2.5 above the reference. Cigarettes: Berkeley Earth's
approximation, 22 µg/m³ over 24 h ≈ one cigarette (shown as approximate).
"""

WHO_GUIDELINE = 5.0
NAAQS_ANNUAL = 40.0
AQLI_YEARS_PER_UG = 0.098
CIGARETTE_UG = 22.0


def life_years_lost(annual_pm25: float, reference: float = WHO_GUIDELINE) -> float:
    return max(annual_pm25 - reference, 0.0) * AQLI_YEARS_PER_UG


def cigarettes_per_day(pm25_24h_mean: float) -> float:
    return pm25_24h_mean / CIGARETTE_UG
```

`src/dhundh/health/gemm.py`:
```python
"""Global Exposure Mortality Model, NCD+LRI (Burnett et al., PNAS 2018).

    HR(z) = exp( θ·ln(z/α + 1) / (1 + exp(−(z − μ)/ν)) ),  z = max(0, PM2.5 − 2.4)
"""
import math

THETA = 0.1430
ALPHA = 1.6
MU = 15.5
NU = 36.8
COUNTERFACTUAL = 2.4


def gemm_hazard_ratio(pm25: float) -> float:
    z = max(0.0, pm25 - COUNTERFACTUAL)
    return math.exp(THETA * math.log(z / ALPHA + 1) / (1 + math.exp(-(z - MU) / NU)))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/health -k "exposure or gemm" -v`
Expected: `9 passed`

- [ ] **Step 6: Commit**

```powershell
git add src/dhundh/health/exposure.py src/dhundh/health/gemm.py tests/health/test_exposure.py tests/health/test_gemm.py
git commit -m "feat(health): add AQLI life-years, cigarette equivalents, GEMM"
```

---

### Task 9: Full-suite verification

**Files:** none created.

- [ ] **Step 1: Run the whole suite**

Run: `.venv\Scripts\python -m pytest -v`
Expected: `54 passed` on the RTX 4050 machine (1 smoke + 4 grid/boundary + 5 advection + 3 diffusion + 5 solver + 4 linearity + 1 CUDA + 22 AQI + 3 exposure + 6 GEMM).

- [ ] **Step 2: Confirm the CPU-only path**

Run: `$env:CUDA_VISIBLE_DEVICES = "-1"; .venv\Scripts\python -m pytest -v; Remove-Item Env:CUDA_VISIBLE_DEVICES`
Expected: `53 passed, 1 skipped` — this is what GitHub Actions will see in Plan 5.

- [ ] **Step 3: Tag the milestone**

```powershell
git tag plan-1-foundation
```
