# Plan 1 outcome — foundation, physics engine, health math

**Branch:** `feat/plan-1-foundation` (12 commits on `main` b0181a9) · **Tag:** `plan-1-foundation`
**Suite:** 64 passed on RTX 4050 · 63 passed + 1 skipped CPU-only · no warnings

## Measured
| Property | Result |
|---|---|
| Gaussian puff vs exact solution, 2 km / 1 km grid | 6.99% / 1.76% rel. L2 error — second-order (×3.98) |
| Puff test rejects wrong diffusivity | K halved → 4.3%, K doubled → 6.4% (bound 2.5%) |
| Mass conservation (periodic) | 4e-16 |
| Sharp front | no negatives, no overshoot |
| Source + deposition ODE | matches exact solution to 1e-6 |
| Scaling a source (what-if) | exact to 1.7e-16 |
| Batched tracers vs separate runs | bit-identical |
| Source additivity | 1.0e-9 (limiter is nonlinear; see R7) |
| CUDA float32 vs CPU float64 | 4.5e-7 |
| AQLI Delhi 2023 | reproduces published 8.2 y (WHO) / 4.7 y (NAAQS) |

## Decisions that changed or extended the plan
- **R5 / R15 — time step.** `stable_dt` sums advective, diffusive and deposition rates, and the advective rate is the sum of per-axis maxima. The plan's per-process minimum blew up when all three bound (±1.4e7); a per-cell advective rate let one stage reach −0.5·C under alternating winds. Both have regression tests.
- **R7 — additivity.** The plan's claim that additivity improves with smoother fields is wrong; deviation grows with K as plumes share limiter stencils. The product defines the total as Σ tracers and relies only on exact scaling, so no code change. Later plans must not assert tight `run(ΣE) == Σ run(E_k)` at operational K.
- **R8 — AQI breakpoints** verified by three secondary reproductions (primary CPCB PDF unreachable); Severe upper edges follow the "Severe width = Very Poor width" rule.
- **R10 — continuous AQI interpolation** (edges 0, 30, 60, …) instead of CPCB integer bands (31–60, …); differs by ≤ ~1.7 AQI points at band starts. Plan 2's bulletin cross-check decides whether to switch.
- **R11 — NaN and negative hourly values count as missing** (spec §3.3).
- **R13 — GEMM θ, α, μ, ν are UNVERIFIED** against the PNAS SI (blocked); the docstring says so.
- **R2 (revised)** — commit trailers name the model that wrote each commit.

## Open items, by owner
**Before GEMM numbers reach users (manual):** verify θ, α, μ, ν and the age group against the Burnett 2018 SI or the authors' DataGEMM calculator (github.com/mszyszkowicz/DataGEMM); soften "may overstate" to match; add a `COEFFICIENTS_VERIFIED` gate.

**Plan 2 (ingest):** CPCB bulletin cross-check of `daily_aqi` with station-level hourly concentrations (R9, and settles R10); enforce exactly 24 h per station-day; normalise `pd.NA` / `inf` to NaN before AQI.

**Plan 3 (physics on Delhi):** inflow/Dirichlet boundary with per-side values (zero_gradient recirculates at inflow edges — 2.65× mass in a probe; per-side values need axis-aware plumbing past the transpose trick); `Forcing.__post_init__` validation (finite, h > 0, K ≥ 0, shapes); `vd` as tensor; expose `cfl`/`max_dt` in `integrate`; positivity test with fronts under non-uniform wind; zero_gradient flux-budget test; nesting (spec §4.5 test 5); clamp physics output ≥ 0 before `sub_index`.

**Plan 5 (pipeline):** `torch.set_flush_denormal(True)` on CPU inference (float32 runs produce subnormals).

**Plan 6 (frontend):** vectorised `sub_index` for grid AQI; GEMM gate above.

**Deferred polish (can wait):** input validation in `pad_last`/`Grid`/exposure functions; E731 lambdas; extra unit coverage (zero_gradient advection, face-alignment ramp, y-only diffusion, AQI window 5-vs-6, band edges, tie-break documentation); linearity fixture runtime (~58 s) and global RNG use; `capped` flag uses unrounded value.
