# Dhundh (धुंध) — Design Spec

**Date:** 2026-11-03
**Status:** Approved in brainstorming, pending written-spec review
**One line:** A free, public, physics-grounded air-quality tool for Delhi NCR that forecasts PM2.5 for 72 hours, explains where the pollution comes from, translates it into health terms, and lets anyone test policy interventions — with a visual identity drawn from Delhi's own sky.

---

## 1. Goals and non-goals

### Goals
1. **Forecast** hourly PM2.5 on a 1 km grid over Delhi NCR, 72 hours ahead, with calibrated uncertainty. PM10 and NO₂ are forecast at stations only (extra corrector output heads), so the full multi-pollutant AQI is shown at stations and the PM2.5 sub-index everywhere on the grid.
2. **Attribute** PM2.5 at any point to 8 source sectors using physical transport, not statistical guesswork.
3. **Translate** exposure into health terms using published, verifiable functions (CPCB AQI, AQLI, GEMM).
4. **Simulate** policy interventions instantly in the browser.
5. **Prove** accuracy publicly: sealed test years, published baselines, a public model card including failures.
6. **Run free**: ₹0 hosting, hourly refresh, robust to traffic spikes on smog days.

### Non-goals (v1)
- No user accounts, notifications backend, or personal data storage.
- No indoor air modelling, mask-efficacy modelling, or route planning.
- No full chemical transport model (WRF-Chem/CAMx); CAMS stands in as an input.
- No fine-tuning of Aurora (1.3B params exceeds the 6 GB training GPU); Aurora/CAMS used only as inputs and baselines.
- Regions outside Delhi NCR (Punjab/Haryana are modelled only as upwind source regions).

### Honesty principle
"Perfect accuracy" is not claimed anywhere. The product claims what it can prove: physics-constrained, validated on held-out years, benchmarked, uncertainty always shown.

---

## 2. System architecture

One repository, five units, each with a single job and a file-based interface to the next.

```
ingest/  ──Parquet/Zarr──▶  physics/  ──tracer fields──▶  model/  ──corrected fields──▶  pipeline/publish  ──static files──▶  web/
```

| Unit | Job | Inputs | Outputs |
|---|---|---|---|
| `ingest/` | Fetch, clean, QC, store | External APIs/archives | Parquet (stations), Zarr (gridded) |
| `physics/` | Transport emissions → concentrations per sector | Met fields, emissions, fires | 8 tracer fields × 72 h |
| `model/` | Learn and apply residual correction + uncertainty | Station obs, tracer fields, CAMS, met | Corrected quantile fields |
| `pipeline/` | Hourly orchestration on GitHub Actions | All of the above | Published bundle on Cloudflare R2 |
| `web/` | Static frontend | Published bundle | The product |

### 2.1 Hosting and compute
- **Training:** local RTX 4050 (6 GB VRAM), 24 GB RAM, PyTorch + CUDA, mixed precision.
- **Inference:** ONNX Runtime on GitHub Actions CPU runners, hourly cron.
- **Storage:** Cloudflare R2 (forecast bundles), Cloudflare Pages (static site).
- **Secrets:** OpenAQ API key, NASA FIRMS MAP_KEY, R2 credentials — GitHub Secrets only.

---

## 3. Data sources

### 3.1 Historical (training/backfill)
| Data | Source | Use |
|---|---|---|
| Hourly station PM2.5/PM10/NO₂, Delhi 2017–2023 | OpenCity CSVs (CPCB-derived, 39 stations) | Ground truth |
| CPCB India-wide hourly 2015–2019 | Princeton CPCB archive (Globus) | Gap filling, NCR stations |
| 2024–2025 station data | OpenAQ v3 (CPCB regulatory network) | Sealed test set |
| Wind u/v, T, RH, rain, boundary-layer height | ERA5 (Copernicus CDS) | Physics driver, features |
| Active fires + fire radiative power | NASA FIRMS VIIRS 375 m (science-quality archive) | Fire emissions |
| Gridded emissions, 8 pollutants, ~0.4 km, base 2020 | ESSD 15, 3183 (2023) Delhi NCR inventory | Physics source term |
| Tropospheric NO₂ column | Sentinel-5P TROPOMI via Earth Engine | Feature; inventory sanity check |
| CAMS reanalysis/forecast PM2.5 | Copernicus ADS / Open-Meteo | Feature; baseline |

### 3.2 Live (hourly pipeline)
| Data | Source |
|---|---|
| Station observations | OpenAQ v3 API |
| Forecast winds, BLH, met (72 h) | Open-Meteo (ECMWF/GFS-based) |
| CAMS PM2.5 forecast | Open-Meteo air-quality endpoint |
| Fires (NRT, ~3 h latency) | NASA FIRMS API |

### 3.3 Data quality rules
- Drop negatives, flat-lined sensors (identical value ≥ 6 h), and values > 1,500 µg/m³ unless corroborated by ≥ 2 neighbouring stations.
- FIRMS: NRT for live, science-quality archive for training; VIIRS false positives over industrial sites filtered with a static persistent-heat-source mask.
- Missing station-hours are **masked**, never imputed with fabricated values, in both training and inference.
- Every ingest run writes a QC report (counts dropped per rule per station).

---

## 4. Physics engine (`physics/`)

### 4.1 Governing equation
Column-integrated (well-mixed boundary layer) advection–diffusion for each tracer *k*:

Boundary-layer entrainment is omitted in v1 (documented in §11):

```
∂C_k/∂t + ∇·(u C_k) = ∇·(K_h ∇C_k) + E_k(x,y,t) / h(x,y,t) − (v_d / h) · C_k
```

Solved in conservative flux form, so mass is conserved exactly by construction (equal to the advective form `u·∇C` wherever the flow is divergence-free).

- `C_k` — concentration of tracer k (µg/m³)
- `u` — boundary-layer-mean horizontal wind (m/s)
- `K_h` — horizontal eddy diffusivity (m²/s), Smagorinsky-style from wind deformation, floored
- `E_k` — emission flux (µg/m²/s) for sector k
- `h` — boundary-layer height (m); low `h` = stronger concentration ("the lid")
- `v_d` — dry deposition velocity (m/s), size-class constant for PM2.5

### 4.2 Tracers (8)
transport · industry · power · residential · dust · waste burning · stubble fires · background/boundary inflow.
Fire emissions: `E_fire = FRP × CE` where CE is a published FRE-to-PM2.5 emission coefficient (Kaiser et al./GFAS lineage), verified from primary source at implementation.

### 4.3 Numerics
- Finite-volume, flux-limited advection (monotone limiter, e.g. van Leer/MC) → **exact mass conservation, positivity-preserving**.
- Explicit diffusion; timestep from CFL and diffusion stability limits each step.
- Two nested grids, one-way: **4 km regional** (Punjab–Haryana–Delhi) feeds boundary conditions of **1 km Delhi NCR**.
- PyTorch implementation (same code GPU for backfill, CPU for live).

### 4.4 Linearity (why attribution and what-if work)
The equation is linear in `E_k`. Total primary PM2.5 = Σ_k C_k exactly. Scaling sector k by α scales its field by α. Secondary aerosol (nonlinear chemistry) is carried by the ML residual and is labelled **approximate** wherever scenarios touch it.

### 4.5 Verification tests
1. Gaussian puff under constant `u`, `K_h`: numerical vs analytic solution, error converging at expected order with grid refinement.
2. Mass conservation in closed domain to ~machine precision.
3. Positivity: no negative concentrations under sharp fronts.
4. Linearity: field(αE) == α·field(E) within float tolerance.
5. Nesting: mass flux continuity across the regional → city boundary.

---

## 5. ML corrector (`model/`)

### 5.1 Target
Residual `r = observed − physics_total` for PM2.5 at each station, for each lead hour 1…72. Two additional heads forecast PM10 and NO₂ directly at stations (no physics prior for those in v1).

### 5.2 Graph
- Nodes: ~80 stations (Delhi, Noida, Gurugram, Faridabad, Ghaziabad).
- Edges: stations within a distance threshold; **dynamic wind-aware weights** (edge A→B strengthens when the wind projection onto A→B is positive), following AirPhyNet.

### 5.3 Features per node per hour
Past 72 h PM2.5/PM10/NO₂ · physics total and 8 tracers · raw CAMS forecast · u, v, T, RH, BLH, rain · upwind FRP aggregate · daily TROPOMI NO₂ · hour, weekday, Diwali flag, GRAP stage.

### 5.4 Architecture
GRU encoder → graph-ODE block whose operator is graph advection + diffusion → decoder emitting quantiles (q10, q50, q90), trained with pinball loss.
Spatial step: sparse variational Gaussian process (GPyTorch) interpolates station residuals to the 1 km grid; GP variance combined with quantile spread for grid uncertainty.

### 5.5 Training
- PyTorch + CUDA, AMP, on RTX 4050; Optuna hyperparameter search.
- Export to ONNX; parity test asserts ONNX vs PyTorch outputs match within tolerance.

### 5.6 Validation protocol
| Split | Years | Notes |
|---|---|---|
| Train | 2017–2022 | Includes 2020 lockdown |
| Validate | 2023 | Tuning only |
| Test (sealed) | 2024–2025 | Opened once, at the end |

Normalisation statistics from training years only.

**Metrics:** MAE, RMSE, R² at leads 1/6/24/48/72 h, per season; CRPS; interval coverage (80% band must cover ≈ 80%).
**Baselines to beat:** persistence, raw CAMS, physics-only, published STGCN on Delhi (MAE 10.53 @ 1 h, 20.18 @ 24 h; Jana et al. 2024).
**Physical sanity checks:**
- 2020 lockdown: transport tracer must drop consistent with observed NO₂ fall.
- Stubble share vs literature: IIT-K/TERI ~22% (1–25 Nov avg), IITM AQEWS 30–35% peak daily mean (Oct–Nov 2021). Literature spans 4–42%; our estimates must sit within published ranges and always carry uncertainty.

**Model card** published on the site with all metrics, including where the model fails.

---

## 6. Health and AQI (`model/health` — pure functions, fully unit-tested)

| Output | Method | Verification |
|---|---|---|
| India AQI | CPCB piecewise-linear sub-indices; max sub-index; 24 h avg for PM/NO₂/SO₂/NH₃, 8 h max for CO/O₃; validity: ≥3 pollutants incl. PM, ≥16 h data | Breakpoints from CPCB National AQI document; tests against published CPCB daily bulletins |
| Life expectancy loss | AQLI: 0.098 years per µg/m³ above WHO 5 µg/m³ | Reproduces AQLI 2025 Delhi: (88.4−5)×0.098 = 8.17 ≈ 8.2 y; vs NAAQS 40: 4.74 ≈ 4.7 y |
| Long-term mortality risk | GEMM hazard ratio (Burnett et al., PNAS 2018), coefficients from SI tables | Reproduce a published GEMM example value |
| Short-term daily risk | Indian multi-city time-series coefficient | Coefficient verified from primary paper at implementation |
| Cigarette equivalent | ~22 µg/m³ PM2.5 over 24 h ≈ 1 cigarette (Berkeley Earth) | Labelled approximate |
| Best time outside | Lowest forecast q50 window in waking hours, next 24 h, shown with band | — |

AQI shown with raw µg/m³ always (AQI caps at 500). Standing notice: information, not medical advice.

---

## 7. Hourly pipeline (`pipeline/`)

GitHub Actions cron, hourly:
1. Ingest live data → QC.
2. Run physics: regional + city grids, 72 h, 8 tracers.
3. Run ONNX corrector + GP → quantile fields.
4. Compute back-trajectories for a fixed lattice of receptor points (for "trace").
5. Package bundle: per-hour Float16 tiles (~100 KB each), station JSON, manifest with run time and data-freshness per source.
6. Upload to R2; swap manifest pointer atomically last.

**Failure behaviour**
- Station offline → masked node; manifest lists it.
- A source stale → run proceeds; manifest marks staleness; UI shows "data as of".
- Run fails → manifest not swapped; site keeps last good run and shows its age.

---

## 8. Frontend (`web/`)

### 8.1 Stack
Svelte 5 + TypeScript + Vite · MapLibre GL + Protomaps vector tiles (custom style) · custom WebGL2 layer (particles + field LUT) · KaTeX · PWA · static on Cloudflare Pages.

### 8.2 Visual identity — "Dhundh, with chronograph precision"
- **Palette = data.** The concentration scale is sampled from photographs of Delhi's sky at each AQI band: clean blue-white → sulphur → ochre → rust → soot. Provisional stops: `#E4ECEF #E6DCC0 #D8B475 #BE8546 #8C4A2A #4A2E24`, final values sampled from real photographs and verified in OKLab for monotonic lightness.
- **Surfaces:** Makrana-marble warm whites; jali (Mughal lattice) geometry as structural pattern, never decoration for its own sake.
- **The smog-sun:** a pale disc whose visibility tracks current PM2.5.
- **Type:** Cormorant Garamond (display, numerals) · Tiro Devanagari Hindi (Hindi) · Martian Mono (readings, labels).
- **Chronograph:** 72-tick dial as the time scrubber; ring thickness = uncertainty; mono numerics.
- **Restraint:** one accent reserved for danger.
- **Night variant:** dusk indigo with sodium-lamp orange, following device theme.
- **Bilingual:** Hindi / English throughout.

### 8.3 Screens
1. **Air now** — full-bleed map, particles at true forecast wind speed, smog-sun, reading at your location (µg/m³ + AQI).
2. **Chronograph scrubber** — 72 h, uncertainty ring, best-time-outside window.
3. **What's in my air** — 8-sector share ring with uncertainty; "Trace" animates back-trajectories upwind.
4. **Health** — life years, cigarettes, sensitive-group guidance.
5. **Policy lab** — per-sector sliders; GPU weighted sum of tracer fields, instant; secondary aerosol marked approximate.
6. **Methods & model card** — equations, sources, metrics, failures.

### 8.4 Quality bars
- Colour never encodes meaning alone; numbers always present.
- WCAG AA contrast on all text; palette legible in greyscale and for common colour-vision deficiencies.
- 60 fps target on mid-range Android; static-image fallback without WebGL2.
- Mobile-first layouts at 360 px width with 16 px gutters.

---

## 9. Testing strategy
- **Physics:** analytic, conservation, positivity, linearity, nesting tests (§4.5).
- **Health/AQI:** table-driven tests against CPCB bulletins and AQLI/GEMM published values.
- **ML:** time-split discipline test (assert no test-year rows in training loader), ONNX parity, calibration coverage report.
- **Pipeline:** dry-run with recorded API fixtures; failure-injection (stale source, dead station, failed upload).
- **Frontend:** unit tests for data decoding/scenario math (must equal the Python reference); visual smoke tests on phone and desktop widths.

---

## 10. Build order (for the implementation plan)
1. Repo scaffold, environment (CUDA PyTorch), data contracts.
2. `ingest/` historical backfill + QC.
3. `physics/` solver + verification tests.
4. `model/health` pure functions + tests.
5. Physics backfill 2017–2025 on GPU.
6. `model/` corrector training, validation, sealed test, model card.
7. `pipeline/` hourly job + R2 publishing.
8. `web/` design system (palette, type, jali, dial), then screens 1→6.
9. End-to-end launch checks.

---

## 11. Known limitations (shown publicly)
- Boundary-layer entrainment omitted in v1.
- ERA5/forecast BLH is weakest in very stable nights — exactly Delhi's worst winter nights.
- Emission inventory base year 2020; sectors drift over time (the ML residual partly absorbs this).
- Secondary aerosol attribution is approximate.
- Uncertainty grows away from stations; shown, not hidden.

---

## 12. Sources
- OpenCity Delhi hourly station data: https://data.opencity.in/dataset/delhi-hourly-air-quality-reports
- Princeton CPCB 2015–2019: https://datacommons.princeton.edu/discovery/catalog/doi-10-34770-60j3-yp02
- OpenAQ v3: https://docs.openaq.org/using-the-api/quick-start
- ESSD Delhi NCR emissions inventory: https://essd.copernicus.org/articles/15/3183/2023/
- TROPOMI NO₂ in Earth Engine: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_NRTI_L3_NO2
- ERA5 BLH evaluation: https://amt.copernicus.org/articles/15/3075/2022/
- FIRMS/VIIRS over Punjab–Haryana: https://www.downtoearth.org.in/pollution/counting-north-india-s-stubble-fires-how-do-nasa-satellites-capture-them-73695
- Aurora: https://microsoft.com/en-us/research/publication/aurora-a-foundation-model-of-the-atmosphere
- STGCN Delhi (Jana et al. 2024): https://ideas.repec.org/a/eee/tefoso/v208y2024ics0040162524004827.html
- STEEP / DCRNN Delhi (2025): https://journals.sagepub.com/doi/10.3233/IDA-240028
- AirPhyNet (ICLR 2024): https://www.iclr.cc/media/iclr-2024/Slides/18940.pdf
- PIGNN-GPR Delhi: https://talks.cam.ac.uk/talk/index/232567
- FLEXPART stubble transport (Takigawa et al. 2020): https://www.jstage.jst.go.jp/article/sola/16/0/16_2020-015/_article
- HYSPLIT stubble (Kundu & Hooda 2024): https://link.springer.com/content/pdf/10.1007/s11270-024-07062-z.pdf
- Stubble share, IITM AQEWS: https://pmc.ncbi.nlm.nih.gov/articles/PMC10275965
- Stubble share estimates overview: https://scroll.in/article/1011056/what-is-stubble-burnings-contribution-to-delhis-pollution-it-depends-on-which-data-you-look-at
- IITM DSS: https://moes.gov.in/sites/default/files/PIB1808616.pdf
- GEMM (Burnett et al. 2018): https://pmc.ncbi.nlm.nih.gov/articles/PMC6156628
- AQLI 2025 Delhi: https://aqli.epic.uchicago.edu/post/delhi-residents-losing-8-2-years-of-their-lives-to-high-air-pollution-report-en
- CPCB AQI report: https://theasthmafiles.org/sites/default/files/artifacts/media/pdf/final_report_of_cpcb_on_air_quality_index_2014-15.pdf
- WebGL wind particles: https://blog.mapbox.com/how-i-built-a-wind-map-with-webgl-b63022b5537f
