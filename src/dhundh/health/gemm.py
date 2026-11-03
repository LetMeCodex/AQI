"""Global Exposure Mortality Model, NCD+LRI (Burnett et al., PNAS 2018).

    HR(z) = exp( θ·ln(z/α + 1) / (1 + exp(−(z − μ)/ν)) ),  z = max(0, PM2.5 − 2.4)

Verification status: the functional form and the 2.4 µg/m³ counterfactual are
confirmed from the paper. θ, α, μ and ν are NOT yet verified against the SI
Appendix (unreachable when written). The paper's NCD+LRI θ is age-specific; one
secondary source lists these values as the age 25–29 row, so this single curve
may overstate risk for older groups. Verify against the SI or the authors'
DataGEMM calculator before showing these numbers to users.

Delhi's annual means (~88 µg/m³) sit above the Chinese cohort's upper range
(~84 µg/m³), so values there are a mild extrapolation.
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
