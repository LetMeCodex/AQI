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
