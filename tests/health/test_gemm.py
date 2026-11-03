import pytest

from dhundh.health.gemm import gemm_hazard_ratio


def test_no_excess_risk_at_or_below_counterfactual():
    assert gemm_hazard_ratio(2.4) == 1.0
    assert gemm_hazard_ratio(0.0) == 1.0


# Self-consistency only: expected values were computed from these same constants.
@pytest.mark.parametrize("pm25, expected", [(10, 1.118171), (35, 1.308554), (88.4, 1.646972), (150, 1.879718)])
def test_reference_values(pm25, expected):
    assert gemm_hazard_ratio(pm25) == pytest.approx(expected, abs=1e-6)


def test_risk_rises_monotonically():
    values = [gemm_hazard_ratio(p) for p in range(0, 300, 5)]
    assert all(b >= a for a, b in zip(values, values[1:]))
