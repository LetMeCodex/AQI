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
