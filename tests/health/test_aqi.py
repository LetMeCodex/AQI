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


def test_sub_index_rejects_nan():
    with pytest.raises(ValueError):
        sub_index("pm25", float("nan"))


def test_nan_hours_count_as_missing():
    nan = float("nan")
    assert daily_concentration("pm25", [nan] * 9 + [100.0] * 15) is None
    assert daily_concentration("pm25", [nan] * 8 + [100.0] * 16) == 100.0


def test_negative_hours_count_as_missing():
    assert daily_concentration("pm25", [-5.0] + [100.0] * 23) == 100.0
    result = daily_aqi({"pm25": [-5.0] + [100.0] * 23, "pm10": [200.0] * 24, "no2": [50.0] * 24})
    assert result.aqi == 233


def test_half_up_rounding_ignores_float_noise():
    # The 24-hour mean of 30.3 picks up float noise, so its sub-index is 50.4999..., not 50.5.
    assert sub_index("pm25", daily_concentration("pm25", [30 * 1.01] * 24)) != 50.5
    result = daily_aqi({"pm25": [30 * 1.01] * 24, "pm10": [10.0] * 24, "no2": [10.0] * 24})
    assert result.aqi == 51
