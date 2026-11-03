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
