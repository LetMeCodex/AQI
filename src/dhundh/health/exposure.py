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
