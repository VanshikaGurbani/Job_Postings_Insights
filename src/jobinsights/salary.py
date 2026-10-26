"""Salary parsing from posting text and normalisation of the structured pay columns.

Everything is converted to annual USD so hourly, weekly and monthly pay are comparable.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

ANNUAL_FACTOR = {
    "hourly": 2080,
    "daily": 260,
    "weekly": 52,
    "biweekly": 26,
    "monthly": 12,
    "yearly": 1,
}
ANNUAL_MIN, ANNUAL_MAX = 10_000.0, 1_500_000.0

_NUM = r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?\s?(k\b)?"
_SECOND = r"\$?\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?\s?(k\b)?"
_RANGE = re.compile(
    rf"{_NUM}(?:\s*(?:/\s*(?:yr|hr|year|hour)\.?)?\s*(?:-|–|—|�|to|and)\s*{_SECOND})?",
    re.IGNORECASE,
)
_UNIT = re.compile(
    r"(?:per|/|a|an|each|\bp\.?\s?)\s*(hour|hr|year|yr|annum|month|mo|week|wk|day)\b"
    r"|\b(hourly|annually|annual|yearly|monthly|weekly|bi-?weekly)\b",
    re.IGNORECASE,
)
_UNIT_MAP = {
    "hour": "hourly", "hr": "hourly", "hourly": "hourly",
    "year": "yearly", "yr": "yearly", "annum": "yearly", "annually": "yearly",
    "annual": "yearly", "yearly": "yearly",
    "month": "monthly", "mo": "monthly", "monthly": "monthly",
    "week": "weekly", "wk": "weekly", "weekly": "weekly",
    "day": "daily", "biweekly": "biweekly", "bi-weekly": "biweekly",
}  # fmt: skip
# Dollar amounts near these words are not the base pay.
_NOT_BASE = re.compile(
    r"bonus|commission|average|earnings|sign[\s-]?on|signing|referral|stipend|reimburs|tuition|401|match|"
    r"per\s+mile|mileage|revenue|funding|raised|budget|million|billion|"
    r"insurance|premium|deductible|allowance|\bOTE\b|equity|stock|shift\s+differential|"
    r"(?:per|an|each)\s+(?:visit|patient|unit|piece|delivery|referral|load|stop)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Pay:
    low: float
    high: float
    period: str  # hourly / yearly / ...
    source: str  # "text" or "structured"

    @property
    def annual_low(self) -> float:
        return self.low * ANNUAL_FACTOR[self.period]

    @property
    def annual_high(self) -> float:
        return self.high * ANNUAL_FACTOR[self.period]

    @property
    def annual_mid(self) -> float:
        return (self.annual_low + self.annual_high) / 2


def _amount(whole: str, cents: str | None, k: str | None) -> float:
    value = float(whole.replace(",", "") + (f".{cents}" if cents else ""))
    return value * 1000 if k else value


def infer_period(low: float, high: float) -> str | None:
    """Period from magnitude when none is stated: wage-like vs salary-like numbers."""
    if 7 <= low and high <= 400:
        return "hourly"
    if low >= 20_000:
        return "yearly"
    return None


def _plausible(low: float, high: float, period: str) -> bool:
    f = ANNUAL_FACTOR[period]
    return high >= low and ANNUAL_MIN <= low * f and high * f <= ANNUAL_MAX


def parse_salary_text(text: str) -> Pay | None:
    """First plausible base-pay range in `text`, or None.

    A match with an explicit period wins over one whose period is only inferred.
    """
    inferred: Pay | None = None
    for m in _RANGE.finditer(text):
        window_before = text[max(0, m.start() - 40) : m.start()]
        window_after = text[m.end() : m.end() + 40]
        if _NOT_BASE.search(window_before[-25:]) or _NOT_BASE.search(window_after[:25]):
            continue
        low = _amount(m.group(1), m.group(2), m.group(3))
        high = _amount(m.group(4), m.group(5), m.group(6)) if m.group(4) else low
        if m.group(4) and m.group(6) and not m.group(3):
            low *= 1000 if low < 1000 else 1  # "$85-120K" means both are in thousands
        if high < low:
            continue
        unit = _UNIT.search(window_after[:30]) or _UNIT.search(window_before[-25:])
        period = None
        if unit:
            period = _UNIT_MAP.get((unit.group(1) or unit.group(2)).lower().replace(" ", ""))
        if period and _plausible(low, high, period):
            return Pay(low, high, period, "text")
        if period is None and inferred is None:
            guess = infer_period(low, high)
            if guess and _plausible(low, high, guess):
                inferred = Pay(low, high, guess, "text")
    return inferred


def normalize_structured(
    min_salary: float | None,
    max_salary: float | None,
    pay_period: str | None,
    med_salary: float | None = None,
) -> Pay | None:
    """Clean the platform's pay columns.

    The columns contain mislabelled periods (e.g. an annual figure tagged HOURLY), so the
    stated period is trusted only when the magnitude is consistent with it.
    """
    lo = min_salary if _present(min_salary) else med_salary
    hi = max_salary if _present(max_salary) else med_salary
    if not _present(lo) or not _present(hi):
        return None
    lo, hi = float(lo), float(hi)
    period = (pay_period or "").strip().lower()
    period = {"yearly": "yearly", "hourly": "hourly", "monthly": "monthly", "weekly": "weekly"}.get(
        period
    )
    candidates = [period] if period else []
    candidates += [p for p in ("hourly", "yearly") if p != period]
    for p in candidates:
        if p == "hourly" and not (7 <= lo and hi <= 400):
            continue
        if p == "yearly" and lo < 10_000:
            continue
        if p in ("monthly", "weekly") and not _plausible(lo, hi, p):
            continue
        if _plausible(lo, hi, p):
            return Pay(lo, hi, p, "structured")
    return None


def _present(x: object) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))
