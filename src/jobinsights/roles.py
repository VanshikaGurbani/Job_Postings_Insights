"""Role family and US state from the posting title / location."""

from __future__ import annotations

import re

# First match wins, so the specific families come before the generic ones.
ROLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "AI/ML Engineer",
        re.compile(
            r"machine learning|\bml\b|\bai\b|artificial intelligence|deep learning|\bnlp\b|"
            r"computer vision|\bllm\b|generative",
            re.IGNORECASE,
        ),
    ),
    ("Data Scientist", re.compile(r"data scien|research scientist|applied scien", re.IGNORECASE)),
    (
        "Data Engineer",
        re.compile(r"data engineer|etl|analytics engineer|big data|data architect", re.I),
    ),
    (
        "Data Analyst",
        re.compile(
            r"data analy|business analy|bi analy|\bbi developer|analytics|reporting analy|"
            r"business intelligence|insights? analy",
            re.IGNORECASE,
        ),
    ),
    (
        "Software Engineer",
        re.compile(
            r"software|developer|full[\s-]?stack|front[\s-]?end|back[\s-]?end|devops|"
            r"\bsde\b|web developer|programmer|site reliability|cloud engineer|"
            r"application engineer",
            re.IGNORECASE,
        ),
    ),
)
_STATE = re.compile(r",\s*([A-Z]{2})\s*$")
US_STATES = frozenset(
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ "
    "NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split()
)


def role_family(title: str) -> str:
    for name, rx in ROLE_PATTERNS:
        if rx.search(title):
            return name
    return "Other"


def state_of(location: str) -> str | None:
    m = _STATE.search(location or "")
    return m.group(1) if m and m.group(1) in US_STATES else None
