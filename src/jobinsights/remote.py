"""Rule-based work-arrangement detection: remote / hybrid / onsite / unknown."""

from __future__ import annotations

import re

LABELS = ("remote", "hybrid", "onsite", "unknown")

_I = re.IGNORECASE

# "remote" in the sense of the job location, not "remote sensing" or "remote clients".
_REMOTE_NOUN = (
    r"control|sensing|access|monitoring|areas?|locations?|clients?|customers?|teams?|"
    r"employees|users?|servers?|sites?|support|troubleshooting|diagnos\w+|desktop|learners?|"
    r"communit\w+|villages?|patients?|workers?|management|control"
)
_NEGATED_REMOTE = re.compile(
    r"\b(?:not|no|isn'?t|is not|cannot be|non)[\s-]+(?:a\s+|an\s+|fully\s+)?"
    r"(?:remote|telecommut\w+|work[\s-]from[\s-]home)",
    _I,
)
_HYBRID = re.compile(
    r"\bhybrid\b(?!\s+(?:cloud|vehicles?|cars?|engines?|batter\w+|electric|seeds?|technolog\w+))",
    _I,
)
_REMOTE_STRONG = re.compile(
    r"(?:\bfully[\s-]+remote\b|\b100\s?%\s*remote\b|\bremote[\s-]+first\b|\bremote[\s-]+only\b"
    r"|\bremote\s+(?:position|role|job|opportunity|work|employment|based|eligible|friendly)\b"
    r"|\bthis\s+(?:position|role|job)\s+is\s+(?:\w+\s+)?remote\b"
    r"|\b(?:work|working)\s+(?:remotely|from\s+home|from\s+anywhere|at\s+home)\b"
    r"|\bwork[\s-]from[\s-]home\b|\bwfh\b|\btelecommut\w+\b|\btelework\w*\b"
    r"|\bvirtual\s+(?:position|role|job)\b|\bhome[\s-]based\b"
    r"|\blocation\s*:\s*remote\b|\bremote\s*\((?:us|usa|united states)\)"
    r"|\b(?:can|may|able\s+to)\s+(?:be\s+)?(?:work(?:ed)?|perform(?:ed)?)\s+remotely\b"
    r"|\bremote\s+(?:within|in)\s+the\s+(?:us|united states)\b)",
    _I,
)
_REMOTE_WEAK = re.compile(rf"\bremote\b(?!\s+(?:{_REMOTE_NOUN})\b)", _I)
_ONSITE = re.compile(
    r"\bon[\s-]?site\b|\bin[\s-]+(?:the[\s-]+)?office\b|\bin[\s-]person\b|\bat\s+our\s+(?:office|facility|location)s?\b"
    r"|\bmust\s+(?:be\s+)?(?:able\s+to\s+)?(?:work|report)\s+(?:on[\s-]?site|in[\s-]person)\b",
    _I,
)


def detect_remote(text: str) -> str:
    """Classify the posting text. Hybrid beats remote beats onsite."""
    negated = bool(_NEGATED_REMOTE.search(text))
    if _HYBRID.search(text):
        return "hybrid"
    if not negated and (_REMOTE_STRONG.search(text) or _REMOTE_WEAK.search(text)):
        return "remote"
    if negated or _ONSITE.search(text):
        return "onsite"
    return "unknown"


def legacy_remote(text: str) -> str:
    """The original notebook rule: any of three substrings means 'Yes'."""
    low = text.lower()
    return "Yes" if ("remote" in low or "work from home" in low or "hybrid" in low) else "No"
