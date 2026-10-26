"""Rule-based work-arrangement detection: remote / hybrid / onsite / unknown.

Posting text often has its line breaks stripped, so section headers run into the previous word
("commissionRemote work", "INHybrid"). Word starts are therefore matched with ``(?<![a-z])``
instead of ``\\b``, which still rejects the middle of an ordinary lower-case word.
"""

from __future__ import annotations

import re

LABELS = ("remote", "hybrid", "onsite", "unknown")

_I = re.IGNORECASE
# Start of a word, tolerant of run-together text: not after a lower-case letter, or a capital
# right after one ("commissionRemote").
_S = r"(?:(?<![a-z])|(?-i:(?<=[a-z])(?=[A-Z])))"

# "remote" in the sense of the job location, not "remote sensing" or "remote clients".
_REMOTE_NOUN = (
    r"control|sensing|access|monitoring|areas?|locations?|clients?|customers?|teams?|"
    r"employees|users?|servers?|sites?|support|troubleshooting|diagnos\w+|desktop|learners?|"
    r"communit\w+|villages?|patients?|workers?|management|control"
)
_NEGATED_REMOTE = re.compile(
    rf"{_S}(?:not|no|isn'?t|is not|cannot be|non)[\s-]+(?:a\s+|an\s+|fully\s+)?"
    r"(?:remote|telecommut\w+|work[\s-]from[\s-]home)",
    _I,
)
# "Opportunity for hybrid schedule: No" is a refusal, not hybrid work.
_HYBRID_REFUSED = re.compile(r"hybrid\s+(?:schedule|work|option|opportunity)s?\s*:\s*no\b", _I)
_HYBRID = re.compile(
    rf"{_S}hybrid\b(?!\s+(?:cloud|vehicles?|cars?|engines?|batter\w+|electric|seeds?|technolog\w+))"
    r"|\b\d(?:\s*-\s*\d)?\s+days?\s+(?:per\s+week\s+|a\s+week\s+)?(?:on-?site|in[\s-]+(?:the\s+)?office)",
    _I,
)
_REMOTE_STRONG = re.compile(
    rf"(?:{_S}fully[\s-]+remote\b|\b100\s?%\s*remote\b|{_S}remote[\s-]+first\b|{_S}remote[\s-]+only\b"
    rf"|{_S}remote\s+(?:position|role|job|opportunity|work|employment|based|eligible|friendly)\b"
    r"|\bthis\s+(?:position|role|job)\s+is\s+(?:\w+\s+)?remote\b"
    r"|\b(?:work|working)\s+(?:remotely|from\s+home|from\s+anywhere|at\s+home)\b"
    r"|\bwork[\s-]from[\s-]home\b|\bwfh\b|\btelecommut\w+\b|\btelework\w*\b"
    r"|\bvirtual\s+(?:position|role|job)\b|\bhome[\s-]based\b"
    rf"|\blocation\s*:\s*remote\b|{_S}remote\s*\((?:us|usa|united states)\)"
    r"|\b(?:can|may|able\s+to)\s+(?:be\s+)?(?:work(?:ed)?|perform(?:ed)?)\s+remotely\b"
    rf"|{_S}remote\s+(?:within|in)\s+the\s+(?:us|united states))",
    _I,
)
_REMOTE_WEAK = re.compile(rf"{_S}remote\b(?!\s+(?:{_REMOTE_NOUN})\b)", _I)
# "in person" alone also describes meetings, so it only counts when it describes the role.
_ONSITE = re.compile(
    r"\bon[\s-]?site\b|\bin[\s-]+(?:the[\s-]+)?office\b"
    r"|\bin[\s-]person\s+(?:role|position|job|work|only)\b"
    r"|\b(?:at|in)\s+our\s+(?:\w+\s+){0,3}(?:office|facility|location|plant|store)s?\b"
    r"|\bmust\s+(?:be\s+)?(?:able\s+to\s+)?(?:work|report)\s+(?:on[\s-]?site|in[\s-]person)\b",
    _I,
)


def detect_remote(text: str) -> str:
    """Classify the posting text. Hybrid beats remote beats onsite."""
    negated = bool(_NEGATED_REMOTE.search(text))
    if _HYBRID.search(_HYBRID_REFUSED.sub(" ", text)):
        return "hybrid"
    if not negated and (_REMOTE_STRONG.search(text) or _REMOTE_WEAK.search(text)):
        return "remote"
    if negated or _ONSITE.search(text) or _HYBRID_REFUSED.search(text):
        return "onsite"
    return "unknown"


def legacy_remote(text: str) -> str:
    """The original notebook rule: any of three substrings means 'Yes'."""
    low = text.lower()
    return "Yes" if ("remote" in low or "work from home" in low or "hybrid" in low) else "No"
