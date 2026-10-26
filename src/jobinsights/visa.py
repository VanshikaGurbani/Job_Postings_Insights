"""Work-authorisation and contract-style signals from the posting text.

The target reader is a candidate on F-1 OPT / STEM OPT, so the questions are: does the posting
offer sponsorship or welcome OPT/visa holders, or does it exclude them?

visa: friendly   - sponsorship offered, or OPT / H-1B / EAD / all visa types explicitly welcomed
      restricted - no sponsorship, or citizen / green-card / clearance only
      mixed      - both kinds of statement appear (read the text)
      not_stated - nothing about work authorisation

mentions_opt flags OPT / STEM OPT / F-1 appearing at all. A mention can be a restriction
("OPT not accepted"), so it is a prompt to read the posting, not a green light.
contract_like flags staffing / contract postings (W2, C2C, 1099, "our client", contract-to-hire).
"""

from __future__ import annotations

import re

VISA_LABELS = ("friendly", "restricted", "mixed", "not_stated")
_I = re.IGNORECASE

_RESTRICTED = re.compile(
    r"sponsorship\s+(?:available|offered|provided)\s*[:\-]\s*no\b"
    r"|\busc\s+only\b"
    r"|(?:no|not|without|unable to|cannot|can't|won't|will not|does not|do not|doesn't|don't|"
    r"not able to|not eligible for|ineligible for)\s+(?:\w+\s+){0,3}?"
    r"(?:visa\s+)?sponsor\w*"
    r"|sponsorship\s+(?:is\s+)?(?:not|unavailable)\s+(?:available|offered|provided)"
    r"|(?:u\.?s\.?|united states)\s+citizen(?:ship)?\s+(?:is\s+)?(?:only|required|needed)"
    r"|must\s+be\s+(?:a\s+)?(?:u\.?s\.?|united states)\s+citizen"
    r"|citizens?\s+(?:only|or\s+(?:permanent\s+)?(?:residents?|green\s*card))"
    r"|(?:gc|green\s*card)\s*(?:/|or|and)\s*(?:usc|citizen)"
    r"|(?:active|current)\s+(?:top\s+secret|secret|ts/sci)\s+(?:security\s+)?clearance"
    r"|must\s+(?:currently\s+)?(?:be\s+)?(?:authorized|eligible|legally\s+authorized)\s+to\s+work"
    r"[^.]{0,60}without\s+(?:\w+\s+){0,2}sponsorship"
    r"|without\s+(?:the\s+need\s+for\s+)?(?:current\s+or\s+future\s+)?(?:visa\s+)?sponsorship",
    _I,
)
_FRIENDLY = re.compile(
    r"(?<!not )(?<!n't )\b(?:will|can|may|able to|happy to|open to|willing to)\s+"
    r"(?:(?!not\b|never\b|no\b|require\b|need\b)\w+\s+){0,2}?sponsor\w*"
    r"|sponsorship\s+(?:is\s+)?(?:available|offered|provided|possible)(?!\s*[:\-]\s*no\b)"
    r"|(?<!not )(?<!n't )(?<!never )(?<!no )\b(?:offer|provide)s?\s+(?:\w+\s+){0,2}?"
    r"(?:visa\s+)?sponsorship"
    r"|\bh-?1b\s+(?:transfers?|ok|okay|accepted|welcome|candidates?|visa|holders?)"
    r"|\bh-?4\b"
    r"|all\s+(?:visa|work\s+authori[sz]ation)\s+(?:types|holders|statuses)"
    r"|any\s+(?:valid\s+)?work\s+authori[sz]ation|\btn\s+visa\b",
    _I,
)
# "opt" and "ead" are ordinary words in lower case, and CPT is also a medical-coding term,
# so these only count in capitals and CPT only next to OPT.
_FRIENDLY_CAPS = re.compile(
    r"\b(?:STEM OPT|OPT|EAD)\b|\bOPT\s*[/,&]\s*CPT\b|\bCPT\s*[/,&]\s*OPT\b"
    r"|\bF-?1\s+(?:visa|OPT|CPT)"
)
_OPT_MENTION = re.compile(r"\b(?:STEM OPT|OPT)\b|\bF-?1\b|\bCPT\s*[/,&]\s*OPT\b")
_CONTRACT = re.compile(
    r"\bw-?2\b|\bc2c\b|corp[\s-]*to[\s-]*corp|\b1099\b|our\s+client|contract[\s-]to[\s-](?:hire|perm)"
    r"|\bc2h\b|staffing|third[\s-]party|(?:contract|temporary)\s+(?:position|role|assignment)"
    r"|\bcontract\s+(?:job|opportunity)\b|(?:\d+|six|twelve)[\s-]*month\s+contract",
    _I,
)


def detect_visa(text: str) -> str:
    restricted = bool(_RESTRICTED.search(text))
    friendly = bool(_FRIENDLY_CAPS.search(text) or _FRIENDLY.search(text))
    if restricted and friendly:
        return "mixed"
    if restricted:
        return "restricted"
    if friendly:
        return "friendly"
    return "not_stated"


def mentions_opt(text: str) -> bool:
    return bool(_OPT_MENTION.search(text))


def is_contract_like(text: str) -> bool:
    return bool(_CONTRACT.search(text))
