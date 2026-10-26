"""Join postings to public H-1B employer history (USCIS H-1B Employer Data Hub, FY2021-2023).

A company with past H-1B approvals has shown it files petitions; that says nothing about whether
a particular posting sponsors. Employer names in the hub are legal entities ("AMAZON COM SERVICES
LLC"), so names are normalised and matched exactly or by word-prefix, with a guard against generic
names that would match many unrelated employers.
"""

from __future__ import annotations

import bisect
import re
from pathlib import Path

import pandas as pd

from . import config

H1B_DIR = config.RAW_DIR / "h1b"
H1B_YEARS = (2021, 2022, 2023)
H1B_URL = "https://www.uscis.gov/sites/default/files/document/data/h1b_datahubexport-{year}.csv"
MATCH_OUT = config.PROCESSED_DIR / "h1b_match.parquet"

_SUFFIX = {
    "INC", "LLC", "LLP", "LP", "LTD", "LIMITED", "CORP", "CORPORATION", "CO", "COMPANY",
    "PLLC", "PC", "PA", "THE", "INCORPORATED", "HOLDINGS", "GROUP",
}  # fmt: skip
_GENERIC = {
    "AMERICAN", "FIRST", "NATIONAL", "UNITED", "GLOBAL", "SOLUTIONS", "SERVICES", "TECHNOLOGIES",
    "SYSTEMS", "INTERNATIONAL", "ADVANCED", "PREMIER", "ALLIED", "UNIVERSAL", "GENERAL", "PACIFIC",
    "ATLANTIC", "CAPITAL", "DIGITAL", "MEDICAL", "HEALTH", "HEALTHCARE", "STAFFING", "CONSULTING",
}  # fmt: skip
MAX_PREFIX_EMPLOYERS = 25
MIN_APPROVALS = 5  # total approvals (initial + continuing, FY2021-23) to call it "H-1B history"


def normalize_company(name: object) -> str:
    """Upper-case, strip punctuation and trailing legal suffixes ('Amazon.com Services LLC')."""
    if not isinstance(name, str):
        return ""
    s = re.sub(r"[^A-Z0-9 ]+", " ", name.upper().replace("&", " AND "))
    tokens = s.split()
    while tokens and tokens[-1] in _SUFFIX:
        tokens.pop()
    while tokens and tokens[0] == "THE":
        tokens.pop(0)
    return " ".join(tokens)


def download(years=H1B_YEARS, dest: Path = H1B_DIR) -> None:
    import urllib.request

    dest.mkdir(parents=True, exist_ok=True)
    for y in years:
        target = dest / f"h1b_datahubexport-{y}.csv"
        if not target.is_file():
            req = urllib.request.Request(
                H1B_URL.format(year=y), headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                target.write_bytes(r.read())


def load_employers(src: Path = H1B_DIR) -> pd.DataFrame:
    """One row per normalised employer with approvals summed over the loaded years."""
    frames = []
    for f in sorted(src.glob("h1b_datahubexport-*.csv")):
        frames.append(pd.read_csv(f, dtype=str, encoding="utf-8", encoding_errors="replace"))
    if not frames:
        raise SystemExit("No H-1B files found. Run `jobinsights download-h1b`.")
    df = pd.concat(frames, ignore_index=True)
    for col in ("Initial Approval", "Continuing Approval"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["employer_norm"] = df["Employer"].map(normalize_company)
    # "X DBA Y": also index the trade name
    dba = df["Employer"].fillna("").str.upper().str.extract(r"\bDBA\s+(.+)$")[0]
    alias = df.assign(employer_norm=dba.map(normalize_company))[dba.notna()]
    df = pd.concat([df, alias], ignore_index=True)
    df = df[df["employer_norm"] != ""]
    g = df.groupby("employer_norm").agg(
        initial=("Initial Approval", "sum"), continuing=("Continuing Approval", "sum")
    )
    return g.reset_index()


class Matcher:
    def __init__(self, employers: pd.DataFrame):
        emp = employers.sort_values("employer_norm").reset_index(drop=True)
        self.names = emp["employer_norm"].tolist()
        self.initial = emp["initial"].to_numpy()
        self.continuing = emp["continuing"].to_numpy()
        self.index = {n: i for i, n in enumerate(self.names)}

    def match(self, company: object) -> dict | None:
        p = normalize_company(company)
        if not p:
            return None
        if p in self.index:
            i = self.index[p]
            return self._row(p, [i], "exact")
        tokens = p.split()
        if len(p) < 4 or (len(tokens) == 1 and (len(p) < 6 or p in _GENERIC)):
            return None
        lo = bisect.bisect_left(self.names, p + " ")
        hi = bisect.bisect_left(self.names, p + "!")  # '!' sorts just above ' '
        idx = list(range(lo, hi))
        if not idx or len(idx) > MAX_PREFIX_EMPLOYERS:
            return None
        return self._row(p, idx, "prefix")

    def _row(self, p: str, idx: list[int], kind: str) -> dict:
        return {
            "match_type": kind,
            "h1b_employers": len(idx),
            "h1b_initial": int(self.initial[idx].sum()),
            "h1b_continuing": int(self.continuing[idx].sum()),
            "h1b_names": "; ".join(self.names[i] for i in idx[:5]),
        }


def match_companies(companies: pd.Series) -> pd.DataFrame:
    """Match each distinct company name; returns one row per name with H-1B history columns."""
    matcher = Matcher(load_employers())
    rows = []
    for name in companies.dropna().unique():
        m = matcher.match(name)
        if m:
            rows.append({"company_name": name, **m})
    out = pd.DataFrame(rows)
    out["h1b_total"] = out["h1b_initial"] + out["h1b_continuing"]
    return out


def run() -> pd.DataFrame:
    from . import data

    clean = data.load_clean()
    if "company_name" not in clean.columns:
        raise SystemExit("This dataset has no company_name column (use the Kaggle file).")
    out = match_companies(clean["company_name"])
    out["h1b_history"] = out["h1b_total"] >= MIN_APPROVALS
    MATCH_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(MATCH_OUT, index=False)
    named = clean["company_name"].fillna("") != ""
    matched = clean["company_name"].isin(out["company_name"])
    hist = clean["company_name"].isin(out.loc[out["h1b_history"], "company_name"])
    print(
        f"{out['company_name'].nunique():,} company names matched "
        f"({named.sum():,} postings have a name; {matched.sum():,} matched; {hist.sum():,} from "
        f"companies with >= {MIN_APPROVALS} approvals) -> {MATCH_OUT}"
    )
    return out
