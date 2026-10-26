"""Run the rule-based extractors over the cleaned postings table."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import config
from .remote import detect_remote
from .salary import normalize_structured, parse_salary_text
from .taxonomy import extract_skills
from .visa import detect_visa, is_contract_like, mentions_opt

RULES_OUT = config.PROCESSED_DIR / "extracted_rules.parquet"


def _nan_to_none(x: object) -> float | None:
    return None if pd.isna(x) else float(x)


def extract_rules(df: pd.DataFrame) -> pd.DataFrame:
    """One row per posting: skills, remote class, salary from text and from structured columns."""
    rows = []
    for rec in df.itertuples(index=False):
        text = f"{rec.title}. {rec.description}"
        skills = extract_skills(text)
        text_pay = parse_salary_text(rec.description)
        struct_pay = normalize_structured(
            _nan_to_none(getattr(rec, "min_salary", None)),
            _nan_to_none(getattr(rec, "max_salary", None)),
            getattr(rec, "pay_period", None)
            if isinstance(getattr(rec, "pay_period", None), str)
            else None,
            _nan_to_none(getattr(rec, "med_salary", None)),
        )
        best = struct_pay or text_pay
        rows.append(
            {
                "job_id": rec.job_id,
                "skills_rules": skills,
                "remote_rules": detect_remote(rec.description),
                "visa_rules": detect_visa(rec.description),
                "opt_mention_rules": mentions_opt(rec.description),
                "contract_rules": is_contract_like(rec.description),
                "salary_text_low": text_pay.annual_low if text_pay else None,
                "salary_text_high": text_pay.annual_high if text_pay else None,
                "salary_struct_low": struct_pay.annual_low if struct_pay else None,
                "salary_struct_high": struct_pay.annual_high if struct_pay else None,
                "salary_annual_low": best.annual_low if best else None,
                "salary_annual_high": best.annual_high if best else None,
                "salary_source": best.source if best else None,
            }
        )
    return pd.DataFrame(rows)


def run_rules(df: pd.DataFrame, workers: int | None = None) -> pd.DataFrame:
    """`extract_rules` over chunks in parallel (skill regexes cost ~8 ms per posting)."""
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    bounds = np.linspace(0, len(df), max(1, workers * 4) + 1).astype(int)
    chunks = [df.iloc[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b > a]
    if workers == 1 or len(df) < 500:
        return extract_rules(df)
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(extract_rules, chunks))
    return pd.concat(parts, ignore_index=True)


def extract(src=config.CLEAN_POSTINGS, dest=RULES_OUT, workers: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(src)
    out = run_rules(df, workers)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    print(f"rules baseline on {len(out):,} postings -> {dest}")
    return out
