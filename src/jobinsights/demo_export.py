"""Aggregate-only data bundle for the hosted demo (`jobinsights export-demo`).

The demo app reads only these small files, never postings: no description text, no posting rows,
no links. Everything is a count, share or median over groups, and group medians are blank below
a minimum number of observations. Company-level rows are limited to companies with several
openings and carry only counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .insights import (
    DATA_ROLES,
    ENTRY_LEVELS,
    SENIOR_LEVELS,
    TECH_ROLES,
    fit_pay_model,
    hard_skills,
    skill_matrix,
    skill_premiums,
    sponsor_landscape,
)

DEMO_DIR = Path("demo") / "data"
ALL_TECH = "All tech roles"
MIN_PAY_N = 15  # no pay median below this many postings with stated pay
MIN_COMPANY_OPENINGS = 3
MIN_CITY_POSTINGS = 30
MIN_PAIR_COUNT = 15


def _groups(df: pd.DataFrame):
    """(role label, level group, rows) for every role plus the pooled tech roles."""
    levels = {
        "all": None,
        "entry": ENTRY_LEVELS,
        "senior": SENIOR_LEVELS,
    }
    for role in (*TECH_ROLES, ALL_TECH):
        sub = df[df["role"].isin(TECH_ROLES)] if role == ALL_TECH else df[df["role"] == role]
        for name, lv in levels.items():
            yield role, name, sub if lv is None else sub[sub["formatted_experience_level"].isin(lv)]


def role_skill(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for role, group, sub in _groups(df):
        exploded = sub["skills_rules"].map(hard_skills).explode().dropna()
        for skill, n in exploded.value_counts().items():
            rows.append((role, group, skill, int(n), len(sub)))
    return pd.DataFrame(rows, columns=["role", "group", "skill", "n_with_skill", "n_postings"])


def skill_sets(df: pd.DataFrame) -> pd.DataFrame:
    """Distinct technical skill sets per role and level group with how many postings have each."""
    rows = []
    for role, group, sub in _groups(df):
        if group == "senior":
            continue
        sets = sub["skills_rules"].map(lambda s: "|".join(sorted(hard_skills(s))))
        counts = sets[sets.str.count(r"\|") >= 1].value_counts()  # at least two skills
        rows += [(role, group, s, int(n)) for s, n in counts.items()]
    return pd.DataFrame(rows, columns=["role", "group", "skills", "n"])


def cooccurrence(df: pd.DataFrame, min_skill_n: int = 30) -> pd.DataFrame:
    """Skill pairs per role: joint count, conditional share and lift (>1 = go together)."""
    rows = []
    for role in (*TECH_ROLES, ALL_TECH):
        sub = df[df["role"].isin(TECH_ROLES)] if role == ALL_TECH else df[df["role"] == role]
        lists = sub["skills_rules"].map(hard_skills)
        counts = lists.explode().value_counts()
        names = sorted(counts[counts >= min_skill_n].index)
        if len(names) < 2:
            continue
        m = skill_matrix(lists, names).astype(np.int64)
        joint = m.T @ m
        n = len(m)
        for i, a in enumerate(names):
            for j in range(i + 1, len(names)):
                c = int(joint[i, j])
                if c < MIN_PAIR_COUNT:
                    continue
                lift = c * n / (int(joint[i, i]) * int(joint[j, j]))
                rows.append((role, a, names[j], c, int(joint[i, i]), int(joint[j, j]), n, lift))
    cols = ["role", "skill_a", "skill_b", "both", "n_a", "n_b", "n_postings", "lift"]
    return pd.DataFrame(rows, columns=cols)


def _city(location: str) -> str | None:
    head = (location or "").split(",")[0].strip()
    return head or None


def geography(df: pd.DataFrame) -> pd.DataFrame:
    """Postings, pay, entry share, remote share and sponsor-screen share by state and city."""
    rows = []
    sets = {"Data & AI": DATA_ROLES, "Software Engineer": ("Software Engineer",)}
    df = df.assign(city=df["location"].map(_city))
    for label, roles in sets.items():
        g = df[df["role"].isin(roles) & df["state"].notna()]
        for level, key, minimum in (
            ("state", "state", 1),
            ("city", ["city", "state"], MIN_CITY_POSTINGS),
        ):
            grouped = g.groupby(key)
            for name, sub in grouped:
                if len(sub) < minimum:
                    continue
                state, city = (name, "") if level == "state" else (name[1], name[0])
                pay = sub["salary_mid"].dropna()
                shortlisted = (sub["visa_rules"] != "restricted") & (
                    (sub["visa_rules"] == "friendly") | sub["h1b_history"]
                )
                rows.append(
                    {
                        "family": label,
                        "level": level,
                        "state": state,
                        "city": city,
                        "postings": len(sub),
                        "pay_n": len(pay),
                        "median_pay": float(pay.median()) if len(pay) >= MIN_PAY_N else None,
                        "entry_share": float(
                            sub["formatted_experience_level"].isin(ENTRY_LEVELS).mean()
                        ),
                        "remote_hybrid_share": float(
                            sub["remote_rules"].isin(["remote", "hybrid"]).mean()
                        ),
                        "sponsor_screen_share": float(shortlisted.mean()),
                    }
                )
    return pd.DataFrame(rows)


def employers(df: pd.DataFrame) -> pd.DataFrame:
    """Companies with at least a few data / AI openings, counts only (no posting detail)."""
    g = df[df["role"].isin(DATA_ROLES) & df["company_name"].notna()]
    grouped = g.groupby("company_name")
    out = pd.DataFrame(
        {
            "openings": grouped.size(),
            "top_role": grouped["role"].agg(lambda r: r.value_counts().index[0]),
            "entry_openings": grouped["formatted_experience_level"].apply(
                lambda v: int(v.isin(ENTRY_LEVELS).sum())
            ),
            "restricted_openings": grouped["visa_rules"].apply(
                lambda v: int((v == "restricted").sum())
            ),
            "friendly_text_openings": grouped["visa_rules"].apply(
                lambda v: int((v == "friendly").sum())
            ),
            "h1b_approvals_fy21_23": grouped["h1b_total"].first().astype(int),
            "pay_n": grouped["salary_mid"].count(),
            "median_pay": grouped["salary_mid"].median(),
        }
    )
    out = out[out["openings"] >= MIN_COMPANY_OPENINGS]
    out.loc[out["pay_n"] < MIN_COMPANY_OPENINGS, "median_pay"] = np.nan
    return out.sort_values("openings", ascending=False).reset_index()


def role_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sponsors = {r["role"]: r for r in sponsor_landscape(df)}
    for role in (*TECH_ROLES, "Other"):
        if role not in sponsors:
            continue
        g = df[df["role"] == role]
        pay = g["salary_mid"].dropna()
        sp = sponsors[role]
        rows.append(
            {
                "role": role,
                "postings": len(g),
                "pay_stated_share": float(g["salary_mid"].notna().mean()),
                "pay_n": len(pay),
                "median_pay": float(pay.median()) if len(pay) >= MIN_PAY_N else None,
                "entry_share": sp["entry_or_associate"],
                "remote_share": float((g["remote_rules"] == "remote").mean()),
                "hybrid_share": float((g["remote_rules"] == "hybrid").mean()),
                "restricted_share": sp["restricted"],
                "friendly_text_share": sp["friendly_text"],
                "company_h1b_history_share": sp["company_h1b_history"],
                "sponsor_screen_share": sp["shortlist"],
            }
        )
    return pd.DataFrame(rows)


def build(df: pd.DataFrame) -> dict:
    return {
        "role_overview": role_overview(df),
        "role_skill": role_skill(df),
        "skill_sets": skill_sets(df),
        "cooccurrence": cooccurrence(df),
        "geography": geography(df),
        "employers": employers(df),
    }


def run(out_dir: Path = DEMO_DIR) -> Path:
    from .report import DASHBOARD_PARQUET

    if not DASHBOARD_PARQUET.is_file():
        raise SystemExit("Run `jobinsights report` first.")
    df = pd.read_parquet(DASHBOARD_PARQUET)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in build(df).items():
        frame.to_csv(out_dir / f"{name}.csv", index=False, float_format="%.6g")
    (out_dir / "pay_model.json").write_text(json.dumps(fit_pay_model(df)), encoding="utf-8")
    prem = skill_premiums(df)
    (out_dir / "premiums.json").write_text(json.dumps(prem, indent=1), encoding="utf-8")
    meta = {
        "n_postings": int(len(df)),
        "source": "LinkedIn Job Postings (2023-2024), Kaggle arshkon/linkedin-job-postings",
        "min_pay_n": MIN_PAY_N,
    }
    for name in ("clusters.json", "eval_results.json"):
        src = config.DATA_DIR.parent / "outputs" / name
        if src.is_file():
            (out_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    size = sum(f.stat().st_size for f in out_dir.iterdir())
    print(f"demo data -> {out_dir} ({size / 1e6:.2f} MB)")
    return out_dir
