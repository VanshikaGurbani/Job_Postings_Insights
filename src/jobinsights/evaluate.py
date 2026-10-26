"""Field-level precision / recall / F1 of each extractor against the hand-labelled set."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .evalset import load_gold
from .salary import ANNUAL_FACTOR
from .taxonomy import LEGACY_KEYWORDS, legacy_skills

N_BOOT = 1000
SALARY_TOL = 0.01  # a salary counts as correct if both ends are within 1%


@dataclass
class PRF:
    p: float
    r: float
    f1: float


def prf(tp: float, fp: float, fn: float) -> PRF:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if p == p and r == r and p + r else float("nan")
    return PRF(p, r, f1)


def skill_counts(pred: list[set], gold: list[set]) -> np.ndarray:
    """Per-posting (tp, fp, fn) so bootstrap can resample postings."""
    return np.array([[len(p & g), len(p - g), len(g - p)] for p, g in zip(pred, gold, strict=True)])


def salary_counts(pred: list[tuple | None], gold: list[tuple | None]) -> np.ndarray:
    """Strict: a prediction is a TP only if the annual low and high both match the label."""
    out = []
    for p, g in zip(pred, gold, strict=True):
        if p is None and g is None:
            out.append((0, 0, 0))
        elif p is None:
            out.append((0, 0, 1))
        elif g is None:
            out.append((0, 1, 0))
        else:
            ok = all(abs(a - b) <= SALARY_TOL * b for a, b in zip(p, g, strict=True))
            out.append((1, 0, 0) if ok else (0, 1, 1))
    return np.array(out)


def presence_counts(pred: list[bool], gold: list[bool]) -> np.ndarray:
    return np.array(
        [[p and g, p and not g, g and not p] for p, g in zip(pred, gold, strict=True)], dtype=int
    )


def boot_ci(counts: np.ndarray, seed: int = 0) -> tuple[PRF, PRF, PRF]:
    """Point estimate and 95% bootstrap bounds (resampling postings) for micro P/R/F1."""
    rng = np.random.default_rng(seed)
    point = prf(*counts.sum(axis=0))
    draws = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(counts), len(counts))
        d = prf(*counts[idx].sum(axis=0))
        draws.append((d.p, d.r, d.f1))
    lo = np.nanpercentile(draws, 2.5, axis=0)
    hi = np.nanpercentile(draws, 97.5, axis=0)
    return point, PRF(*lo), PRF(*hi)


def fmt(counts: np.ndarray) -> dict:
    pt, lo, hi = boot_ci(counts)
    return {
        "precision": round(pt.p, 3),
        "recall": round(pt.r, 3),
        "f1": round(pt.f1, 3),
        "f1_ci95": [round(lo.f1, 3), round(hi.f1, 3)],
        "n_postings": int(len(counts)),
    }


def _annual(salary: dict | None) -> tuple[float, float] | None:
    if not salary or not salary.get("present") or not salary.get("min") or not salary.get("period"):
        return None
    f = ANNUAL_FACTOR[salary["period"]]
    hi = salary["max"] or salary["min"]
    return salary["min"] * f, hi * f


def evaluate() -> dict:
    gold = load_gold()
    if not gold:
        raise SystemExit("No gold labels yet. Run `jobinsights label`.")
    clean = pd.read_parquet(config.CLEAN_POSTINGS).set_index("job_id")
    rules = pd.read_parquet(config.PROCESSED_DIR / "extracted_rules.parquet").set_index("job_id")
    llm_frames = {
        f.stem.removeprefix("extracted_llm_"): pd.read_parquet(f).set_index("job_id")
        for f in sorted(config.PROCESSED_DIR.glob("extracted_llm_*.parquet"))
    }
    ids = [j for j in sorted(gold) if j in clean.index]
    for df in llm_frames.values():  # score only postings every system processed
        ids = [j for j in ids if j in df.index]

    g_sk = [set(gold[j]["skills"]) for j in ids]
    legacy_vocab = set(LEGACY_KEYWORDS)
    # The legacy matcher only knows 18 skills; compare it on that vocabulary only.
    g_sk_legacy = [g & legacy_vocab for g in g_sk]
    systems: dict[str, dict] = {
        "legacy": {
            "skills": [set(legacy_skills(clean.loc[j, "description"])) for j in ids],
            "remote": [
                "remote" if legacy_yes(clean.loc[j, "description"]) else "unknown" for j in ids
            ],
            "salary": [None] * len(ids),
        },
        "rules": {
            "skills": [set(rules.loc[j, "skills_rules"]) for j in ids],
            "remote": [rules.loc[j, "remote_rules"] for j in ids],
            "salary": [_text_salary(rules.loc[j]) for j in ids],
        },
    }
    for name, df in llm_frames.items():
        systems[f"llm:{name}"] = {
            "skills": [set(df.loc[j, "skills_llm"]) for j in ids],
            "remote": [df.loc[j, "remote_llm"] for j in ids],
            "salary": [_llm_salary(df.loc[j]) for j in ids],
        }

    g_remote = [gold[j]["remote"] for j in ids]
    g_salary = [_annual(gold[j]["salary"]) for j in ids]
    results: dict = {"n_labelled": len(ids), "systems": {}}
    for name, s in systems.items():
        res: dict = {}
        if name == "legacy":
            res["skills_18_legacy_vocab"] = fmt(
                skill_counts([p & legacy_vocab for p in s["skills"]], g_sk_legacy)
            )
        else:
            res["skills_full_taxonomy"] = fmt(skill_counts(s["skills"], g_sk))
            res["skills_18_legacy_vocab"] = fmt(
                skill_counts([p & legacy_vocab for p in s["skills"]], g_sk_legacy)
            )
            res["salary_presence"] = fmt(
                presence_counts(
                    [p is not None for p in s["salary"]], [g is not None for g in g_salary]
                )
            )
            res["salary_strict"] = fmt(salary_counts(s["salary"], g_salary))
        offers_remote = [r in ("remote", "hybrid") for r in s["remote"]]
        res["offers_remote_or_hybrid"] = fmt(
            presence_counts(offers_remote, [r in ("remote", "hybrid") for r in g_remote])
        )
        if name != "legacy":
            res["remote_4class_accuracy"] = round(
                float(np.mean([p == g for p, g in zip(s["remote"], g_remote, strict=True)])), 3
            )
        results["systems"][name] = res
    results["gold_remote_distribution"] = pd.Series(g_remote).value_counts().to_dict()
    results["gold_salary_present"] = int(sum(g is not None for g in g_salary))
    out = config.DATA_DIR.parent / "outputs" / "eval_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=1))
    return results


def legacy_yes(text: str) -> bool:
    from .remote import legacy_remote

    return legacy_remote(text) == "Yes"


def _text_salary(row: pd.Series) -> tuple[float, float] | None:
    lo = row["salary_text_low"]
    return None if pd.isna(lo) else (float(lo), float(row["salary_text_high"]))


def _llm_salary(row: pd.Series) -> tuple[float, float] | None:
    lo = row["salary_llm_low"]
    return None if pd.isna(lo) else (float(lo), float(row["salary_llm_high"]))
