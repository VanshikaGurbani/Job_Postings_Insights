"""Build the dashboard table and the measured results block from pipeline outputs.

`jobinsights report` reads only files the other commands wrote, so every number in
outputs/RESULTS.md (and the README block copied from it) is regenerated, never typed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config
from .remote import legacy_remote
from .roles import role_family, state_of
from .taxonomy import SKILL_NAMES, legacy_skills

OUT_DIR = config.DATA_DIR.parent / "outputs"
DASHBOARD_PARQUET = config.PROCESSED_DIR / "dashboard.parquet"
RESULTS_MD = OUT_DIR / "RESULTS.md"
README = Path("README.md")
MARK_START, MARK_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def build_dashboard_table() -> pd.DataFrame:
    clean = pd.read_parquet(config.CLEAN_POSTINGS)
    rules = pd.read_parquet(config.PROCESSED_DIR / "extracted_rules.parquet")
    df = clean.merge(rules, on="job_id")
    clusters_path = config.PROCESSED_DIR / "clusters.parquet"
    if clusters_path.is_file():
        df = df.merge(pd.read_parquet(clusters_path), on="job_id", how="left")
    df["role"] = df["title"].map(role_family)
    df["state"] = df["location"].map(state_of)
    df["salary_mid"] = (df["salary_annual_low"] + df["salary_annual_high"]) / 2
    keep = [
        "job_id", "title", "role", "state", "location", "formatted_experience_level",
        "remote_rules", "salary_mid", "salary_source", "skills_rules", "cluster", "x", "y",
    ]  # fmt: skip
    out = df[[c for c in keep if c in df.columns]]
    DASHBOARD_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DASHBOARD_PARQUET, index=False)
    return out


def _legacy_vs_rules(df: pd.DataFrame, clean: pd.DataFrame) -> dict:
    """How different the original notebook rules are from the regex rules on the full data."""
    leg = clean["description"].map(legacy_skills)
    leg_r = float(leg.map(lambda s: "R" in s).mean())
    new_r = float(df["skills_rules"].map(lambda s: "R" in s).mean())
    leg_remote = float((clean["description"].map(legacy_remote) == "Yes").mean())
    new_remote = float(df["remote_rules"].isin(["remote", "hybrid"]).mean())
    return {
        "postings_flagged_R_legacy": round(leg_r, 3),
        "postings_flagged_R_regex": round(new_r, 4),
        "remote_or_hybrid_legacy": round(leg_remote, 3),
        "remote_or_hybrid_regex": round(new_remote, 3),
    }


def _load(name: str) -> dict | None:
    p = OUT_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def build_results() -> str:
    clean = pd.read_parquet(config.CLEAN_POSTINGS)
    rules = pd.read_parquet(config.PROCESSED_DIR / "extracted_rules.parquet")
    n = len(clean)
    src_file = config.RAW_DIR / "source.json"
    src = json.loads(src_file.read_text())["source"] if src_file.is_file() else "unknown"
    lines = [f"**Dataset:** {n:,} cleaned postings (source: `{src}`).", ""]

    lines += [
        "### Rules baseline coverage (no labels needed)",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    has_skill = rules["skills_rules"].map(len) >= 1
    lines.append(f"| Postings with at least one taxonomy skill | {_pct(has_skill.mean())} |")
    lines.append(f"| Taxonomy size | {len(SKILL_NAMES)} skills |")
    sal = rules["salary_annual_low"].notna()
    lines.append(f"| Postings with a usable annual salary | {_pct(sal.mean())} |")
    src_counts = rules["salary_source"].value_counts()
    lines.append(
        f"| ...from the structured columns / only from the text | "
        f"{int(src_counts.get('structured', 0)):,} / {int(src_counts.get('text', 0)):,} |"
    )
    dist = rules["remote_rules"].value_counts(normalize=True)
    lines.append(
        "| Work arrangement (remote / hybrid / onsite / unknown) | "
        + " / ".join(_pct(dist.get(k, 0)) for k in ("remote", "hybrid", "onsite", "unknown"))
        + " |"
    )
    cmp = _legacy_vs_rules(rules, clean)
    lines.append(
        f'| Postings the original substring rule tags with skill "R" vs the regex rule | '
        f"{_pct(cmp['postings_flagged_R_legacy'])} vs {_pct(cmp['postings_flagged_R_regex'])} |"
    )
    lines.append(
        f'| "Remote" share: original rule (any mention, incl. hybrid) vs regex remote+hybrid | '
        f"{_pct(cmp['remote_or_hybrid_legacy'])} vs {_pct(cmp['remote_or_hybrid_regex'])} |"
    )
    lines.append("")

    ev = _load("eval_results.json")
    lines += ["### Extraction accuracy vs hand labels", ""]
    if ev is None:
        lines.append("_Not available: the hand-labelled set has not been scored yet._")
    else:
        lines.append(
            f"{ev['n_labelled']} hand-labelled postings (stratified, not population-"
            "proportional; 95% bootstrap CI on F1 in brackets)."
        )
        lines += ["", "| System | Task | Precision | Recall | F1 |", "|---|---|---|---|---|"]
        for system, res in ev["systems"].items():
            for task, m in res.items():
                if isinstance(m, dict):
                    lines.append(
                        f"| {system} | {task} | {m['precision']:.2f} | {m['recall']:.2f} | "
                        f"{m['f1']:.2f} ({m['f1_ci95'][0]:.2f}-{m['f1_ci95'][1]:.2f}) |"
                    )
    lines.append("")

    cl = _load("clusters.json")
    if cl:
        best = next(r for r in cl["sweep"] if r["k"] == cl["chosen_k"])
        lines += [
            "### Skill clusters",
            "",
            f"{cl['n_clustered']:,} of {cl['n_total']:,} postings have at least "
            f"{cl['min_hard_skills']} non-soft skills and are clustered "
            f"({cl['representation']}).",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| k chosen from sweep (silhouette among seed-stable k) | {cl['chosen_k']} |",
            f"| Silhouette (cosine) at chosen k | {best['silhouette']:.3f} |",
            f"| Seed stability, mean pairwise ARI over 5 seeds | {best['stability_ari']:.3f} |",
            f"| Original method (raw binary skills, k=15), silhouette | "
            f"{cl['baseline_original_method']['silhouette']:.3f} |",
            "",
            "| Cluster | Size | Top skills | Titles in data/software/AI role families |",
            "|---|---|---|---|",
        ]
        for c, info in cl["clusters"].items():
            lines.append(
                f"| {c}: {info['name']} | {info['size']:,} | {', '.join(info['top_skills'])} | "
                f"{_pct(info['tech_title_share'])} |"
            )
        lines.append("")

    sm = _load("salary_model.json")
    if sm:
        lines += [
            "### Salary regression (postings that state pay)",
            "",
            f"{sm['n_modelled']:,} of {sm['n_postings']:,} postings "
            f"({_pct(sm['n_modelled'] / sm['n_postings'])}) state a usable salary. {sm['cv']}. "
            f"Target: {sm['target']}.",
            "",
            "| Model | MAE (USD) | Median AE (USD) | MAPE | R^2 (log) |",
            "|---|---|---|---|---|",
        ]
        labels = {
            "median": "Global median (baseline)",
            "role_exp_median": "Median by role family x experience (baseline)",
            "ridge": "Ridge (title TF-IDF + skills + categories)",
            "gbm": "Gradient boosting (same features)",
        }
        for key, label in labels.items():
            m = sm["models"][key]
            lines.append(
                f"| {label} | {m['mae_usd']:,.0f} | {m['median_ae_usd']:,.0f} | "
                f"{_pct(m['mape'])} | {m['r2_log']:.2f} |"
            )
        lines.append("")
    return "\n".join(lines)


def run() -> None:
    build_dashboard_table()
    text = build_results()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text(text, encoding="utf-8")
    if README.is_file():
        body = README.read_text(encoding="utf-8")
        if MARK_START in body and MARK_END in body:
            head, rest = body.split(MARK_START, 1)
            _, tail = rest.split(MARK_END, 1)
            README.write_text(f"{head}{MARK_START}\n{text}\n{MARK_END}{tail}", encoding="utf-8")
            print(f"README results block updated from {RESULTS_MD}")
    print(text)
