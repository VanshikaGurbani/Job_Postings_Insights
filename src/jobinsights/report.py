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
    h1b_path = config.PROCESSED_DIR / "h1b_match.parquet"
    if h1b_path.is_file() and "company_name" in df.columns:
        m = pd.read_parquet(h1b_path)[["company_name", "h1b_total", "h1b_history"]]
        df = df.merge(m, on="company_name", how="left")
        df["h1b_total"] = df["h1b_total"].fillna(0).astype(int)
        df["h1b_history"] = df["h1b_history"].fillna(False).astype(bool)
    df["role"] = df["title"].map(role_family)
    df["state"] = df["location"].map(state_of)
    df["salary_mid"] = (df["salary_annual_low"] + df["salary_annual_high"]) / 2
    keep = [
        "job_id", "title", "role", "state", "location", "formatted_experience_level",
        "remote_rules", "visa_rules", "opt_mention_rules", "contract_rules", "salary_mid",
        "salary_source", "skills_rules", "cluster", "x", "y", "company_name", "job_posting_url",
        "h1b_total", "h1b_history",
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
    vd = rules["visa_rules"].value_counts(normalize=True)
    lines.append(
        "| Work authorisation (friendly / restricted / mixed / not stated) | "
        + " / ".join(_pct(vd.get(k, 0)) for k in ("friendly", "restricted", "mixed", "not_stated"))
        + " |"
    )
    lines.append(
        f"| Postings mentioning OPT / STEM OPT / F-1 | {_pct(rules['opt_mention_rules'].mean())} |"
    )
    lines.append(f"| Contract / staffing-style postings | {_pct(rules['contract_rules'].mean())} |")
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

    h1b_path = config.PROCESSED_DIR / "h1b_match.parquet"
    if h1b_path.is_file() and "company_name" in clean.columns:
        m = pd.read_parquet(h1b_path)
        named = clean["company_name"].fillna("") != ""
        hist_names = set(m.loc[m["h1b_history"], "company_name"])
        in_hist = clean["company_name"].isin(hist_names)
        restricted = (
            rules.set_index("job_id").loc[clean["job_id"], "visa_rules"].eq("restricted").to_numpy()
        )
        friendly = (
            rules.set_index("job_id").loc[clean["job_id"], "visa_rules"].eq("friendly").to_numpy()
        )
        shortlist = ((friendly | in_hist.to_numpy()) & ~restricted).mean()
        chk_path = config.LABELS_DIR / "h1b_match_check.json"
        chk = json.loads(chk_path.read_text()) if chk_path.is_file() else None
        top = (
            clean[in_hist]["company_name"]
            .value_counts()
            .head(10)
            .rename("openings")
            .reset_index()
            .merge(m[["company_name", "h1b_total"]], on="company_name")
        )
        lines += [
            "### Company H-1B history (public USCIS data, FY2021-2023)",
            "",
            "A join on company names, not on individual postings: a company with past approvals "
            "has shown it files H-1B petitions, which does not mean this posting sponsors.",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Postings with a company name matched to a USCIS employer | "
            f"{_pct(clean['company_name'].isin(m['company_name']).sum() / named.sum())} |",
            f"| Postings from companies with >= 5 H-1B approvals in FY2021-23 | "
            f"{_pct(in_hist.sum() / named.sum())} |",
            "| Postings that are not restricted and (friendly text or H-1B history) | "
            f"{_pct(shortlist)} |",
        ]
        if chk:
            lines.append(
                f"| Match spot check (random sample, read by eye) | "
                f"{chk['judged_same_company']} of {chk['n']} same company |"
            )
        lines += [
            "",
            "| Top companies with H-1B history by openings | Openings | Approvals FY21-23 |",
            "|---|---|---|",
        ]
        for r in top.itertuples():
            lines.append(f"| {r.company_name} | {r.openings:,} | {r.h1b_total:,} |")
        lines.append("")

    ev = _load("eval_results.json")
    lines += ["### Extraction accuracy vs reference labels", ""]
    if ev is None:
        lines.append("_Not available: the reference set has not been scored yet._")
    else:
        lines.append(
            f"{ev['n_labelled']} postings with reference labels (stratified, not population-"
            "proportional; 95% bootstrap CI on F1 in brackets). The rules and prompts were "
            "frozen before this sample was drawn and labelled, so these are held-out scores."
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

    vc = (ev or {}).get("visa_and_contract_rules") or {}
    if vc.get("n"):
        lines += [
            "### Work-authorisation and contract signals (rules vs reference labels)",
            "",
            f"{vc['n']} labelled postings, including ones the rule flagged (friendly / "
            "restricted) and random ones. Flagged postings are over-represented, so "
            "**only precision is meaningful here; recall cannot be estimated** because "
            "only about 5% of postings say anything about work authorisation.",
            "",
            "| Signal | Precision | n labelled |",
            "|---|---|---|",
            f'| Rule says "friendly" (sponsorship / OPT welcomed) | '
            f"{vc['visa_friendly']['precision']:.2f} | {vc['n']} |",
            f'| Rule says "restricted" or "mixed" | '
            f"{vc['visa_restricted_or_mixed']['precision']:.2f} | {vc['n']} |",
            f"| Rule says contract / staffing-style | {vc['contract']['precision']:.2f} | "
            f"{vc['n']} |",
            "",
        ]

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
