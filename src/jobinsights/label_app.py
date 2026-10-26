"""Streamlit tool for hand-labelling the evaluation set: `jobinsights label`.

Candidates for the skills checklist are the pooled outputs of every extractor, shown in
alphabetical order and unchecked, so the labeller is not told which system proposed what.
Anything else can be added from the full taxonomy or typed in.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from jobinsights import config, evalset
from jobinsights.taxonomy import SKILL_NAMES, extract_skills, legacy_skills

HIGHLIGHT = re.compile(
    r"(\$\s?\d[\d,\.]*\s?[kK]?|\bremote\w*|\bhybrid\b|\bon[\s-]?site\b|work from home|\bwfh\b"
    r"|\bper (?:hour|year)\b|\bsalary\b|\bpay\b)",
    re.IGNORECASE,
)


def candidates(title: str, description: str, job_id: int) -> list[str]:
    pool = set(extract_skills(f"{title}. {description}")) | set(legacy_skills(description))
    llm_files = sorted(config.PROCESSED_DIR.glob("extracted_llm_*.parquet"))
    for f in llm_files:
        df = pd.read_parquet(f)
        row = df[df["job_id"] == job_id]
        if len(row):
            pool |= set(row.iloc[0]["skills_llm"])
    return sorted(pool)


def main() -> None:
    st.set_page_config(page_title="Label postings", layout="wide")
    ids = evalset.load_ids()
    gold = evalset.load_gold()
    df = pd.read_parquet(config.CLEAN_POSTINGS).set_index("job_id")
    order = evalset.labeling_order(ids)
    ready = {int(f.stem) for f in config.CACHE_LLM.glob("*/*/*.json")}
    todo = [j for j in order if j not in gold and j in ready]
    st.sidebar.metric("Labelled", f"{len(gold)} / {len(order)}")
    st.sidebar.progress(len(gold) / len(order))
    if not todo:
        if len(gold) >= len(order):
            st.success("All postings labelled.")
        else:
            st.info("Waiting for the local model to process the next postings; reload shortly.")
        return
    job_id = todo[0]
    row = df.loc[job_id]

    st.subheader(f"{row['title']}  ·  {row['location']}")
    st.caption(f"job_id {job_id}")
    body = html.escape(row["description"])
    st.markdown(HIGHLIGHT.sub(lambda m: f"<mark>{m.group(0)}</mark>", body), unsafe_allow_html=True)

    st.divider()
    with st.form(f"label_{job_id}", clear_on_submit=True):
        remote = st.radio(
            "Work arrangement",
            ["remote", "hybrid", "onsite", "unknown"],
            horizontal=True,
            index=None,
            help="onsite = text says in-person/on-site. unknown = text does not say.",
        )
        st.markdown("**Base pay stated in the text** (ignore bonus / commission / OTE)")
        c1, c2, c3, c4 = st.columns(4)
        present = c1.checkbox("Pay is stated")
        lo = c2.number_input("min", min_value=0.0, value=0.0, step=1.0)
        hi = c3.number_input("max (= min if single)", min_value=0.0, value=0.0, step=1.0)
        period = c4.selectbox("period", ["", "hourly", "weekly", "monthly", "yearly"])
        st.markdown("**Skills the posting asks the candidate to have or use**")
        cands = candidates(row["title"], row["description"], job_id)
        picked = st.multiselect("Tick the ones that are really required/used", cands, default=[])
        extra = st.multiselect(
            "Others from the taxonomy", [s for s in SKILL_NAMES if s not in cands]
        )
        other_text = st.text_input("Skills missing from the taxonomy (comma separated)")
        submitted = st.form_submit_button("Save and next")
    if submitted:
        if remote is None:
            st.error("Pick a work arrangement.")
            st.stop()
        if present and (not period or hi < lo or lo <= 0):
            st.error("Pay stated: give min <= max and a period.")
            st.stop()
        evalset.save_gold(
            {
                "job_id": job_id,
                "stratum": ids[job_id],
                "remote": remote,
                "salary": {
                    "present": bool(present),
                    "min": lo if present else None,
                    "max": (hi or lo) if present else None,
                    "period": period or None if present else None,
                },
                "skills": sorted(set(picked) | set(extra)),
                "skills_other": [s.strip() for s in other_text.split(",") if s.strip()],
                "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        st.rerun()


if __name__ == "__main__":
    main()
