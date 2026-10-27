"""Streamlit dashboard: `jobinsights dashboard` (needs the dashboard extra and a report run)."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from jobinsights import config
from jobinsights.report import DASHBOARD_PARQUET
from jobinsights.taxonomy import FAMILY

MIN_N = 15  # hide groups with fewer salary observations than this
REMOTE_ORDER = ["remote", "hybrid", "onsite", "unknown"]
REMOTE_COLORS = {
    "remote": "#2a9d8f",
    "hybrid": "#8ab17d",
    "onsite": "#264653",
    "unknown": "#c9ced6",
}


@st.cache_data
def load() -> pd.DataFrame:
    if not DASHBOARD_PARQUET.is_file():
        st.error("Run `jobinsights run-all` (or `jobinsights report`) first.")
        st.stop()
    return pd.read_parquet(DASHBOARD_PARQUET)


def main() -> None:
    st.set_page_config(page_title="Job postings insights", layout="wide")
    df = load()
    st.title("Job postings insights")
    st.caption(
        f"{len(df):,} US postings. Skills, salary and work arrangement are extracted by the "
        "rules baseline; salary charts use only the postings that state pay."
    )

    roles = sorted(df["role"].unique())
    with st.sidebar:
        st.header("Filters")
        role_sel = st.multiselect("Role family", roles, default=roles)
        exp_opts = sorted(df["formatted_experience_level"].dropna().unique())
        exp_sel = st.multiselect("Experience level", exp_opts, default=exp_opts)
        visa_sel = st.multiselect(
            "Work authorisation signal",
            ["friendly", "restricted", "mixed", "not_stated"],
            default=["friendly", "restricted", "mixed", "not_stated"],
            help="friendly = offers sponsorship or welcomes OPT/H-1B/visas; restricted = no "
            "sponsorship or citizen-only. Rule-based; check the posting.",
        )
        only_contract = st.checkbox("Contract / staffing-style postings only")
        only_opt = st.checkbox("Mentions OPT / STEM OPT / F-1")
        has_h1b = "h1b_history" in df.columns
        only_h1b = st.checkbox(
            "Company has H-1B approvals (USCIS FY2021-23)",
            disabled=not has_h1b,
            help="Company-level history, not a promise that this posting sponsors.",
        )
        top_n = st.slider("Top skills shown", 5, 30, 15)
        skill_family = st.selectbox("Skill family", ["All"] + sorted(set(FAMILY.values())), index=0)
    f = df[df["role"].isin(role_sel)]
    f = f[f["formatted_experience_level"].isin(exp_sel) | f["formatted_experience_level"].isna()]

    f = f[f["visa_rules"].isin(visa_sel)]
    if only_contract:
        f = f[f["contract_rules"]]
    if only_opt:
        f = f[f["opt_mention_rules"]]
    if only_h1b:
        f = f[f["h1b_history"]]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Postings", f"{len(f):,}")
    k2.metric("State pay", f"{f['salary_mid'].notna().mean():.0%}")
    k3.metric(
        "Median annual pay",
        f"${f['salary_mid'].median():,.0f}" if f["salary_mid"].notna().any() else "n/a",
    )
    k4.metric("Remote or hybrid", f"{f['remote_rules'].isin(['remote', 'hybrid']).mean():.0%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Skills demand")
        exploded = f[["job_id", "skills_rules"]].explode("skills_rules").dropna()
        if skill_family != "All":
            exploded = exploded[exploded["skills_rules"].map(FAMILY) == skill_family]
        counts = exploded["skills_rules"].value_counts().head(top_n) / max(len(f), 1) * 100
        fig = px.bar(
            counts[::-1],
            orientation="h",
            labels={"count": "% of postings", "value": "% of postings", "skills_rules": ""},
        )
        fig.update_layout(showlegend=False, height=460, margin=dict(l=0, r=0, t=10, b=0))
        fig.update_traces(marker_color="#2a6f97")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Remote share by role")
        share = (
            f.groupby("role")["remote_rules"]
            .value_counts(normalize=True)
            .rename("share")
            .reset_index()
        )
        fig = px.bar(
            share,
            x="share",
            y="role",
            color="remote_rules",
            orientation="h",
            category_orders={"remote_rules": REMOTE_ORDER},
            color_discrete_map=REMOTE_COLORS,
            labels={"share": "share of postings", "role": "", "remote_rules": ""},
        )
        fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0), xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Salary (postings that state pay)")
    s = f.dropna(subset=["salary_mid"])
    a, b = st.columns(2)
    with a:
        by_role = s.groupby("role")["salary_mid"].agg(["median", "count"]).query("count >= @MIN_N")
        fig = px.bar(
            by_role.reset_index().sort_values("median"),
            x="median",
            y="role",
            orientation="h",
            text="count",
            labels={"median": "median annual pay (USD)", "role": ""},
            title="By role family (label = n)",
        )
        fig.update_traces(marker_color="#e9a03b", texttemplate="n=%{text}")
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        by_state = (
            s.dropna(subset=["state"])
            .groupby("state")["salary_mid"]
            .agg(["median", "count"])
            .query("count >= @MIN_N")
            .sort_values("median", ascending=False)
            .head(15)
        )
        fig = px.bar(
            by_state.reset_index(),
            x="state",
            y="median",
            text="count",
            labels={"median": "median annual pay (USD)", "state": ""},
            title=f"By state, top 15 with n >= {MIN_N} (label = n)",
        )
        fig.update_traces(marker_color="#e9a03b", texttemplate="n=%{text}")
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Pay is annualised (hourly x 2080). Postings that state pay are not a random sample, so "
        "these are medians for pay-transparent postings, not for all postings."
    )

    if "h1b_history" in f.columns:
        st.subheader("Shortlist: not restricted, and sponsorship text or company H-1B history")
        short = f[
            (f["visa_rules"] != "restricted") & ((f["visa_rules"] == "friendly") | f["h1b_history"])
        ]
        short = short.sort_values(["visa_rules", "h1b_total"], ascending=[True, False])
        cols = {
            "company_name": "Company",
            "title": "Title",
            "location": "Location",
            "role": "Role family",
            "visa_rules": "Visa text signal",
            "h1b_total": "Company H-1B approvals FY21-23",
            "contract_rules": "Contract / staffing-style",
            "salary_mid": "Annual pay (USD)",
            "job_posting_url": "Link",
        }
        view = short[list(cols)].rename(columns=cols)
        st.caption(
            f"{len(short):,} postings match the filters above. Company history is a hint, "
            "not a guarantee; read each posting."
        )
        st.dataframe(
            view.head(500),
            hide_index=True,
            use_container_width=True,
            column_config={"Link": st.column_config.LinkColumn("Link", display_text="open")},
        )
        st.download_button(
            "Download shortlist (CSV)", view.to_csv(index=False).encode("utf-8"), "shortlist.csv"
        )

    if "cluster" in f.columns and f["cluster"].notna().any():
        st.subheader("Skill clusters")
        c = f.dropna(subset=["cluster"]).copy()
        names_file = config.DATA_DIR.parent / "outputs" / "clusters.json"
        names = {}
        if names_file.is_file():
            info = json.loads(names_file.read_text(encoding="utf-8"))["clusters"]
            names = {int(k): f"{k}: {v['name']}" for k, v in info.items()}
        c["cluster"] = c["cluster"].astype(int).map(lambda i: names.get(i, str(i)))
        sample = c.sample(min(6000, len(c)), random_state=0)
        fig = px.scatter(
            sample,
            x="x",
            y="y",
            color="cluster",
            hover_data=["title"],
            opacity=0.6,
            labels={"x": "component 1", "y": "component 2"},
        )
        fig.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Clusters named by their most distinctive skills; full table in outputs/RESULTS.md."
        )


if __name__ == "__main__":
    main()
