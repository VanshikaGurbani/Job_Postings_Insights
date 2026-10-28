"""Hosted demo: explore the job-market findings from aggregate data only.

Reads the small files in demo/data (counts, shares and medians per group; no postings, no
description text). Rebuild them with `jobinsights export-demo`. Run locally with
`streamlit run demo/streamlit_app.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))  # lets the app run without installing the package

from jobinsights import insights  # noqa: E402

DATA = ROOT / "data"
BLUE, TEAL, ORANGE, GREY = "#2a6f97", "#2a9d8f", "#e9a03b", "#c9ced6"
ALL_TECH = "All tech roles"
ROLES = [*insights.TECH_ROLES, ALL_TECH]
MIN_NOTE = "Medians are blank where fewer than 15 postings state pay."


@st.cache_data
def table(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


@st.cache_data
def blob(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def fig_layout(fig, height: int = 420):
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), legend_title_text="")
    return fig


def overview() -> None:
    meta = blob("meta")
    ov = table("role_overview")
    data_roles = ov[ov["role"].isin(insights.DATA_ROLES)]
    da = ov[ov["role"] == "Data Analyst"].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Postings analysed", f"{meta['n_postings']:,}")
    b.metric("Data and AI postings", f"{int(data_roles['postings'].sum()):,}")
    c.metric("Data Analyst median pay", f"${da['median_pay']:,.0f}")
    d.metric("Data Analyst postings passing the sponsor screen", pct(da["sponsor_screen_share"]))
    st.caption(
        "Sponsor screen: the posting does not rule sponsorship out, and either says it is possible "
        "or comes from a company with H-1B approvals in FY2021-23. It is a filter, not a promise."
    )
    show = ov.rename(
        columns={
            "role": "Role family",
            "postings": "Postings",
            "median_pay": "Median pay (USD)",
            "pay_stated_share": "State pay",
            "entry_share": "Entry / associate",
            "remote_share": "Remote",
            "hybrid_share": "Hybrid",
            "restricted_share": "Rule out sponsorship",
            "sponsor_screen_share": "Pass sponsor screen",
        }
    )
    pct_cols = ["State pay", "Entry / associate", "Remote", "Hybrid"]
    pct_cols += ["Rule out sponsorship", "Pass sponsor screen"]
    show[pct_cols] = (show[pct_cols] * 100).round(0)
    cfg = {c: st.column_config.NumberColumn(c, format="%d%%") for c in pct_cols}
    cfg["Median pay (USD)"] = st.column_config.NumberColumn(format="dollar")
    cols = ["Role family", "Postings", "Median pay (USD)", *pct_cols]
    st.dataframe(show[cols], hide_index=True, column_config=cfg, width="stretch")
    st.caption(MIN_NOTE + " 'Other' is every posting that is not a data, AI or software role.")

    melted = ov[ov["role"].isin(insights.TECH_ROLES)].melt(
        id_vars="role",
        value_vars=["entry_share", "sponsor_screen_share", "pay_stated_share"],
        var_name="metric",
        value_name="share",
    )
    names = {
        "entry_share": "Entry / associate / internship",
        "sponsor_screen_share": "Pass sponsor screen",
        "pay_stated_share": "State pay",
    }
    melted["metric"] = melted["metric"].map(names)
    fig = px.bar(
        melted,
        x="share",
        y="role",
        color="metric",
        barmode="group",
        orientation="h",
        color_discrete_sequence=[TEAL, BLUE, ORANGE],
        labels={"share": "share of postings", "role": ""},
    )
    fig.update_layout(xaxis_tickformat=".0%")
    st.plotly_chart(fig_layout(fig, 380), width="stretch")


def skills_tab() -> None:
    rs = table("role_skill")
    a, b, c = st.columns([1, 1, 1])
    role = a.selectbox("Role family", ROLES, index=ROLES.index("Data Analyst"), key="sk_role")
    group = b.radio(
        "Seniority",
        ["all", "entry", "senior"],
        horizontal=True,
        key="sk_group",
        format_func={"all": "All", "entry": "Entry / associate", "senior": "Mid-senior+"}.get,
    )
    top = c.slider("Skills shown", 5, 25, 15, key="sk_top")
    sub = rs[(rs["role"] == role) & (rs["group"] == group)].copy()
    sub["share"] = sub["n_with_skill"] / sub["n_postings"]
    sub = sub.sort_values("share", ascending=False).head(top)
    st.subheader(f"Technical skills in {role} postings (n={int(sub['n_postings'].iloc[0]):,})")
    fig = px.bar(
        sub[::-1], x="share", y="skill", orientation="h", labels={"share": "", "skill": ""}
    )
    fig.update_traces(marker_color=BLUE)
    fig.update_layout(xaxis_tickformat=".0%")
    st.plotly_chart(fig_layout(fig, 24 * len(sub) + 60), width="stretch")

    st.subheader("What changes with seniority")
    e = rs[(rs["role"] == role) & (rs["group"] == "entry")].set_index("skill")
    s = rs[(rs["role"] == role) & (rs["group"] == "senior")].set_index("skill")
    both = pd.DataFrame(
        {
            "entry": e["n_with_skill"] / e["n_postings"],
            "senior": s["n_with_skill"] / s["n_postings"],
            "n": e["n_with_skill"] + s["n_with_skill"],
        }
    ).fillna(0.0)
    both = both[both["n"] >= 20]
    both["gap"] = both["senior"] - both["entry"]
    gaps = pd.concat([both.nsmallest(6, "gap"), both.nlargest(6, "gap")]).drop_duplicates()
    gaps = gaps.sort_values("gap").reset_index(names="skill")
    gaps["direction"] = gaps["gap"].map(lambda g: "more at senior" if g > 0 else "more at entry")
    fig = px.bar(
        gaps,
        x="gap",
        y="skill",
        color="direction",
        orientation="h",
        color_discrete_map={"more at senior": BLUE, "more at entry": TEAL},
        labels={"gap": "difference in share of postings (senior minus entry)", "skill": ""},
    )
    fig.update_layout(xaxis_tickformat="+.0%")
    st.plotly_chart(fig_layout(fig, 380), width="stretch")
    st.caption(
        f"Entry n={int(e['n_postings'].iloc[0]):,}, mid-senior n={int(s['n_postings'].iloc[0]):,}. "
        "Skills with fewer than 20 mentions across both groups are left out."
    )

    st.subheader("Skills that go together")
    co = table("cooccurrence")
    co = co[co["role"] == role]
    skills = sorted(set(co["skill_a"]) | set(co["skill_b"]))
    pick = st.selectbox("Pick a skill", skills, index=skills.index("SQL") if "SQL" in skills else 0)
    mine = co[(co["skill_a"] == pick) | (co["skill_b"] == pick)].copy()
    mine["partner"] = mine.apply(lambda r: r.skill_b if r.skill_a == pick else r.skill_a, axis=1)
    mine["n_pick"] = mine.apply(lambda r: r.n_a if r.skill_a == pick else r.n_b, axis=1)
    mine["share_with_pick"] = mine["both"] / mine["n_pick"]
    best = mine.sort_values("share_with_pick", ascending=False).head(10)
    best = best.rename(columns={"both": "postings with both"})
    out = best[["partner", "share_with_pick", "lift", "postings with both"]].copy()
    out["share_with_pick"] = (100 * out["share_with_pick"]).round(0)
    st.caption(
        f"Among {role} postings that list {pick}, the share that also list each skill. "
        "Lift above 1 means the pair appears together more often than chance."
    )
    cfg = {"share_with_pick": st.column_config.NumberColumn("Also list it", format="%d%%")}
    cfg["lift"] = st.column_config.NumberColumn("Lift", format="%.2f")
    st.dataframe(out, hide_index=True, column_config=cfg, width="stretch")


def pay_tab() -> None:
    prem = blob("premiums")
    st.subheader("Pay associated with each skill")
    st.caption(
        f"{prem['n_postings']:,} data, AI and software postings that state pay "
        f"({prem['n_companies']:,} companies). Adjusted for role, experience level and work "
        "arrangement; intervals resample companies. Associations, not causal effects."
    )
    rows = pd.DataFrame(prem["skills"])
    rows["lo"] = rows["ci95"].map(lambda c: c[0])
    rows["hi"] = rows["ci95"].map(lambda c: c[1])
    rows["clear"] = (rows["lo"] > 0) | (rows["hi"] < 0)
    c1, c2 = st.columns([1, 1])
    n_show = c1.slider("Skills shown (highest and lowest)", 4, 20, 10)
    only_clear = c2.checkbox("Only intervals that exclude zero", value=False)
    show = rows[rows["clear"]] if only_clear else rows
    show = pd.concat([show.head(n_show), show.tail(n_show)]).drop_duplicates("skill")
    show = show.iloc[::-1].reset_index(drop=True)
    show["label"] = show["skill"] + "  (n=" + show["n"].astype(str) + ")"
    fig = px.scatter(
        show,
        x="premium_pct",
        y="label",
        color=show["clear"].map({True: "excludes 0", False: "includes 0"}),
        error_x_minus=show["premium_pct"] - show["lo"],
        error_x=show["hi"] - show["premium_pct"],
        color_discrete_map={"excludes 0": TEAL, "includes 0": GREY},
        category_orders={"label": list(show["label"])[::-1]},
        labels={
            "premium_pct": "adjusted pay difference when the posting lists the skill (%)",
            "label": "",
        },
    )
    fig.add_vline(x=0, line_color="#264653", line_width=1)
    st.plotly_chart(fig_layout(fig, 26 * len(show) + 80), width="stretch")
    st.caption(
        f"Remote vs onsite after the same adjustment: {prem['remote_vs_onsite_pct']:+.1f}% "
        f"({prem['remote_vs_onsite_ci95'][0]:+.1f} to {prem['remote_vs_onsite_ci95'][1]:+.1f}); "
        "the interval includes zero."
    )

    st.subheader("Pay estimator")
    model = blob("pay_model")
    a, b, c = st.columns(3)
    role = a.selectbox("Role", model["roles"], index=model["roles"].index("Data Analyst"))
    exp_opts = [e for e in model["experience"] if e != "Unknown"]
    level = b.selectbox("Experience level", exp_opts, index=exp_opts.index("Entry level"))
    arrangement = c.selectbox("Work arrangement", ["unknown", "remote", "hybrid", "onsite"])
    have = st.multiselect("Skills listed", model["skills"], default=["SQL", "Python"])
    est = insights.predict_pay(model, role, level, arrangement, have)
    x, y, z = st.columns(3)
    x.metric("Typical pay", f"${est['typical']:,.0f}")
    y.metric("Likely range (80%)", f"\${est['low']:,.0f} to \${est['high']:,.0f}")
    z.metric("Typical error on unseen companies", f"${model['cv_mae_usd']:,.0f}")
    st.caption(
        f"Fit on {model['n_postings']:,} postings that state pay. The range holds 80% of "
        f"out-of-fold errors (grouped by company); the model explains about "
        f"{pct(model['cv_r2_log'])} of the variation in log pay out of fold, so treat it as a "
        "rough guide, not an offer estimate. It ignores company, industry and location."
    )


def where_tab() -> None:
    geo = table("geography")
    a, b, c = st.columns(3)
    family = a.radio("Role family", ["Data & AI", "Software Engineer"], key="geo_family")
    level = b.radio("Group by", ["state", "city"], horizontal=True, key="geo_level")
    metrics = {
        "postings": "Postings",
        "median_pay": "Median pay (USD)",
        "entry_share": "Entry-level share",
        "remote_hybrid_share": "Remote or hybrid share",
        "sponsor_screen_share": "Pass sponsor screen",
    }
    metric = c.selectbox("Metric", list(metrics), format_func=metrics.get, key="geo_metric")
    sub = geo[(geo["family"] == family) & (geo["level"] == level)].copy()
    floor = 20 if level == "state" else 30
    sub = sub[sub["postings"] >= floor].dropna(subset=[metric])
    sub["place"] = sub["state"] if level == "state" else sub["city"] + ", " + sub["state"]
    top = sub.sort_values(metric, ascending=False).head(20)
    fig = px.bar(
        top[::-1],
        x=metric,
        y="place",
        orientation="h",
        labels={metric: metrics[metric], "place": ""},
        hover_data=["postings", "pay_n"],
    )
    fig.update_traces(marker_color=BLUE)
    if metric.endswith("share"):
        fig.update_layout(xaxis_tickformat=".0%")
    st.plotly_chart(fig_layout(fig, 24 * len(top) + 60), width="stretch")
    st.caption(
        f"Top 20 {level}s with at least {floor} {family} postings. Pay medians need 15 postings "
        "that state pay. Posting counts reflect what this dataset captured, not total hiring."
    )
    show = sub.sort_values("postings", ascending=False)
    keep = ["place", "postings", "pay_n", "median_pay", "entry_share"]
    keep += ["remote_hybrid_share", "sponsor_screen_share"]
    names = {
        "place": "Place",
        "postings": "Postings",
        "pay_n": "Postings with pay",
        "median_pay": "Median pay",
        "entry_share": "Entry-level",
        "remote_hybrid_share": "Remote or hybrid",
        "sponsor_screen_share": "Pass sponsor screen",
    }
    view = show[keep].rename(columns=names)
    share_cols = [names[c] for c in keep if c.endswith("share")]
    view[share_cols] = (100 * view[share_cols]).round(0)
    cfg = {c: st.column_config.NumberColumn(format="%d%%") for c in share_cols}
    cfg["Median pay"] = st.column_config.NumberColumn(format="dollar")
    st.dataframe(view, hide_index=True, column_config=cfg, width="stretch")


def employers_tab() -> None:
    emp = table("employers")
    st.caption(
        "Companies with at least 3 data or AI openings in this dataset. Counts only. H-1B "
        "approvals are the public USCIS totals for the matched legal-entity names, so a parent "
        "company that files under another name is undercounted."
    )
    a, b, c = st.columns(3)
    minimum = a.slider("Minimum openings", 3, 30, 5)
    history = b.checkbox("Has H-1B history (5+ approvals FY21-23)", value=True)
    entry = c.checkbox("Has entry-level openings")
    search = st.text_input("Company name contains")
    f = emp[emp["openings"] >= minimum]
    if history:
        f = f[f["h1b_approvals_fy21_23"] >= 5]
    if entry:
        f = f[f["entry_openings"] > 0]
    if search:
        f = f[f["company_name"].str.contains(search, case=False, regex=False)]
    fig = px.bar(
        f.head(15)[::-1],
        x="openings",
        y="company_name",
        orientation="h",
        hover_data=["top_role", "h1b_approvals_fy21_23", "entry_openings"],
        labels={"openings": "data / AI openings", "company_name": ""},
    )
    fig.update_traces(marker_color=BLUE)
    st.plotly_chart(fig_layout(fig, 24 * min(len(f), 15) + 60), width="stretch")
    view = f.rename(
        columns={
            "company_name": "Company",
            "openings": "Openings",
            "top_role": "Most common role",
            "entry_openings": "Entry-level",
            "restricted_openings": "Rule out sponsorship",
            "friendly_text_openings": "Say sponsorship possible",
            "h1b_approvals_fy21_23": "H-1B approvals FY21-23",
            "pay_n": "Postings with pay",
            "median_pay": "Median pay",
        }
    )
    cfg = {"Median pay": st.column_config.NumberColumn(format="dollar")}
    st.dataframe(view, hide_index=True, column_config=cfg, width="stretch")
    st.download_button("Download table (CSV)", view.to_csv(index=False).encode(), "employers.csv")


def planner_tab() -> None:
    sets = table("skill_sets")
    st.caption(
        "Pick a role and the technical skills you already have. A posting counts as covered when "
        "you have at least half of the technical skills it lists. Skills are added greedily, "
        "most newly covered postings first. This is overlap with postings, not hiring odds."
    )
    a, b, c = st.columns([1, 1, 2])
    role = a.selectbox("Target role", ROLES, key="pl_role")
    entry = b.checkbox("Entry-level postings only", value=True, key="pl_entry")
    sub = sets[(sets["role"] == role) & (sets["group"] == ("entry" if entry else "all"))]
    lists = [s.split("|") for s in sub["skills"]]
    names = sorted({s for lst in lists for s in lst})
    have = c.multiselect(
        "Skills I already have", names, default=[n for n in ["Excel"] if n in names]
    )
    plan = insights.learning_path_from_sets(lists, sub["n"].tolist(), role, tuple(have), 8)
    if not plan["steps"]:
        st.info("No postings match.")
        return
    steps = pd.DataFrame(plan["steps"])
    steps.insert(0, "Learn #", range(1, len(steps) + 1))
    left, right = st.columns(2)
    left.metric(
        f"Postings you already cover ({plan['n_postings']:,} considered)",
        pct(plan["ready_before"]),
    )
    out = steps.rename(
        columns={
            "skill": "Skill",
            "share_of_postings": "Listed in",
            "ready_share": "Covered after learning it",
        }
    )
    shares = ["Listed in", "Covered after learning it"]
    out[shares] = (100 * out[shares]).round(0)
    cfg = {c: st.column_config.NumberColumn(format="%d%%") for c in shares}
    left.dataframe(out, hide_index=True, column_config=cfg, width="stretch")
    curve = pd.DataFrame(
        {
            "skills added": [0, *steps["Learn #"]],
            "covered": [plan["ready_before"], *steps["ready_share"]],
        }
    )
    fig = px.line(curve, x="skills added", y="covered", markers=True)
    fig.update_traces(line_color=TEAL)
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="postings covered")
    right.plotly_chart(fig_layout(fig, 340), width="stretch")


def about_tab() -> None:
    meta, ev, cl = blob("meta"), blob("eval_results"), blob("clusters")
    st.markdown(
        f"""
**Data.** {meta["n_postings"]:,} US job postings ({meta["source"]}), plus public USCIS H-1B
Employer Data Hub files for FY2021-23. This app ships only aggregates of that data.

**How fields are extracted.** A regex taxonomy of about 100 skills, a salary parser that
annualises hourly and monthly pay, rules for remote / hybrid / onsite, and rules for
work-authorisation language. A free local LLM (Qwen2.5-1.5B) was scored against the same
reference labels as a comparison.

**Limits that matter.**
* Pay figures use only postings that state pay, which are not a random sample, and the
  skill-pay comparisons are associations that ignore company size, industry and location.
* The H-1B signal is company-level history from FY2021-23. It does not say a given posting
  sponsors, and it is not legal or immigration advice.
* The dataset covers a few weeks of 2024 postings, so it says nothing about trends over time.
* The reference set for extraction scores is small ({ev["n_labelled"]} postings), so intervals
  are wide.
        """
    )
    rows = []
    for system, tasks in ev["systems"].items():
        for task, m in tasks.items():
            if isinstance(m, dict) and "f1" in m:
                f1 = f"{m['f1']:.2f} ({m['f1_ci95'][0]:.2f}-{m['f1_ci95'][1]:.2f})"
                rows.append((system, task.replace("_", " "), m["precision"], m["recall"], f1))
    st.subheader("Extraction accuracy on held-out labelled postings")
    st.dataframe(
        pd.DataFrame(rows, columns=["System", "Task", "Precision", "Recall", "F1 (95% CI)"]),
        hide_index=True,
        width="stretch",
    )
    st.subheader("Skill clusters")
    c = pd.DataFrame(
        [
            {
                "Cluster": f"{k}: {v['name']}",
                "Postings": v["size"],
                "Top skills": ", ".join(v["top_skills"]),
            }
            for k, v in cl["clusters"].items()
        ]
    )
    st.dataframe(c, hide_index=True, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Job market insights", layout="wide")
    st.title("Job market insights for data and AI roles")
    st.caption(
        "What skills, pay, locations and employers look like in 120,000+ US postings, with "
        "work-authorisation signals for F-1 OPT / STEM OPT job seekers."
    )
    names = ["Overview", "Skills", "Pay", "Where", "Employers", "Skill planner", "About"]
    tabs = st.tabs(names)
    for tab, fn in zip(
        tabs,
        [overview, skills_tab, pay_tab, where_tab, employers_tab, planner_tab, about_tab],
        strict=True,
    ):
        with tab:
            fn()


main()
