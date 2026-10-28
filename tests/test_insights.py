import numpy as np
import pandas as pd
import pytest

from jobinsights import insights


def _frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic postings where pay really does depend on role, level and the ML skill."""
    rng = np.random.default_rng(seed)
    role = rng.choice(["Data Analyst", "Data Scientist", "Software Engineer", "Other"], n)
    level = rng.choice(["Entry level", "Mid-Senior level"], n)
    ml = rng.random(n) < 0.4
    sql = rng.random(n) < 0.6
    base = np.where(role == "Software Engineer", 130_000, 95_000)
    base = base * np.where(level == "Entry level", 0.8, 1.0) * np.where(ml, 1.2, 1.0)
    pay = base * np.exp(rng.normal(0, 0.05, n))
    skills = [
        [s for s, on in (("Machine Learning", m), ("SQL", q), ("Communication", True)) if on]
        for m, q in zip(ml, sql, strict=True)
    ]
    return pd.DataFrame(
        {
            "job_id": np.arange(n),
            "company_name": [f"co{i % 150}" for i in range(n)],
            "role": role,
            "formatted_experience_level": level,
            "remote_rules": rng.choice(["remote", "onsite", "unknown"], n),
            "visa_rules": rng.choice(["friendly", "restricted", "not_stated"], n),
            "contract_rules": rng.random(n) < 0.2,
            "h1b_history": rng.random(n) < 0.4,
            "h1b_total": rng.integers(0, 500, n),
            "salary_mid": pay,
            "skills_rules": skills,
        }
    )


def test_hard_skills_drops_soft_and_business_skills():
    assert insights.hard_skills(["SQL", "Communication", "Recruiting", "Python"]) == [
        "SQL",
        "Python",
    ]


def test_skill_matrix_ignores_unknown_names():
    m = insights.skill_matrix(pd.Series([["SQL", "Nope"], []]), ["SQL", "Python"])
    assert m.tolist() == [[True, False], [False, False]]


def test_skill_premium_recovers_planted_effect_and_ignores_null_skill():
    df = _frame()
    res = insights.skill_premiums(df, min_n=30, n_boot=60)
    by = {r["skill"]: r for r in res["skills"]}
    assert 15 < by["Machine Learning"]["premium_pct"] < 25  # planted +20%
    lo, hi = by["Machine Learning"]["ci95"]
    assert lo > 0 and lo < by["Machine Learning"]["premium_pct"] < hi
    assert by["SQL"]["ci95"][0] < 0 < by["SQL"]["ci95"][1]  # no real effect
    assert "Communication" not in by
    assert res["n_postings"] == (df["role"] != "Other").sum()  # only tech roles are modelled


def test_sponsor_landscape_shares_are_consistent():
    rows = insights.sponsor_landscape(_frame(), roles=("Data Analyst",))
    assert len(rows) == 1
    r = rows[0]
    assert 0 <= r["shortlist"] <= 1
    assert r["shortlist"] <= 1 - r["restricted"] + 1e-9


def test_top_employers_exclude_restricted_postings():
    df = _frame()
    df.loc[df["company_name"] == "co0", "visa_rules"] = "restricted"
    names = {e["company_name"] for e in insights.top_sponsor_employers(df, roles=("Data Analyst",))}
    assert "co0" not in names


def test_learning_path_is_greedy_monotone_and_skips_known_skills():
    df = _frame()
    # every posting asks for three technical skills, SQL most often
    df["skills_rules"] = [["SQL", "Python", "Tableau"] if i % 2 else ["SQL", "AWS", "R"]
                          for i in range(len(df))]  # fmt: skip
    lp = insights.learning_path(df, "Data Analyst", levels=None, have=("SQL",), steps=3)
    got = [s["skill"] for s in lp["steps"]]
    assert "SQL" not in got
    shares = [s["ready_share"] for s in lp["steps"]]
    assert shares == sorted(shares)
    assert lp["ready_before"] == 0.0  # 1 of 3 skills is below the 50% bar
    assert shares[-1] == pytest.approx(1.0)


def test_learning_path_empty_when_no_postings_match():
    lp = insights.learning_path(_frame(), "Data Engineer")
    assert lp["steps"] == [] and lp["n_postings"] == 0


def test_findings_markdown_fills_numbers_from_results():
    res = insights.compute(_frame())
    text = insights.findings_markdown(res)
    assert "Machine Learning" in text
    assert "{" not in text and "nan" not in text.lower()
