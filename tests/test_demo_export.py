import numpy as np
import pandas as pd
import pytest

from jobinsights import demo_export, insights
from test_insights import _frame


def _full_frame() -> pd.DataFrame:
    df = _frame(900, seed=3)
    rng = np.random.default_rng(3)
    df["state"] = rng.choice(["NY", "CA", "TX"], len(df))
    df["location"] = df["state"].map(
        {"NY": "New York, NY", "CA": "San Jose, CA", "TX": "Austin, TX"}
    )
    df["title"] = "should never be exported"
    return df


def test_aggregates_contain_no_posting_level_columns():
    frames = demo_export.build(_full_frame())
    banned = {"title", "description", "job_id", "job_posting_url", "location"}
    for name, frame in frames.items():
        assert not banned & set(frame.columns), name


def test_pay_medians_are_suppressed_below_the_minimum_count():
    df = _full_frame()
    df.loc[df["state"] == "TX", "salary_mid"] = np.nan
    df.loc[df.index[:5], "salary_mid"] = 100_000.0
    geo = demo_export.geography(df)
    tiny = geo[(geo["pay_n"] > 0) & (geo["pay_n"] < demo_export.MIN_PAY_N)]
    assert tiny["median_pay"].isna().all()


def test_employers_only_lists_companies_with_enough_openings():
    emp = demo_export.employers(_full_frame())
    assert (emp["openings"] >= demo_export.MIN_COMPANY_OPENINGS).all()


def test_cooccurrence_lift_is_symmetric_definition():
    co = demo_export.cooccurrence(_full_frame(), min_skill_n=20)
    row = co[co["role"] == "Data Analyst"].iloc[0]
    expected = row["both"] * row["n_postings"] / (row["n_a"] * row["n_b"])
    assert row["lift"] == pytest.approx(expected)


def test_planner_on_published_skill_sets_matches_planner_on_postings():
    df = _full_frame()
    direct = insights.learning_path(df, "Data Analyst", levels=insights.ENTRY_LEVELS)
    sets = demo_export.skill_sets(df)
    sub = sets[(sets["role"] == "Data Analyst") & (sets["group"] == "entry")]
    from_sets = insights.learning_path_from_sets(
        [s.split("|") for s in sub["skills"]], sub["n"].tolist(), "Data Analyst"
    )
    assert from_sets["n_postings"] == direct["n_postings"]
    assert [s["skill"] for s in from_sets["steps"]] == [s["skill"] for s in direct["steps"]]
    for a, b in zip(from_sets["steps"], direct["steps"], strict=True):
        assert a["ready_share"] == pytest.approx(b["ready_share"])


def test_pay_model_orders_skill_and_level_effects_and_band_contains_estimate():
    df = _full_frame()
    model = insights.fit_pay_model(df, min_n=30)
    with_ml = insights.predict_pay(
        model, "Data Analyst", "Entry level", "onsite", ["Machine Learning"]
    )
    without = insights.predict_pay(model, "Data Analyst", "Entry level", "onsite", [])
    senior = insights.predict_pay(model, "Data Analyst", "Mid-Senior level", "onsite", [])
    assert with_ml["typical"] > without["typical"]
    assert senior["typical"] > without["typical"]
    assert with_ml["low"] < with_ml["typical"] < with_ml["high"]
