import pandas as pd

from jobinsights.data import MIN_DESCRIPTION_CHARS, clean_postings, clean_text


def test_clean_text_collapses_whitespace_and_handles_nan():
    assert clean_text("a \n\n b\t\tc ") == "a b c"
    assert clean_text(float("nan")) == ""


def test_clean_postings_drops_duplicates_and_short_descriptions():
    long = "x" * MIN_DESCRIPTION_CHARS
    raw = pd.DataFrame(
        {
            "job_id": [1, 1, 2, 3],
            "title": ["A", "A", "B", "C"],
            "description": [long, long, "short", long + "  \n more"],
            "extra": [0, 0, 0, 0],
        }
    )
    out = clean_postings(raw)
    assert out["job_id"].tolist() == [1, 3]
    assert "extra" not in out.columns
    assert "\n" not in out["description"].iloc[1]


def test_extract_rules_end_to_end_on_synthetic_postings():
    from jobinsights.extract import extract_rules

    clean = pd.DataFrame(
        {
            "job_id": [1, 2],
            "title": ["Data Analyst", "Cashier"],
            "description": [
                "Hybrid role. Pay: $70,000 - $85,000 per year. Python, SQL and Tableau required.",
                "Part time. $15.50 per hour. On-site at our store. Excellent customer service.",
            ],
            "min_salary": [None, 15.5],
            "max_salary": [None, 15.5],
            "med_salary": [None, None],
            "pay_period": [None, "HOURLY"],
        }
    )
    out = extract_rules(clean).set_index("job_id")
    assert {"Python", "SQL", "Tableau"} <= set(out.loc[1, "skills_rules"])
    assert out.loc[1, "remote_rules"] == "hybrid"
    assert (out.loc[1, "salary_annual_low"], out.loc[1, "salary_source"]) == (70000, "text")
    assert out.loc[2, "salary_source"] == "structured"
    assert out.loc[2, "salary_annual_low"] == 15.5 * 2080
    assert out.loc[2, "remote_rules"] == "onsite"
