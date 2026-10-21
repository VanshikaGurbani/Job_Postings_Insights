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
