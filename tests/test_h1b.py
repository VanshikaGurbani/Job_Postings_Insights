import pandas as pd

from jobinsights.h1b import Matcher, normalize_company


def test_normalize_company():
    assert normalize_company("Amazon.com Services LLC") == "AMAZON COM SERVICES"
    assert normalize_company("The Boeing Company") == "BOEING"
    assert normalize_company("CROFT & Associates") == "CROFT AND ASSOCIATES"
    assert normalize_company(None) == ""


def _matcher():
    emp = pd.DataFrame(
        {
            "employer_norm": [
                "GOOGLE",
                "AMAZON COM SERVICES",
                "AMAZON WEB SERVICES",
                "ACME ROBOTICS",
            ]
            + [f"PREMIER {i}" for i in range(30)],
            "initial": [10, 20, 5, 1] + [1] * 30,
            "continuing": [5, 30, 5, 0] + [0] * 30,
        }
    )
    return Matcher(emp)


def test_exact_match_after_normalisation():
    m = _matcher().match("Google LLC")
    assert m["match_type"] == "exact" and m["h1b_initial"] == 10


def test_prefix_match_aggregates_entities():
    m = _matcher().match("Amazon")
    assert m["match_type"] == "prefix"
    assert (m["h1b_employers"], m["h1b_initial"], m["h1b_continuing"]) == (2, 25, 35)


def test_generic_or_ambiguous_names_do_not_match():
    matcher = _matcher()
    assert matcher.match("Premier") is None  # generic word, and 30 employers share the prefix
    assert matcher.match("Acme Foods") is None  # no employer starts with this name
    assert matcher.match("AB") is None
