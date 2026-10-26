import numpy as np
import pandas as pd
import pytest

from jobinsights.cluster import choose_k, name_clusters, skill_matrix
from jobinsights.evaluate import boot_ci, prf, salary_counts, skill_counts
from jobinsights.llm import clean_skills, parse_output, to_pay
from jobinsights.taxonomy import SKILL_NAMES


def test_skill_matrix_shape_and_values():
    X = skill_matrix(pd.Series([["Python", "SQL"], []]))
    assert X.shape == (2, len(SKILL_NAMES))
    assert X[0].sum() == 2 and X[1].sum() == 0


def test_choose_k_prefers_stable_high_silhouette():
    table = pd.DataFrame(
        {"k": [4, 6, 8], "silhouette": [0.30, 0.25, 0.20], "stability_ari": [0.5, 0.8, 0.9]}
    )
    assert choose_k(table) == 6  # k=4 has the best silhouette but is not stable


def test_choose_k_falls_back_to_most_stable():
    table = pd.DataFrame({"k": [4, 6], "silhouette": [0.3, 0.1], "stability_ari": [0.4, 0.6]})
    assert choose_k(table) == 6


def test_name_clusters_uses_distinctive_skills():
    idx = {s: i for i, s in enumerate(SKILL_NAMES)}
    X = np.zeros((6, len(SKILL_NAMES)), dtype=np.float32)
    X[:3, idx["Python"]] = 1  # cluster 0 is Python-heavy
    X[:3, idx["SQL"]] = 1
    X[3:, idx["Accounting"]] = 1
    X[:, idx["Excel"]] = 1  # ubiquitous, so not distinctive
    labels = np.array([0, 0, 0, 1, 1, 1])
    titles = pd.Series(["Data Analyst"] * 3 + ["Accountant"] * 3)
    names = name_clusters(X, labels, titles)
    assert names[0]["top_skills"][0] in ("Python", "SQL")
    assert names[1]["top_skills"][0] == "Accounting"
    assert names[0]["top_role_family"] == "Data Analyst"


class TestMetrics:
    def test_prf(self):
        m = prf(tp=8, fp=2, fn=2)
        assert (m.p, m.r) == (0.8, 0.8)
        assert m.f1 == pytest.approx(0.8)

    def test_prf_empty_is_nan_not_zero(self):
        assert np.isnan(prf(0, 0, 0).f1)

    def test_skill_counts(self):
        c = skill_counts([{"a", "b"}, {"c"}], [{"a"}, {"c", "d"}])
        assert c.tolist() == [[1, 1, 0], [1, 0, 1]]

    def test_salary_counts_strict_and_tolerance(self):
        pred = [(100_000, 120_000), (100_000, 120_000), None, (1, 2)]
        gold = [(100_500, 120_500), (90_000, 120_000), (50_000, 60_000), None]
        assert salary_counts(pred, gold).tolist() == [[1, 0, 0], [0, 1, 1], [0, 0, 1], [0, 1, 0]]

    def test_bootstrap_ci_brackets_point_estimate(self):
        counts = np.array([[1, 0, 0]] * 40 + [[0, 1, 1]] * 10)
        point, lo, hi = boot_ci(counts)
        assert lo.f1 <= point.f1 <= hi.f1


class TestLlmOutput:
    def test_parse_valid_with_chatter_around_json(self):
        out = parse_output(
            'Sure!\n{"remote": "hybrid", "salary": {"present": true, "min": "$40/hr", '
            '"max": 50, "period": "hour"}, "skills": ["SQL", 3]}'
        )
        assert out["remote"] == "hybrid"
        assert out["salary"] == {"present": True, "min": 40.0, "max": 50.0, "period": "hourly"}
        assert out["skills"] == ["SQL"]

    @pytest.mark.parametrize("bad", ["no json", '{"remote": "maybe"}', '{"remote": "remote" '])
    def test_parse_invalid_returns_none(self, bad):
        assert parse_output(bad) is None

    def test_to_pay_annualises(self):
        assert to_pay({"present": True, "min": 20, "max": 30, "period": "hourly"}) == (
            20 * 2080,
            30 * 2080,
        )
        assert to_pay({"present": True, "min": 20, "max": 30, "period": None}) == (None, None)
        assert to_pay({"present": False, "min": None, "max": None, "period": None}) == (None, None)

    def test_clean_skills_maps_to_taxonomy_and_drops_unknown(self):
        assert clean_skills(["power bi", "PowerBI", "POS Systems", "Python"]) == [
            "Power BI",
            "Python",
        ]
