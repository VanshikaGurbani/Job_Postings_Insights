import math

import pytest

from jobinsights.remote import detect_remote, legacy_remote
from jobinsights.salary import normalize_structured, parse_salary_text
from jobinsights.taxonomy import extract_skills, legacy_skills, normalize_skill


class TestSkills:
    def test_word_boundaries(self):
        assert extract_skills("Strong knowledge of the laws and excellent judgement") == []

    def test_r_needs_programming_context(self):
        assert "R" not in extract_skills("Join our R&D team. Part R.")
        assert "R" in extract_skills("Experience with Python, R and SQL")
        assert "R" in extract_skills("Statistical computing in R, Python")
        assert "R" in extract_skills("R programming and RStudio")

    def test_legacy_substring_matcher_is_wrong_in_the_known_ways(self):
        text = "We value excellent communication; our laws apply."
        assert "R" in legacy_skills(text)
        assert "Excel" in legacy_skills(text)
        assert extract_skills(text) == []

    def test_excel_verb_vs_tool(self):
        assert "Excel" not in extract_skills("You will excel in a fast-paced team")
        assert "Excel" in extract_skills("Advanced Excel and PowerPoint skills")

    def test_symbols_and_aliases(self):
        found = extract_skills("C++, C#, Node.js, PostgreSQL and Amazon Web Services")
        assert {"C++", "C#", "JavaScript", "PostgreSQL", "AWS"} <= set(found)

    def test_sql_server_is_not_plain_sql(self):
        found = extract_skills("Experience with SQL Server")
        assert "SQL Server" in found and "SQL" not in found

    @pytest.mark.parametrize(
        ("raw", "canon"),
        [("power bi", "Power BI"), ("PostgreSQL", "PostgreSQL"), ("ml", "Machine Learning"),
         ("MS Excel", "Excel"), ("nonsense skill", None), ("", None)],
    )  # fmt: skip
    def test_normalize_skill(self, raw, canon):
        assert normalize_skill(raw) == canon


class TestRemote:
    @pytest.mark.parametrize(
        ("text", "label"),
        [
            ("This is a fully remote position.", "remote"),
            ("Work from home with occasional travel", "remote"),
            ("Hybrid schedule, 3 days in office", "hybrid"),
            ("This role is not remote and requires on-site presence", "onsite"),
            ("Must work on-site at our plant", "onsite"),
            ("Remote sensing analyst at our lab", "unknown"),
            ("Manage remote teams across the country", "unknown"),
            ("Hybrid cloud architecture experience", "unknown"),
            ("Lovely team and great benefits", "unknown"),
        ],
    )
    def test_detect_remote(self, text, label):
        assert detect_remote(text) == label

    def test_legacy_counts_hybrid_as_remote(self):
        assert legacy_remote("hybrid role") == "Yes"
        assert detect_remote("hybrid role") == "hybrid"


class TestSalaryText:
    def test_hourly_range(self):
        pay = parse_salary_text("Pay: $18.00 - $21.00 per hour plus benefits")
        assert (pay.low, pay.high, pay.period) == (18.0, 21.0, "hourly")
        assert pay.annual_low == pytest.approx(18 * 2080)

    def test_yearly_range_with_commas_and_k(self):
        assert parse_salary_text("$60,000 - $88,000 a year").annual_high == 88000
        pay = parse_salary_text("pay is $85-120K per year")
        assert (pay.low, pay.high) == (85000, 120000)

    def test_period_inferred_from_magnitude(self):
        assert parse_salary_text("Compensation: $95,000 - $110,000").period == "yearly"

    def test_ignores_bonus_and_unrelated_money(self):
        assert parse_salary_text("sign-on bonus of $5,000 and tuition of $5,250") is None
        assert parse_salary_text("We raised $50 million in funding") is None

    def test_no_salary(self):
        assert parse_salary_text("Competitive pay and benefits") is None


class TestSalaryStructured:
    def test_yearly_passthrough(self):
        pay = normalize_structured(60000, 80000, "YEARLY")
        assert (pay.annual_low, pay.annual_high) == (60000, 80000)

    def test_hourly_annualised(self):
        pay = normalize_structured(20, 30, "HOURLY")
        assert pay.annual_low == 20 * 2080

    def test_mislabelled_period_is_repaired(self):
        pay = normalize_structured(150000, 180000, "HOURLY")
        assert pay.period == "yearly"

    def test_missing_or_implausible(self):
        assert normalize_structured(math.nan, math.nan, "YEARLY") is None
        assert normalize_structured(3, 5, "YEARLY") is None

    def test_median_only(self):
        pay = normalize_structured(None, None, "YEARLY", med_salary=70000)
        assert pay.annual_mid == 70000
