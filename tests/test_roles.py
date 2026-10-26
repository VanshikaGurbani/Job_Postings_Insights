import pytest

from jobinsights.roles import role_family, state_of


@pytest.mark.parametrize(
    ("title", "role"),
    [
        ("Senior Machine Learning Engineer", "AI/ML Engineer"),
        ("Data Scientist II", "Data Scientist"),
        ("Data Engineer - $175k+", "Data Engineer"),
        ("Business Intelligence Analyst", "Data Analyst"),
        ("Sr. Java Developer", "Software Engineer"),
        ("Registered Nurse", "Other"),
    ],
)
def test_role_family(title, role):
    assert role_family(title) == role


def test_state_of():
    assert state_of("New York, NY") == "NY"
    assert state_of("United States") is None
    assert state_of("Paris, FR") is None
