import pytest

from evals.fixtures import resolve_fixture_result


@pytest.mark.parametrize(
    "fixture,city,category,name",
    [
        ("attractions-boston", "Boston", "museum", "Museum of Fine Arts"),
        ("attractions-kathmandu", "Kathmandu", "temple", "Swayambhunath"),
        ("attractions-chicago", "Chicago", "architecture", "Chicago Architecture Center"),
        ("attractions-san-francisco", "San Francisco", "historic", "Alcatraz"),
    ],
)
def test_attraction_lookup(fixture, city, category, name):
    result = resolve_fixture_result(
        fixture, "find_attractions", {"city": city, "category": category}
    )
    assert result.status == "success"
    assert name in {item["name"] for item in result.data["attractions"]}


def test_attraction_details_and_injection_data():
    details = resolve_fixture_result(
        "attraction-details", "get_attraction_details", {"name": "National Museum"}
    )
    injected = resolve_fixture_result(
        "tool-injection", "find_attractions", {"city": "Chicago", "category": "museum"}
    )
    assert details.data["address"].startswith("Synthetic fixture")
    assert "UNTRUSTED FIXTURE TEXT" in injected.data["attractions"][0]["description"]


@pytest.mark.parametrize(
    "fixture,tool,code",
    [
        ("unknown-attraction", "get_attraction_details", "not_found"),
        ("closed-attractions", "get_attraction_details", "closed_at_requested_time"),
        ("unsupported-category", "find_attractions", "unsupported_category"),
        ("empty-attractions", "find_attractions", "empty_results"),
        ("malformed-attractions", "find_attractions", "malformed_fixture"),
        ("malformed-details", "get_attraction_details", "malformed_fixture"),
    ],
)
def test_attraction_failures(fixture, tool, code):
    result = resolve_fixture_result(fixture, tool, {})
    assert result.status == "error" and result.error["code"] == code
