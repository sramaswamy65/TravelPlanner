import pytest

from evals.fixtures import resolve_fixture_result


@pytest.mark.parametrize("city", ["Boston", "Chicago", "San Francisco", "Kathmandu"])
def test_dietary_filter_and_limit(city):
    fixture = "restaurants-" + city.casefold().replace(" ", "-")
    result = resolve_fixture_result(
        fixture, "search_restaurants", {"city": city, "dietary_preference": "vegan", "limit": 100}
    )
    assert result.status == "success" and result.data["applied_limit"] == 10
    assert all("vegan" in item["dietary_suitability"] for item in result.data["restaurants"])


@pytest.mark.parametrize(
    "fixture,code",
    [("restaurant-timeout", "timeout"), ("unsupported-restaurants", "unsupported_city")],
)
def test_restaurant_failures(fixture, code):
    result = resolve_fixture_result(fixture, "search_restaurants", {})
    assert result.status == "error" and result.error["code"] == code
