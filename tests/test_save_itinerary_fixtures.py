from evals.fixtures import save_itinerary_fixture


def test_save_validation_and_existing_file(tmp_path):
    assert (
        save_itinerary_fixture(tmp_path, "u", "plan.md", "plan", False).error["code"]
        == "approval_required"
    )
    assert (
        save_itinerary_fixture(tmp_path, "u", "../escape.md", "plan", True).error["code"]
        == "unsafe_path"
    )
    assert (
        save_itinerary_fixture(tmp_path, "u", "/tmp/plan.md", "plan", True).error["code"]
        == "unsafe_path"
    )
    first = save_itinerary_fixture(tmp_path, "u", "plan.md", "plan", True)
    second = save_itinerary_fixture(tmp_path, "u", "plan.md", "plan", True)
    overwrite = save_itinerary_fixture(tmp_path, "u", "plan.md", "new", True, True)
    assert first.status == "success"
    assert second.error["code"] == "overwrite_not_approved"
    assert overwrite.status == "success"
