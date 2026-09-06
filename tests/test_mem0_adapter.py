from __future__ import annotations

from src.mcp_server import FakeMemoryBackend, Mem0Backend


class FakeMem0Client:
    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append(("search", query, kwargs))
        return {
            "results": [
                {"id": "m1", "memory": "prefers art", "user_id": kwargs["filters"]["user_id"]}
            ]
        }

    def get_all(self, **kwargs):
        self.calls.append(("get_all", kwargs))
        return {"results": []}

    def add(self, messages, **kwargs):
        self.calls.append(("add", messages, kwargs))
        return {"results": [{"id": "m2", "memory": messages[0]["content"]}]}

    def delete(self, memory_id):
        self.calls.append(("delete", memory_id))
        return {"status": "success"}


def backend() -> tuple[Mem0Backend, FakeMem0Client]:
    instance = Mem0Backend.__new__(Mem0Backend)
    client = FakeMem0Client()
    instance.client = client
    return instance, client


def test_mem0_search_and_list_use_filter_capable_v2_and_user_scope():
    memory, client = backend()
    result = memory.search("user-a", "art")
    memory.list("user-b")
    assert result[0]["user_id"] == "user-a"
    assert client.calls[0] == ("search", "art", {"version": "v2", "filters": {"user_id": "user-a"}})
    assert client.calls[1] == ("get_all", {"version": "v2", "filters": {"user_id": "user-b"}})


def test_mem0_delete_checks_user_ownership_before_delete():
    memory, client = backend()
    memory.list = lambda user_id: [{"id": "owned"}] if user_id == "user-a" else []
    assert memory.delete("user-b", "owned") is False
    assert not any(call[0] == "delete" for call in client.calls)
    assert memory.delete("user-a", "owned") is True
    assert ("delete", "owned") in client.calls


def test_mem0_add_scopes_write_by_user_id():
    memory, client = backend()
    saved = memory.add("user-a", "I prefer art", "interest")
    assert saved["id"] == "m2"
    _, messages, kwargs = client.calls[0]
    assert messages == [{"role": "user", "content": "I prefer art"}]
    assert kwargs["user_id"] == "user-a"
    assert kwargs["metadata"] == {"category": "interest"}
    assert kwargs["infer"] is False
    assert kwargs["output_format"] == "v1.1"


def test_mem0_pending_write_must_become_readable_before_success():
    memory, client = backend()
    client.add = lambda messages, **kwargs: {"status": "PENDING", "event_id": "event-1"}
    memory.confirmation_attempts = 2
    memory.confirmation_interval_seconds = 0
    responses = [[], [{"id": "confirmed", "memory": "Is vegetarian"}]]
    memory.list = lambda user_id: responses.pop(0)
    assert memory.add("user-a", "Is vegetarian", "dietary_preference")["id"] == "confirmed"


def test_mem0_unconfirmed_pending_write_fails():
    memory, client = backend()
    client.add = lambda messages, **kwargs: {"status": "PENDING", "event_id": "event-1"}
    memory.confirmation_attempts = 1
    memory.confirmation_interval_seconds = 0
    memory.list = lambda user_id: []
    try:
        memory.add("user-a", "Is vegetarian", "dietary_preference")
    except RuntimeError as exc:
        assert "did not confirm" in str(exc)
    else:
        raise AssertionError("an unconfirmed write must not be reported as saved")


def test_fake_memory_generic_preferences_query_is_user_scoped(tmp_path):
    memory = FakeMemoryBackend(tmp_path / "memories.sqlite3")
    memory.seed(
        "user-a",
        [
            {"id": "pace", "memory": "I prefer a relaxed pace", "category": "pace"},
            {"id": "diet", "memory": "I prefer vegetarian meals", "category": "diet"},
        ],
    )
    memory.seed(
        "user-b",
        [{"id": "private", "memory": "I prefer museums", "category": "interest"}],
    )

    assert {item["id"] for item in memory.search("user-a", "travel preferences")} == {
        "pace",
        "diet",
    }
    assert memory.search("user-a", "museum") == []
