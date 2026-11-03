"""Tests for the phone API. Skipped unless the `api` extra is installed."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from foodtimizer.api.app import create_app  # noqa: E402
from foodtimizer.api.tokens import issue_token  # noqa: E402
from foodtimizer.config import problem_to_dict  # noqa: E402
from foodtimizer.identity import namespace_for_subject  # noqa: E402
from foodtimizer.model import (  # noqa: E402
    Ingredient,
    LibraryMeal,
    MacroTarget,
    Problem,
)
from foodtimizer.storage import FileStorage  # noqa: E402

_SECRET = "test-secret-at-least-16"


def _problem() -> Problem:
    return Problem(
        ingredients=(
            Ingredient(
                name="eggs",
                macros={"kcal": 155.0, "protein": 13.0, "carbs": 1.1, "fat": 11.0},
            ),
            Ingredient(
                name="banana",
                macros={"kcal": 89.0, "protein": 1.1, "carbs": 23.0, "fat": 0.3},
            ),
        ),
        targets=(
            MacroTarget(name="kcal", value=2000.0, weight=50.0),
            MacroTarget(name="protein", value=150.0, weight=10.0),
        ),
        meal_library=(LibraryMeal(name="m", tag="snack", ingredients=("eggs",)),),
    )


def _app(tmp_path: Path, *, dev_auth: bool = True):
    seed = problem_to_dict(_problem())

    def factory(namespace: str) -> FileStorage:
        storage = FileStorage(tmp_path / "day.yaml", tmp_path / "logs", namespace=namespace)
        if storage.load_config() is None:
            storage.save_config(seed)
        return storage

    return create_app(
        secret=_SECRET,
        dev_auth=dev_auth,
        storage_factory=factory,
        token_ttl_seconds=3600,
    )


def _token(client: TestClient, sub: str) -> str:
    response = client.post("/v1/auth/dev-login", json={"sub": sub, "email": f"{sub}@example.com"})
    assert response.status_code == 200
    body = response.json()
    assert body["namespace"] == namespace_for_subject(sub)
    return body["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_health_needs_no_token(tmp_path):
    client = TestClient(_app(tmp_path))
    assert client.get("/health").json() == {"ok": True}


def test_dev_login_is_hidden_when_disabled(tmp_path):
    client = TestClient(_app(tmp_path, dev_auth=False))
    response = client.post("/v1/auth/dev-login", json={"sub": "alice"})
    assert response.status_code == 404


def test_missing_and_bad_tokens_are_rejected(tmp_path):
    client = TestClient(_app(tmp_path))
    assert client.get("/v1/me").status_code == 401
    assert client.get("/v1/me", headers=_auth("not-a-token")).status_code == 401
    expired = issue_token(_SECRET, "alice", ttl_seconds=-60)
    assert client.get("/v1/me", headers=_auth(expired)).status_code == 401


def test_logging_food_is_isolated_per_account(tmp_path):
    client = TestClient(_app(tmp_path))
    alice = _token(client, "alice")
    bob = _token(client, "bob")

    created = client.post(
        "/v1/days/2026-10-16/entries",
        headers=_auth(alice),
        json={"ingredient": "eggs", "grams": 50, "slot": "breakfast"},
    )
    assert created.status_code == 200
    body = created.json()
    assert body["entries"][0]["ingredient"] == "eggs"
    assert body["entries"][0]["grams"] == 50
    assert body["totals"]["kcal"] == pytest.approx(77.5)
    assert body["totals"]["protein"] == pytest.approx(6.5)

    bob_day = client.get("/v1/days/2026-10-16", headers=_auth(bob))
    assert bob_day.status_code == 200
    assert bob_day.json()["entries"] == []
    assert bob_day.json()["totals"]["kcal"] == 0

    unknown = client.post(
        "/v1/days/2026-10-16/entries",
        headers=_auth(alice),
        json={"ingredient": "not-a-food", "grams": 10},
    )
    assert unknown.status_code == 400

    removed = client.delete("/v1/days/2026-10-16/entries/0", headers=_auth(alice))
    assert removed.status_code == 200
    assert removed.json()["entries"] == []
    missing = client.delete("/v1/days/2026-10-16/entries/3", headers=_auth(alice))
    assert missing.status_code == 404


def test_weight_history_skips_days_without_a_weigh_in(tmp_path):
    client = TestClient(_app(tmp_path))
    token = _token(client, "alice")
    first = client.put(
        "/v1/days/2026-10-01/weight",
        headers=_auth(token),
        json={"kg": 80.0},
    )
    assert first.status_code == 200
    assert first.json()["weight_kg"] == 80.0
    second = client.put(
        "/v1/days/2026-10-15/weight",
        headers=_auth(token),
        json={"kg": 78.4},
    )
    assert second.status_code == 200

    history = client.get("/v1/weight", headers=_auth(token)).json()["points"]
    assert history == [
        {"date": "2026-10-01", "kg": 80.0},
        {"date": "2026-10-15", "kg": 78.4},
    ]

    cleared = client.put(
        "/v1/days/2026-10-01/weight",
        headers=_auth(token),
        json={"kg": None},
    )
    assert cleared.status_code == 200
    assert cleared.json()["weight_kg"] is None
    remaining = client.get("/v1/weight", headers=_auth(token)).json()["points"]
    assert remaining == [{"date": "2026-10-15", "kg": 78.4}]

    rejected = client.put(
        "/v1/days/2026-10-16/weight",
        headers=_auth(token),
        json={"kg": 0},
    )
    assert rejected.status_code == 400


def test_goals_round_trip_preserves_priority_and_refuses_the_last_goal(tmp_path):
    client = TestClient(_app(tmp_path))
    token = _token(client, "alice")

    updated = client.put(
        "/v1/goals",
        headers=_auth(token),
        json={"goals": [{"name": "kcal", "value": 1800}, {"name": "protein", "value": 0}]},
    )
    assert updated.status_code == 200
    goals = {g["name"]: g for g in updated.json()["goals"]}
    assert set(goals) == {"kcal"}
    assert goals["kcal"]["value"] == 1800
    assert goals["kcal"]["weight"] == 50.0

    refused = client.put(
        "/v1/goals",
        headers=_auth(token),
        json={"goals": [{"name": "kcal", "value": 0}]},
    )
    assert refused.status_code == 400
    still = client.get("/v1/goals", headers=_auth(token)).json()["goals"]
    assert still[0]["value"] == 1800


def test_ingredient_search(tmp_path):
    client = TestClient(_app(tmp_path))
    token = _token(client, "alice")
    found = client.get("/v1/ingredients", headers=_auth(token), params={"q": "ban"})
    assert found.status_code == 200
    names = [row["name"] for row in found.json()["ingredients"]]
    assert names == ["banana"]
    assert found.json()["ingredients"][0]["macros"]["kcal"] == 89.0


def test_api_does_not_replace_the_process_wide_storage(tmp_path, monkeypatch):
    from foodtimizer import storage as store

    sentinel = object()
    monkeypatch.setattr(store, "_active", sentinel)
    client = TestClient(_app(tmp_path))
    token = _token(client, "alice")
    assert client.get("/v1/me", headers=_auth(token)).status_code == 200
    assert store._active is sentinel


def test_cloud_requirements_do_not_install_the_api():
    text = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(encoding="utf-8")
    installed = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert installed[0] == "-e .[cloud]"
    assert "fastapi" not in text.lower()
    assert "uvicorn" not in text.lower()


def test_short_secret_is_rejected(tmp_path):
    with pytest.raises(RuntimeError):
        create_app(
            secret="too-short",
            dev_auth=True,
            storage_factory=lambda namespace: FileStorage(
                tmp_path / "day.yaml", tmp_path / "logs", namespace=namespace
            ),
            token_ttl_seconds=60,
        )


def test_logged_date_matches_the_path(tmp_path):
    """The date in the URL is the date that gets stored."""
    client = TestClient(_app(tmp_path))
    token = _token(client, "alice")
    created = client.post(
        "/v1/days/2026-01-02/entries",
        headers=_auth(token),
        json={"ingredient": "banana", "grams": 100, "slot": ""},
    )
    assert created.status_code == 200
    loaded = client.get("/v1/days/2026-01-02", headers=_auth(token)).json()
    assert loaded["date"] == date(2026, 1, 2).isoformat()
    assert loaded["entries"][0]["slot"] is None
    assert loaded["totals"]["kcal"] == pytest.approx(89.0)
