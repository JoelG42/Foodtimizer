"""FastAPI application for the daily tracker.

Each request opens its own storage handle for that account's namespace.
Nothing here calls :func:`foodtimizer.storage.configure`, so this process
cannot replace the backend the Streamlit app is using.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

from ..config import _problem_from_dict, load_problem, problem_to_dict
from ..editor import delete_target, upsert_target
from ..identity import namespace_for_subject
from ..model import MacroTarget, Problem
from ..storage import FileStorage, SqlStorage, Storage
from ..tracker import (
    DayLog,
    compute_totals,
    daylog_from_data,
    daylog_to_data,
    make_entry,
    unknown_ingredients,
)
from .tokens import InvalidToken, issue_token, read_token

try:
    from fastapi import Depends, FastAPI, HTTPException
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel, Field, field_validator
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The phone API is not installed. Run: pip install -e \".[api]\""
    ) from exc


_bearer = HTTPBearer(auto_error=False)
_MIN_SECRET_LENGTH = 16
_DEFAULT_TOKEN_DAYS = 30


@dataclass(frozen=True)
class _Settings:
    secret: str
    dev_auth: bool
    token_ttl_seconds: int
    storage_factory: Callable[[str], Storage]
    seed_config: Path | None


@dataclass(frozen=True)
class _User:
    sub: str
    email: str | None
    namespace: str


class DevLogin(BaseModel):
    sub: str
    email: str | None = None

    @field_validator("sub")
    @classmethod
    def _sub_present(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("sub is required")
        return value


class EntryCreate(BaseModel):
    ingredient: str
    grams: float = Field(gt=0)
    slot: str | None = None
    note: str | None = None

    @field_validator("ingredient")
    @classmethod
    def _ingredient_present(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("ingredient is required")
        return value

    @field_validator("slot", "note")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class WeightUpdate(BaseModel):
    kg: float | None


class GoalUpdate(BaseModel):
    name: str
    value: float

    @field_validator("name")
    @classmethod
    def _name_present(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("goal name is required")
        return value


class GoalsUpdate(BaseModel):
    goals: list[GoalUpdate]


def create_app(
    *,
    secret: str | None = None,
    dev_auth: bool | None = None,
    storage_factory: Callable[[str], Storage] | None = None,
    seed_config: str | Path | None = None,
    token_ttl_seconds: int | None = None,
) -> FastAPI:
    """Build the API.

    Arguments override environment variables. Tests pass a storage factory
    and leave ``seed_config`` unset so they never touch the real logs.
    """
    settings = _settings_from_env(
        secret=secret,
        dev_auth=dev_auth,
        storage_factory=storage_factory,
        seed_config=seed_config,
        token_ttl_seconds=token_ttl_seconds,
    )
    app = FastAPI(
        title="Foodtimizer API",
        version="0.1.0",
        description=(
            "Daily tracking API for the phone app. "
            "The Streamlit tracker does not use these routes."
        ),
    )

    def require_user(
        creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> _User:
        if creds is None or creds.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=401,
                detail="Missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            payload = read_token(settings.secret, creds.credentials)
        except InvalidToken as exc:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc
        sub = str(payload["sub"])
        email = payload.get("email")
        return _User(
            sub=sub,
            email=str(email) if email else None,
            namespace=namespace_for_subject(sub),
        )

    def library_for(user: _User) -> tuple[Storage, Problem]:
        storage = settings.storage_factory(user.namespace)
        raw = storage.load_config()
        if raw is None and settings.seed_config is not None and settings.seed_config.exists():
            raw = problem_to_dict(load_problem(settings.seed_config))
            storage.save_config(raw)
        if raw is None:
            raise HTTPException(
                status_code=409,
                detail="No food library for this account yet.",
            )
        try:
            problem = _problem_from_dict(raw)
        except Exception as exc:  # noqa: BLE001 - show the config error to the client
            raise HTTPException(
                status_code=500,
                detail=f"Could not load the food library: {exc}",
            ) from exc
        return storage, problem

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/v1/auth/dev-login")
    def dev_login(body: DevLogin) -> dict[str, str]:
        """Issue a token for a chosen account id. Off unless dev auth is enabled."""
        if not settings.dev_auth:
            raise HTTPException(status_code=404, detail="Not found")
        token = issue_token(
            settings.secret,
            body.sub,
            body.email,
            ttl_seconds=settings.token_ttl_seconds,
        )
        return {
            "access_token": token,
            "token_type": "bearer",
            "namespace": namespace_for_subject(body.sub),
        }

    @app.get("/v1/me")
    def me(user: _User = Depends(require_user)) -> dict[str, str | None]:
        return {"sub": user.sub, "email": user.email, "namespace": user.namespace}

    @app.get("/v1/ingredients")
    def ingredients(
        q: str = "",
        user: _User = Depends(require_user),
    ) -> dict[str, Any]:
        _storage, problem = library_for(user)
        needle = q.strip().lower()
        rows = []
        for ing in sorted(problem.ingredients, key=lambda item: item.name.lower()):
            if needle and needle not in ing.name.lower():
                continue
            rows.append({"name": ing.name, "macros": {k: float(v) for k, v in ing.macros.items()}})
        return {"ingredients": rows}

    @app.get("/v1/days/{on_date}")
    def get_day(on_date: date, user: _User = Depends(require_user)) -> dict[str, Any]:
        storage, problem = library_for(user)
        return _day_payload(problem, _load_log(storage, on_date))

    @app.post("/v1/days/{on_date}/entries")
    def add_entry(
        on_date: date,
        body: EntryCreate,
        user: _User = Depends(require_user),
    ) -> dict[str, Any]:
        if not math.isfinite(body.grams):
            raise HTTPException(status_code=400, detail="Grams must be greater than 0.")
        storage, problem = library_for(user)
        known = {ing.name for ing in problem.ingredients}
        if body.ingredient not in known:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown ingredient {body.ingredient!r}.",
            )
        log = _load_log(storage, on_date).with_added(
            make_entry(
                body.ingredient,
                body.grams,
                slot=body.slot,
                note=body.note,
            )
        )
        _save_log(storage, log)
        return _day_payload(problem, log)

    @app.delete("/v1/days/{on_date}/entries/{index}")
    def delete_entry(
        on_date: date,
        index: int,
        user: _User = Depends(require_user),
    ) -> dict[str, Any]:
        storage, problem = library_for(user)
        log = _load_log(storage, on_date)
        try:
            log = log.with_removed(index)
        except IndexError as exc:
            raise HTTPException(status_code=404, detail="No entry at that index.") from exc
        _save_log(storage, log)
        return _day_payload(problem, log)

    @app.put("/v1/days/{on_date}/weight")
    def set_weight(
        on_date: date,
        body: WeightUpdate,
        user: _User = Depends(require_user),
    ) -> dict[str, Any]:
        if body.kg is not None and not math.isfinite(body.kg):
            raise HTTPException(status_code=400, detail="Weight must be greater than 0 kg.")
        storage, problem = library_for(user)
        log = _load_log(storage, on_date)
        try:
            log = log.with_weight(body.kg)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _save_log(storage, log)
        return _day_payload(problem, log)

    @app.get("/v1/weight")
    def weight_history(user: _User = Depends(require_user)) -> dict[str, Any]:
        storage, _problem = library_for(user)
        points = []
        for on_date in sorted(storage.logged_dates()):
            log = _load_log(storage, on_date)
            if log.weight_kg is not None:
                points.append({"date": on_date.isoformat(), "kg": log.weight_kg})
        return {"points": points}

    @app.get("/v1/goals")
    def get_goals(user: _User = Depends(require_user)) -> dict[str, Any]:
        _storage, problem = library_for(user)
        return {"goals": [_goal_payload(t) for t in problem.targets]}

    @app.put("/v1/goals")
    def put_goals(
        body: GoalsUpdate,
        user: _User = Depends(require_user),
    ) -> dict[str, Any]:
        storage, problem = library_for(user)
        original = {t.name: t for t in problem.targets}
        updated = problem
        for goal in body.goals:
            if goal.value < 0 or not math.isfinite(goal.value):
                raise HTTPException(status_code=400, detail="Goal values cannot be negative.")
            if goal.value == 0:
                if any(t.name == goal.name for t in updated.targets):
                    try:
                        updated = delete_target(updated, goal.name)
                    except ValueError as exc:
                        raise HTTPException(status_code=400, detail=str(exc)) from exc
                continue
            existing = original.get(goal.name)
            try:
                updated = upsert_target(
                    updated,
                    goal.name,
                    value=float(goal.value),
                    weight=existing.weight if existing else None,
                    lower=existing.lower if existing else None,
                    upper=existing.upper if existing else None,
                    hard=existing.hard if existing else None,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            storage.save_config(problem_to_dict(updated))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"goals": [_goal_payload(t) for t in updated.targets]}

    return app


def _settings_from_env(
    *,
    secret: str | None,
    dev_auth: bool | None,
    storage_factory: Callable[[str], Storage] | None,
    seed_config: str | Path | None,
    token_ttl_seconds: int | None,
) -> _Settings:
    resolved_secret = secret if secret is not None else os.environ.get("FOODTIMIZER_API_SECRET", "")
    if len(resolved_secret) < _MIN_SECRET_LENGTH:
        raise RuntimeError(
            "Set FOODTIMIZER_API_SECRET to a private string of at least "
            f"{_MIN_SECRET_LENGTH} characters."
        )
    if dev_auth is None:
        dev_auth = os.environ.get("FOODTIMIZER_DEV_AUTH", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
    if token_ttl_seconds is None:
        days = os.environ.get("FOODTIMIZER_API_TOKEN_DAYS", str(_DEFAULT_TOKEN_DAYS))
        try:
            token_ttl_seconds = int(days) * 24 * 60 * 60
        except ValueError as exc:
            raise RuntimeError("FOODTIMIZER_API_TOKEN_DAYS must be a whole number.") from exc
    if token_ttl_seconds <= 0:
        raise RuntimeError("Token lifetime must be greater than 0.")

    seed_path: Path | None
    if storage_factory is None:
        storage_factory = _default_storage_factory()
        if seed_config is None:
            seed_config = os.environ.get("FOODTIMIZER_CONFIG", "examples/day.yaml")
        seed_path = Path(seed_config) if seed_config else None
    else:
        seed_path = Path(seed_config) if seed_config else None

    return _Settings(
        secret=resolved_secret,
        dev_auth=dev_auth,
        token_ttl_seconds=token_ttl_seconds,
        storage_factory=storage_factory,
        seed_config=seed_path,
    )


_engines: dict[str, Any] = {}


def _default_storage_factory() -> Callable[[str], Storage]:
    """One storage per account, from ``DATABASE_URL`` or local files."""
    url = os.environ.get("DATABASE_URL") or None
    config_path = os.environ.get("FOODTIMIZER_CONFIG", "examples/day.yaml")
    logs_dir = os.environ.get("FOODTIMIZER_LOGS_DIR", "logs")

    def factory(namespace: str) -> Storage:
        if not url:
            return FileStorage(config_path, logs_dir, namespace=namespace)
        engine = _engines.get(url)
        if engine is None:
            from sqlalchemy import create_engine

            engine = create_engine(url, pool_pre_ping=True)
            _engines[url] = engine
        return SqlStorage(engine, namespace=namespace)

    return factory


def _load_log(storage: Storage, on_date: date) -> DayLog:
    return daylog_from_data(on_date, storage.load_day(on_date))


def _save_log(storage: Storage, log: DayLog) -> None:
    storage.save_day(log.log_date, daylog_to_data(log))


def _day_payload(problem: Problem, log: DayLog) -> dict[str, Any]:
    ingredient_map = {ing.name: ing for ing in problem.ingredients}
    totals = compute_totals(log, ingredient_map)
    for target in problem.targets:
        totals.setdefault(target.name, 0.0)
    return {
        "date": log.log_date.isoformat(),
        "weight_kg": log.weight_kg,
        "entries": [
            {"index": index, **entry.to_dict()} for index, entry in enumerate(log.entries)
        ],
        "totals": totals,
        "targets": [_goal_payload(t) for t in problem.targets],
        "unknown_ingredients": unknown_ingredients(log, ingredient_map),
    }


def _goal_payload(target: MacroTarget) -> dict[str, Any]:
    return {
        "name": target.name,
        "value": target.value,
        "lower": target.lower,
        "upper": target.upper,
        "weight": target.weight,
        "hard": target.hard,
    }

