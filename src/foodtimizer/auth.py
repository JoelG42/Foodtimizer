"""Authentication + per-user identity for the Streamlit app.

Uses Streamlit's built-in OpenID Connect auth (``st.login`` / ``st.user``),
which works on Streamlit Community Cloud and any OIDC provider (Google,
Auth0, Microsoft, ...). Configure it via an ``[auth]`` section in
``secrets.toml`` (see ``.streamlit/secrets.toml.example``).

Auth is *opt-in*: if no ``[auth]`` section is configured, the app runs in
single-user mode (great for local development and the existing tests). When
it is configured, every visitor must sign in, and each account gets its own
isolated data via :func:`current_namespace`.
"""

from __future__ import annotations

import re

import streamlit as st


def auth_configured() -> bool:
    """True if an ``[auth]`` section exists in secrets (multi-user mode)."""
    try:
        return "auth" in st.secrets
    except Exception:  # noqa: BLE001 - no secrets file locally is fine
        return False


def is_logged_in() -> bool:
    try:
        return bool(st.user.is_logged_in)
    except Exception:  # noqa: BLE001 - st.user unavailable when auth is off
        return False


def current_email() -> str | None:
    try:
        return st.user.email  # type: ignore[no-any-return]
    except Exception:  # noqa: BLE001
        return None


def current_name() -> str | None:
    for attr in ("name", "given_name", "preferred_username"):
        try:
            val = getattr(st.user, attr, None) or st.user.get(attr)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            val = None
        if val:
            return str(val)
    return current_email()


def current_namespace() -> str | None:
    """A stable, filesystem/DB-safe id for the signed-in user.

    Prefers the OIDC ``sub`` claim (immutable) and falls back to email.
    Returns ``None`` when auth is off / nobody is signed in.
    """
    raw: str | None = None
    try:
        raw = st.user.get("sub")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        raw = None
    if not raw:
        raw = current_email()
    if not raw:
        return None
    return "u_" + re.sub(r"[^A-Za-z0-9]", "_", str(raw))[:80]


def require_login() -> None:
    """Gate the app behind sign-in when auth is configured.

    No-op in single-user mode. Otherwise, unauthenticated visitors get a
    minimal sign-in screen and the run is stopped.
    """
    if not auth_configured() or is_logged_in():
        return

    st.title("🍽️ Foodtimizer")
    st.markdown(
        "Plan and track your nutrition — the optimizer builds your day to hit "
        "your macro targets. Please sign in to continue."
    )
    if st.button("Sign in with Google", type="primary"):
        st.login()
    st.stop()


def render_account_controls() -> None:
    """Show the signed-in user + a logout button in the sidebar."""
    if not auth_configured() or not is_logged_in():
        return
    who = current_email() or current_name() or "your account"
    st.sidebar.caption(f"👤 Signed in as {who}")
    if st.sidebar.button("Sign out"):
        st.logout()
