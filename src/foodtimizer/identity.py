"""Account id shared by the Streamlit app and the phone API.

The namespace is the prefix on every stored config and day log. Both clients
must derive it the same way, or a person who signs in on the phone will not
see the food they logged in Streamlit.
"""

from __future__ import annotations

import re


def namespace_for_subject(raw: str) -> str:
    """Filesystem- and database-safe id for an OIDC ``sub`` (or email).

    Non-alphanumerics become underscores, and the sanitized subject is capped
    at 80 characters. The ``u_`` prefix is not part of that cap.
    """
    return "u_" + re.sub(r"[^A-Za-z0-9]", "_", str(raw))[:80]
