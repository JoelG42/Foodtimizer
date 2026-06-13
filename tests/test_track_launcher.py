"""Tests for `foodtimizer-track` argument routing.

The launcher splits the user's argv into Streamlit config flags (which
must appear before `--` on the inner CLI) and app flags (which go
after). Getting this wrong silently breaks `--server.address 0.0.0.0`
and the like, which is what made phones unable to connect.
"""

from __future__ import annotations

from foodtimizer.track_launcher import _split_args


def test_app_only_args_all_go_to_app():
    streamlit_args, app_args = _split_args(
        ["--config", "examples/day.yaml", "--logs-dir", "logs"]
    )
    assert streamlit_args == []
    assert app_args == ["--config", "examples/day.yaml", "--logs-dir", "logs"]


def test_lan_shortcut_expands_to_server_address():
    """The convenience shortcut is the whole reason users get to skip
    Streamlit's flag name."""
    streamlit_args, app_args = _split_args(["--lan"])
    assert streamlit_args == ["--server.address", "0.0.0.0"]
    assert app_args == []


def test_split_form_server_flag_routes_to_streamlit():
    """`--server.address 0.0.0.0` (two tokens) is the form most users
    type; the launcher must group them together and route both."""
    streamlit_args, app_args = _split_args(["--server.address", "0.0.0.0"])
    assert streamlit_args == ["--server.address", "0.0.0.0"]
    assert app_args == []


def test_joined_form_server_flag_routes_to_streamlit():
    """`--server.port=8888` (one token) is the form Streamlit's own docs
    use; both forms must work."""
    streamlit_args, app_args = _split_args(["--server.port=8888"])
    assert streamlit_args == ["--server.port=8888"]
    assert app_args == []


def test_mixed_streamlit_and_app_args_split_correctly():
    """The realistic combo: bind to LAN, custom port, custom config."""
    streamlit_args, app_args = _split_args(
        [
            "--lan",
            "--server.port",
            "9000",
            "--config",
            "examples/day.yaml",
            "--logs-dir",
            "logs",
        ]
    )
    assert streamlit_args == [
        "--server.address",
        "0.0.0.0",
        "--server.port",
        "9000",
    ]
    assert app_args == ["--config", "examples/day.yaml", "--logs-dir", "logs"]


def test_streamlit_flag_without_value_passes_through_alone():
    """`--server.headless` is a boolean toggle with no value; the
    launcher must not greedily consume the next arg."""
    streamlit_args, app_args = _split_args(
        ["--server.headless", "--config", "x.yaml"]
    )
    assert streamlit_args == ["--server.headless"]
    assert app_args == ["--config", "x.yaml"]


def test_browser_and_theme_prefixes_routed_to_streamlit():
    """All Streamlit config namespaces must be recognised, not just
    `--server.*`."""
    streamlit_args, app_args = _split_args(
        ["--browser.gatherUsageStats=false", "--theme.base=dark"]
    )
    assert streamlit_args == [
        "--browser.gatherUsageStats=false",
        "--theme.base=dark",
    ]
    assert app_args == []
