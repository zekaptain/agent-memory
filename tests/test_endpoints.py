"""Model egress is checked at the HTTP boundary, not inferred from a backend name."""

import io
import json
import os
import subprocess
import sys
import urllib.error
from email.message import Message
from unittest.mock import MagicMock

import pytest

from agent_memory import endpoints


@pytest.mark.parametrize(
    "url",
    ["http://localhost:1", "http://127.0.0.1", "http://127.20.30.40", "http://[::1]:2"],
)
def test_loopback_needs_no_remote_permission(monkeypatch, url):
    monkeypatch.delenv(endpoints.REMOTE_ENV, raising=False)
    assert endpoints.is_loopback(url)
    endpoints.require_egress(url, "test")


@pytest.mark.parametrize(
    "url",
    ["http://gpu-box:11434", "http://localhost.example", "http://128.0.0.1", "http://[::2]"],
)
def test_non_loopback_requires_explicit_permission(monkeypatch, url):
    monkeypatch.delenv(endpoints.REMOTE_ENV, raising=False)
    assert not endpoints.is_loopback(url)
    with pytest.raises(endpoints.EgressError, match=endpoints.REMOTE_ENV):
        endpoints.require_egress(url, "model request")
    monkeypatch.setenv(endpoints.REMOTE_ENV, "true")
    assert not endpoints.remote_allowed()
    monkeypatch.setenv(endpoints.REMOTE_ENV, "1")
    endpoints.require_egress(url, "model request")


def test_post_checks_remote_ollama_before_network(monkeypatch):
    monkeypatch.delenv(endpoints.REMOTE_ENV, raising=False)
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b'{"ok":true}'
    send = MagicMock(return_value=response)
    monkeypatch.setattr(endpoints.urllib.request, "urlopen", send)
    url = "http://gpu-box:11434/api/chat"
    with pytest.raises(endpoints.EgressError, match=endpoints.REMOTE_ENV):
        endpoints.post_json(url, {"private": "synthetic"})
    send.assert_not_called()
    monkeypatch.setenv(endpoints.REMOTE_ENV, "1")
    assert endpoints.post_json(url, {})[0] == {"ok": True}
    send.assert_called_once()


def test_a_server_that_was_not_asked_is_unreachable(monkeypatch):
    refused = urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
    send = MagicMock(side_effect=refused)
    monkeypatch.setattr(endpoints.urllib.request, "urlopen", send)
    with pytest.raises(endpoints.Unreachable, match="Connection refused"):
        endpoints.post_json("http://localhost:1/api/chat", {})
    send.assert_called_once()

    monkeypatch.delenv(endpoints.REMOTE_ENV, raising=False)
    with pytest.raises(endpoints.Unreachable, match=endpoints.REMOTE_ENV):
        endpoints.get_json("http://gpu-box:11434/api/tags")
    send.assert_called_once()


@pytest.mark.parametrize(
    "fault",
    [
        TimeoutError("timed out"),
        urllib.error.URLError(TimeoutError("timed out")),
        ConnectionResetError(54, "Connection reset by peer"),
        urllib.error.HTTPError("u", 400, "Bad Request", Message(), io.BytesIO(b"too long")),
    ],
    ids=["read timeout", "connect timeout", "dropped while waiting", "rejected"],
)
def test_a_server_that_was_too_slow_or_said_no_was_reached(monkeypatch, fault):
    """Either may be the doing of what was sent, so neither says the server is gone."""
    monkeypatch.setattr(endpoints.urllib.request, "urlopen", MagicMock(side_effect=fault))
    with pytest.raises(endpoints.EgressError) as raised:
        endpoints.post_json("http://localhost:1/api/chat", {})
    assert not isinstance(raised.value, endpoints.Unreachable)


def test_hosts_are_resolved_from_environment_once():
    env = {**os.environ, "OLLAMA_HOST": "http://gpu-box:11434/"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; from agent_memory import endpoints; "
            "os.environ['OLLAMA_HOST'] = 'http://changed'; "
            "print(endpoints.OLLAMA_HOST)",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "http://gpu-box:11434"


def test_graph_import_does_not_load_judge():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys; import agent_memory.graph; "
            "import agent_memory.graph.resolve; import agent_memory.graph.cli; "
            "print(json.dumps([m for m in sys.modules if m.startswith('agent_memory.judge')]))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == []
