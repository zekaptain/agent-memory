"""Model egress is checked at the HTTP boundary, not inferred from a backend name."""

import json
import os
import subprocess
import sys
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
