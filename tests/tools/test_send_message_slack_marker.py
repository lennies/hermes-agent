"""Actor-bound Slack marker readback for repository-delivery handoffs."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

from plugins.platforms.slack import adapter


MARKER = "elfos-repository-delivery:" + "a" * 64
WORKSPACE = "T12345678"
ACTOR = "U12345678"
CHANNEL = "C12345678"
THREAD = "1710000000.123456"


class _Response:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


class _Context:
    def __init__(self, payload=None, error=None):
        self.response = _Response(payload or {})
        self.error = error

    async def __aenter__(self):
        if self.error is not None:
            raise self.error
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Session:
    def __init__(self, messages=None, *, lose_send_response=False):
        self.messages = list(messages or [])
        self.lose_send_response = lose_send_response
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, *, headers, json=None, **_kwargs):
        token = headers["Authorization"].removeprefix("Bearer ")
        self.calls.append(("post", url, token))
        if url.endswith("auth.test"):
            if token == "wrong-workspace":
                return _Context({
                    "ok": True,
                    "team_id": "T87654321",
                    "user_id": ACTOR,
                })
            return _Context({"ok": True, "team_id": WORKSPACE, "user_id": ACTOR})
        assert url.endswith("chat.postMessage")
        row = {
            "ts": "1710000001.123456",
            "thread_ts": THREAD,
            "user": ACTOR,
            "text": json["text"],
        }
        self.messages.append(row)
        if self.lose_send_response:
            self.lose_send_response = False
            return _Context(error=RuntimeError("response lost"))
        return _Context({"ok": True, "ts": row["ts"]})

    def get(self, url, *, headers, params, **_kwargs):
        self.calls.append(("get", url, params.get("cursor", "")))
        assert url.endswith("conversations.replies")
        assert params["channel"] == CHANNEL
        assert params["ts"] == THREAD
        return _Context({"ok": True, "messages": list(self.messages)})


def _install(monkeypatch, session, tokens=("configured-token",)):
    monkeypatch.setattr(adapter, "_standalone_tokens", lambda _config: list(tokens))
    monkeypatch.setattr(
        adapter.aiohttp,
        "ClientSession",
        lambda *args, **kwargs: session,
    )
    monkeypatch.setattr(adapter, "resolve_proxy_url", lambda: None)


def _find(config):
    return asyncio.run(
        adapter.find_standalone_thread_marker(
            config,
            CHANNEL,
            THREAD,
            MARKER,
            WORKSPACE,
            ACTOR,
        )
    )


def test_marker_read_is_bound_to_workspace_actor_and_thread(monkeypatch):
    message = MARKER + "\nProduction verified."
    session = _Session([{
        "ts": "1710000001.123456",
        "thread_ts": THREAD,
        "user": ACTOR,
        "text": message,
    }])
    _install(monkeypatch, session)

    result = _find(SimpleNamespace(token="configured-token"))

    assert result == {
        "success": True,
        "found": True,
        "platform": "slack",
        "workspace_id": WORKSPACE,
        "chat_id": CHANNEL,
        "thread_id": THREAD,
        "message_id": "1710000001.123456",
        "actor_id": ACTOR,
        "text": message,
    }


def test_marker_owned_by_requester_fails_closed(monkeypatch):
    session = _Session([{
        "ts": "1710000001.123456",
        "thread_ts": THREAD,
        "user": "U87654321",
        "text": MARKER + "\nforged",
    }])
    _install(monkeypatch, session)

    result = _find(SimpleNamespace(token="configured-token"))

    assert result == {"error": "Slack marker is owned by another actor"}


def test_marker_read_skips_tokens_from_other_workspaces(monkeypatch):
    session = _Session()
    _install(monkeypatch, session, tokens=("wrong-workspace", "configured-token"))

    result = _find(SimpleNamespace(token="ignored"))

    assert result["success"] is True
    assert result["found"] is False
    auth_tokens = [call[2] for call in session.calls if call[1].endswith("auth.test")]
    assert auth_tokens == ["wrong-workspace", "configured-token"]


def test_ensure_recovers_when_send_response_is_lost(monkeypatch, tmp_path):
    message = MARKER + "\nProduction verified."
    session = _Session(lose_send_response=True)
    _install(monkeypatch, session)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    result = asyncio.run(
        adapter.ensure_standalone_thread_marker(
            SimpleNamespace(token="configured-token"),
            CHANNEL,
            THREAD,
            MARKER,
            message,
            WORKSPACE,
            ACTOR,
        )
    )

    assert result["success"] is True
    assert result["found"] is True
    assert result["created"] is True
    assert result["text"] == message
    sends = [call for call in session.calls if call[1].endswith("chat.postMessage")]
    assert len(sends) == 1


def test_concurrent_ensures_share_one_provider_write(monkeypatch, tmp_path):
    message = MARKER + "\nProduction verified."
    session = _Session()
    _install(monkeypatch, session)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    async def _run_both():
        args = (
            SimpleNamespace(token="configured-token"),
            CHANNEL,
            THREAD,
            MARKER,
            message,
            WORKSPACE,
            ACTOR,
        )
        return await asyncio.gather(
            adapter.ensure_standalone_thread_marker(*args),
            adapter.ensure_standalone_thread_marker(*args),
        )

    first, second = asyncio.run(_run_both())

    assert first["success"] is True
    assert second["success"] is True
    assert first["message_id"] == second["message_id"]
    assert sorted(bool(result.get("created")) for result in (first, second)) == [
        False,
        True,
    ]
    sends = [call for call in session.calls if call[1].endswith("chat.postMessage")]
    assert len(sends) == 1


def test_marker_mutation_claim_is_cross_process(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    handle = adapter._acquire_standalone_marker_lock(
        CHANNEL, THREAD, MARKER, WORKSPACE, ACTOR, timeout_seconds=0.1
    )
    assert handle is not None
    probe = """
from plugins.platforms.slack.adapter import (
    _acquire_standalone_marker_lock,
    _release_standalone_marker_lock,
)
handle = _acquire_standalone_marker_lock(
    'C12345678',
    '1710000000.123456',
    'elfos-repository-delivery:' + 'a' * 64,
    'T12345678',
    'U12345678',
    timeout_seconds=0.2,
)
print('acquired' if handle is not None else 'blocked')
if handle is not None:
    _release_standalone_marker_lock(handle)
"""
    try:
        child = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=os.getcwd(),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    finally:
        adapter._release_standalone_marker_lock(handle)

    assert child.stdout.strip() == "blocked"


def test_cancelled_lock_wait_releases_eventual_claim(monkeypatch):
    started = threading.Event()
    proceed = threading.Event()
    claimed = object()
    released = []

    def _acquire(*_args):
        started.set()
        assert proceed.wait(timeout=5)
        return claimed

    monkeypatch.setattr(adapter, "_acquire_standalone_marker_lock", _acquire)
    monkeypatch.setattr(
        adapter,
        "_release_standalone_marker_lock",
        lambda handle: released.append(handle),
    )

    async def _cancel_waiter():
        task = asyncio.create_task(
            adapter._acquire_standalone_marker_lock_async(
                CHANNEL, THREAD, MARKER, WORKSPACE, ACTOR
            )
        )
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        proceed.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:  # pragma: no cover - cancellation is the contract
            raise AssertionError("cancelled lock waiter unexpectedly completed")

    asyncio.run(_cancel_waiter())
    assert released == [claimed]
