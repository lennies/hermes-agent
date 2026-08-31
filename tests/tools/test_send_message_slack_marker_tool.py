"""Tool routing for actor-bound repository-delivery Slack markers."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from gateway.config import Platform
from plugins.platforms.slack import adapter
from tools import send_message_tool as tool


MARKER = "elfos-repository-delivery:" + "c" * 64
TARGET = "slack:C12345678:1710000000.123456"


def _setup(monkeypatch):
    config = SimpleNamespace(
        platforms={Platform.SLACK: SimpleNamespace(enabled=True, token="token")}
    )
    monkeypatch.setattr(tool, "prepare_send_message_platforms", lambda: None)
    monkeypatch.setattr(
        tool,
        "resolve_send_target",
        lambda platform, target: (
            "C12345678",
            "1710000000.123456",
            None,
        ),
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr("model_tools._run_async", asyncio.run)


def test_find_marker_returns_exact_actor_bound_readback(monkeypatch):
    _setup(monkeypatch)

    async def find(*_args):
        return {
            "success": True,
            "found": True,
            "workspace_id": "T12345678",
            "chat_id": "C12345678",
            "thread_id": "1710000000.123456",
            "actor_id": "U12345678",
            "message_id": "1710000001.123456",
            "text": MARKER + "\nProduction verified.",
        }

    monkeypatch.setattr(adapter, "find_standalone_thread_marker", find)

    result = json.loads(tool.send_message_tool({
        "action": "find",
        "target": TARGET,
        "marker": MARKER,
        "workspace_id": "T12345678",
        "actor_id": "U12345678",
    }))

    assert result["action"] == "found"
    assert result["message_id"] == "1710000001.123456"


def test_ensure_marker_uses_idempotent_adapter_operation(monkeypatch):
    _setup(monkeypatch)
    message = MARKER + "\nProduction verified."

    async def find(*_args):
        return {"success": True, "found": False}

    async def ensure(*_args):
        return {
            "success": True,
            "found": True,
            "created": True,
            "workspace_id": "T12345678",
            "chat_id": "C12345678",
            "thread_id": "1710000000.123456",
            "actor_id": "U12345678",
            "message_id": "1710000001.123456",
            "text": message,
        }

    monkeypatch.setattr(adapter, "find_standalone_thread_marker", find)
    monkeypatch.setattr(adapter, "ensure_standalone_thread_marker", ensure)

    result = json.loads(tool.send_message_tool({
        "action": "ensure",
        "target": TARGET,
        "message": message,
        "marker": MARKER,
        "workspace_id": "T12345678",
        "actor_id": "U12345678",
    }))

    assert result["action"] == "created"
    assert result["text"] == message
