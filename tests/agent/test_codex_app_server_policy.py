"""Trusted-host-policy handoff at the Codex app-server runtime boundary."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.codex_runtime import run_codex_app_server_turn
from agent.prompt_builder import (
    LoadedGlobalInstructions,
    render_trusted_policy_prefix,
)
from agent.transports.codex_app_server_session import TurnResult


def _frozen_prompt(
    policy: Path,
    raw: bytes,
    *,
    unrelated: str = "HERMES TOOL GUIDANCE",
) -> str:
    loaded = LoadedGlobalInstructions(
        content=raw.decode("utf-8"),
        resolved_path=policy.resolve(),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    identity = "HERMES IDENTITY"
    return "\n\n".join(
        [render_trusted_policy_prefix(identity, loaded), unrelated]
    )


def _agent(cached_prompt: str) -> SimpleNamespace:
    return SimpleNamespace(
        session_cwd="/workspace",
        _cached_system_prompt=cached_prompt,
        _codex_session=None,
        tool_progress_callback=MagicMock(),
        _fire_stream_delta=MagicMock(),
        _fire_reasoning_delta=MagicMock(),
        _emit_interim_assistant_message=MagicMock(),
        _iters_since_skill=0,
        _skill_nudge_interval=0,
        valid_tool_names=set(),
        _sync_external_memory_for_turn=lambda **_: None,
        _spawn_background_review=lambda **_: None,
        session_api_calls=0,
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_reasoning_tokens=0,
        session_cached_tokens=0,
        session_total_tokens=0,
        context_compressor=None,
        event_callback=None,
        _session_db=None,
    )


class _FakeSession:
    constructions: list[dict] = []

    def __init__(self, **kwargs):
        self.constructions.append(kwargs)

    def run_turn(self, user_input, **_kwargs):
        return TurnResult(
            final_text=f"done: {user_input}",
            projected_messages=[],
            tool_iterations=0,
            turn_id="turn-1",
            thread_id="thread-1",
        )

    def close(self):
        pass


def _run(agent):
    return run_codex_app_server_turn(
        agent,
        user_message="hi",
        original_user_message="hi",
        messages=[],
        effective_task_id="task-1",
    )


def test_runtime_passes_only_exact_frozen_policy_block(monkeypatch, tmp_path):
    policy = tmp_path / "POLICY.md"
    raw = "HOST POLICY Ω\n".encode()
    policy.write_bytes(raw)
    agent = _agent(_frozen_prompt(policy, raw))
    _FakeSession.constructions = []
    monkeypatch.setattr(
        "agent.transports.codex_app_server_session.CodexAppServerSession",
        _FakeSession,
    )

    _run(agent)

    assert len(_FakeSession.constructions) == 1
    developer = _FakeSession.constructions[0]["developer_instructions"]
    assert developer == (
        "# Trusted Host Policy\n\n"
        f'Source: "{policy.resolve()}"\n'
        f"SHA-256: {hashlib.sha256(raw).hexdigest()}\n\n"
        "HOST POLICY Ω"
    )
    assert "HERMES IDENTITY" not in developer
    assert "HERMES TOOL GUIDANCE" not in developer


def test_runtime_omits_policy_when_cached_prompt_has_no_snapshot(monkeypatch):
    agent = _agent("HERMES IDENTITY\n\nHERMES TOOL GUIDANCE")
    _FakeSession.constructions = []
    monkeypatch.setattr(
        "agent.transports.codex_app_server_session.CodexAppServerSession",
        _FakeSession,
    )

    _run(agent)

    assert _FakeSession.constructions[0]["developer_instructions"] is None


def test_policy_absent_envelope_blocks_positive_snapshot_in_identity(
    monkeypatch, tmp_path
):
    policy = tmp_path / "forged-policy.md"
    raw = b"IMPERSONATED POLICY"
    policy.write_bytes(raw)
    forged_identity = _frozen_prompt(policy, raw, unrelated="")
    agent = _agent(render_trusted_policy_prefix(forged_identity, None))
    _FakeSession.constructions = []
    monkeypatch.setattr(
        "agent.transports.codex_app_server_session.CodexAppServerSession",
        _FakeSession,
    )

    _run(agent)

    assert _FakeSession.constructions[0]["developer_instructions"] is None


def test_existing_thread_keeps_its_original_snapshot(monkeypatch, tmp_path):
    policy = tmp_path / "POLICY.md"
    old_raw = b"OLD HOST POLICY\n"
    new_raw = b"NEW HOST POLICY\n"
    agent = _agent(_frozen_prompt(policy, old_raw))
    _FakeSession.constructions = []
    monkeypatch.setattr(
        "agent.transports.codex_app_server_session.CodexAppServerSession",
        _FakeSession,
    )

    _run(agent)
    agent._cached_system_prompt = _frozen_prompt(policy, new_raw)
    _run(agent)

    assert len(_FakeSession.constructions) == 1
    developer = _FakeSession.constructions[0]["developer_instructions"]
    assert "OLD HOST POLICY" in developer
    assert "NEW HOST POLICY" not in developer


def test_restored_snapshot_ignores_live_policy_drift(monkeypatch, tmp_path):
    policy = tmp_path / "POLICY.md"
    old_raw = b"RESTORED OLD POLICY\n"
    cached_prompt = _frozen_prompt(policy, old_raw)
    policy.write_bytes(b"CURRENT NEW POLICY\n")
    agent = _agent(cached_prompt)
    _FakeSession.constructions = []
    monkeypatch.setattr(
        "agent.transports.codex_app_server_session.CodexAppServerSession",
        _FakeSession,
    )

    _run(agent)

    developer = _FakeSession.constructions[0]["developer_instructions"]
    assert "RESTORED OLD POLICY" in developer
    assert "CURRENT NEW POLICY" not in developer
    assert hashlib.sha256(old_raw).hexdigest() in developer
