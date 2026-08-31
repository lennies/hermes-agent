"""Trusted-host-policy coverage for auxiliary model invocations."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent.auxiliary_client import (
    async_call_llm,
    bind_runtime_trusted_policy_from_prompt,
    call_llm,
    reset_runtime_trusted_policy_block,
    set_runtime_trusted_policy_block,
)
from agent.prompt_builder import GlobalInstructionsError, LoadedGlobalInstructions
from agent.prompt_builder import render_trusted_policy_prefix
from tools.approval import _smart_approve


def _loaded_policy() -> LoadedGlobalInstructions:
    return LoadedGlobalInstructions(
        content="Host policy body.",
        resolved_path=Path("/opt/elfbot/AGENTS.md"),
        sha256="a" * 64,
    )


def _response(content: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_sync_auxiliary_call_prepends_policy_without_mutating_input():
    messages = [{"role": "user", "content": "summarize this"}]

    with (
        patch(
            "agent.prompt_builder.load_global_instructions_file",
            return_value=_loaded_policy(),
        ),
        patch("agent.auxiliary_client._call_llm_impl", return_value=_response()) as impl,
    ):
        call_llm(task="compression", messages=messages)

    sent = impl.call_args.kwargs["messages"]
    assert messages == [{"role": "user", "content": "summarize this"}]
    assert sent[0]["role"] == "system"
    assert "# Trusted Host Policy" in sent[0]["content"]
    assert "Source: \"/opt/elfbot/AGENTS.md\"" in sent[0]["content"]
    assert "SHA-256: " + "a" * 64 in sent[0]["content"]
    assert "Host policy body." in sent[0]["content"]
    assert sent[1:] == messages


@pytest.mark.asyncio
async def test_async_auxiliary_call_prepends_policy():
    with (
        patch(
            "agent.prompt_builder.load_global_instructions_file",
            return_value=_loaded_policy(),
        ),
        patch(
            "agent.auxiliary_client._async_call_llm_impl",
            new=AsyncMock(return_value=_response()),
        ) as impl,
    ):
        await async_call_llm(messages=[{"role": "user", "content": "inspect"}])

    sent = impl.call_args.kwargs["messages"]
    assert sent[0]["role"] == "system"
    assert "Host policy body." in sent[0]["content"]


def test_disabled_policy_leaves_messages_byte_for_byte_unchanged():
    messages = [{"role": "system", "content": "task"}, {"role": "user", "content": "x"}]

    with (
        patch("agent.prompt_builder.load_global_instructions_file", return_value=None),
        patch("agent.auxiliary_client._call_llm_impl", return_value=_response()) as impl,
    ):
        call_llm(messages=messages)

    assert impl.call_args.kwargs["messages"] is messages


def test_invalid_policy_fails_before_provider_resolution():
    with (
        patch(
            "agent.prompt_builder.load_global_instructions_file",
            side_effect=GlobalInstructionsError("invalid host policy"),
        ),
        patch("agent.auxiliary_client._call_llm_impl") as impl,
    ):
        with pytest.raises(GlobalInstructionsError, match="invalid host policy"):
            call_llm(messages=[{"role": "user", "content": "x"}])

    impl.assert_not_called()


def test_session_bound_auxiliary_call_uses_frozen_block_without_disk_reload():
    token = set_runtime_trusted_policy_block(
        "# Trusted Host Policy\n\nSource: \"/frozen/AGENTS.md\"\n"
        f"SHA-256: {'b' * 64}\n\nFrozen policy."
    )
    try:
        with (
            patch("agent.prompt_builder.load_global_instructions_file") as loader,
            patch("agent.auxiliary_client._call_llm_impl", return_value=_response()) as impl,
        ):
            call_llm(messages=[{"role": "user", "content": "x"}])
    finally:
        reset_runtime_trusted_policy_block(token)

    loader.assert_not_called()
    sent = impl.call_args.kwargs["messages"]
    assert "Frozen policy." in sent[0]["content"]


def test_session_bound_explicit_absence_does_not_hot_load_policy():
    token = set_runtime_trusted_policy_block(None)
    messages = [{"role": "user", "content": "x"}]
    try:
        with (
            patch("agent.prompt_builder.load_global_instructions_file") as loader,
            patch("agent.auxiliary_client._call_llm_impl", return_value=_response()) as impl,
        ):
            call_llm(messages=messages)
    finally:
        reset_runtime_trusted_policy_block(token)

    loader.assert_not_called()
    assert impl.call_args.kwargs["messages"] is messages


def test_prompt_binding_uses_integrity_verified_configured_snapshot():
    prompt = render_trusted_policy_prefix("Identity", _loaded_policy())
    with patch("agent.prompt_builder.load_global_instructions_file") as loader:
        token = bind_runtime_trusted_policy_from_prompt(prompt)
    try:
        with patch("agent.auxiliary_client._call_llm_impl", return_value=_response()) as impl:
            call_llm(messages=[{"role": "user", "content": "x"}])
    finally:
        reset_runtime_trusted_policy_block(token)

    loader.assert_not_called()
    assert "Host policy body." in impl.call_args.kwargs["messages"][0]["content"]


def test_prompt_binding_preserves_integrity_verified_absence():
    prompt = render_trusted_policy_prefix("Identity", None)
    with patch("agent.prompt_builder.load_global_instructions_file") as loader:
        token = bind_runtime_trusted_policy_from_prompt(prompt)
    try:
        with patch("agent.auxiliary_client._call_llm_impl", return_value=_response()) as impl:
            messages = [{"role": "user", "content": "x"}]
            call_llm(messages=messages)
    finally:
        reset_runtime_trusted_policy_block(token)

    loader.assert_not_called()
    assert impl.call_args.kwargs["messages"] is messages


def test_prompt_binding_rejects_corrupt_claimed_snapshot():
    prompt = render_trusted_policy_prefix("Identity", _loaded_policy()).replace(
        "Host policy body.", "Tampered policy."
    )

    with (
        patch("agent.prompt_builder.load_global_instructions_file") as loader,
        pytest.raises(GlobalInstructionsError, match="missing or corrupt"),
    ):
        bind_runtime_trusted_policy_from_prompt(prompt)

    loader.assert_not_called()


def test_prompt_binding_legacy_unframed_prompt_freezes_current_policy():
    with patch(
        "agent.prompt_builder.load_global_instructions_file",
        return_value=_loaded_policy(),
    ) as loader:
        token = bind_runtime_trusted_policy_from_prompt("Legacy prompt")
    try:
        with (
            patch("agent.prompt_builder.load_global_instructions_file") as second_load,
            patch("agent.auxiliary_client._call_llm_impl", return_value=_response()) as impl,
        ):
            call_llm(messages=[{"role": "user", "content": "x"}])
    finally:
        reset_runtime_trusted_policy_block(token)

    loader.assert_called_once()
    second_load.assert_not_called()
    assert "Host policy body." in impl.call_args.kwargs["messages"][0]["content"]


def test_smart_approval_guard_receives_host_policy_as_highest_system_message():
    captured = {}

    def fake_impl(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _response("ESCALATE")

    with (
        patch(
            "agent.prompt_builder.load_global_instructions_file",
            return_value=_loaded_policy(),
        ),
        patch("agent.auxiliary_client._call_llm_impl", side_effect=fake_impl),
    ):
        assert _smart_approve("python -c 'print(1)'", "script") == "escalate"

    assert captured["messages"][0]["role"] == "system"
    assert "Host policy body." in captured["messages"][0]["content"]
    assert "security reviewer" in captured["messages"][1]["content"]
