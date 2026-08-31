"""Global-policy and execution-boundary coverage for standalone Mini-SWE."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agent.prompt_builder import GlobalInstructionsError
from mini_swe_runner import MiniSWERunner


@pytest.fixture()
def hermes_root(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


def _configure(root: Path, policy: Path) -> None:
    (root / "config.yaml").write_text(
        yaml.safe_dump({"global_instructions_file": str(policy)}),
        encoding="utf-8",
    )


def _response(content="done", tool_calls=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=[] if tool_calls is None else tool_calls,
                )
            )
        ]
    )


def _runner(client, *, env_type="local", max_iterations=2):
    with patch("openai.OpenAI", return_value=client):
        runner = MiniSWERunner(
            model="test-model",
            base_url="https://example.invalid/v1",
            api_key="test-key",
            env_type=env_type,
            max_iterations=max_iterations,
        )
    return runner


def _system_messages(client) -> list[str]:
    return [
        call.kwargs["messages"][0]["content"]
        for call in client.chat.completions.create.call_args_list
    ]


@pytest.mark.parametrize("env_type", ["local", "docker", "modal"])
def test_configured_policy_is_included_for_every_environment(
    hermes_root, tmp_path, env_type
):
    policy = tmp_path / "HOST-POLICY.md"
    raw = "HOST POLICY Ω\n".encode()
    policy.write_bytes(raw)
    _configure(hermes_root, policy)
    client = MagicMock()
    client.chat.completions.create.return_value = _response()
    runner = _runner(client, env_type=env_type)
    runner._create_env = MagicMock()
    runner._cleanup_env = MagicMock()

    runner.run_task("ordinary task")

    system_prompt = _system_messages(client)[0]
    assert "# Trusted Host Policy" in system_prompt
    assert f'Source: "{policy.resolve()}"' in system_prompt
    assert f"SHA-256: {hashlib.sha256(raw).hexdigest()}" in system_prompt
    assert "HOST POLICY Ω" in system_prompt
    assert client.chat.completions.create.call_args.kwargs["messages"][1] == {
        "role": "user",
        "content": "ordinary task",
    }


@pytest.mark.parametrize("env_type", ["docker", "modal"])
def test_hosted_environment_without_setting_omits_host_policy(
    hermes_root, env_type
):
    client = MagicMock()
    client.chat.completions.create.return_value = _response()
    runner = _runner(client, env_type=env_type)
    runner._create_env = MagicMock()
    runner._cleanup_env = MagicMock()

    runner.run_task("ordinary task")

    system_prompt = _system_messages(client)[0]
    assert "# Trusted Host Policy" not in system_prompt
    assert str(hermes_root) not in system_prompt


def test_named_profile_reads_policy_only_from_default_root(
    hermes_root, tmp_path, monkeypatch
):
    policy = tmp_path / "HOST-POLICY.md"
    policy.write_text("DEFAULT ROOT POLICY", encoding="utf-8")
    _configure(hermes_root, policy)
    profile = hermes_root / "profiles" / "coder"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("model: test\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(profile))
    client = MagicMock()
    client.chat.completions.create.return_value = _response()
    runner = _runner(client)
    runner._create_env = MagicMock()
    runner._cleanup_env = MagicMock()

    runner.run_task("ordinary task")

    assert "DEFAULT ROOT POLICY" in _system_messages(client)[0]


@pytest.mark.parametrize("failure", ["missing", "directory", "invalid_config"])
@pytest.mark.parametrize("env_type", ["local", "docker", "modal"])
def test_policy_load_failure_precedes_environment_and_api_side_effects(
    hermes_root, tmp_path, failure, env_type
):
    policy = tmp_path / "HOST-POLICY.md"
    if failure == "directory":
        policy.mkdir()
        _configure(hermes_root, policy)
    elif failure == "invalid_config":
        (hermes_root / "config.yaml").write_text("- not-a-mapping\n", encoding="utf-8")
    else:
        _configure(hermes_root, policy)
    client = MagicMock()
    runner = _runner(client, env_type=env_type)
    runner._create_env = MagicMock()

    with pytest.raises(GlobalInstructionsError):
        runner.run_task("must not start")

    runner._create_env.assert_not_called()
    client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("env_type", ["local", "docker", "modal"])
def test_environment_constructor_receives_no_policy_data(
    hermes_root, tmp_path, env_type
):
    policy = tmp_path / "HOST-POLICY.md"
    policy_text = "POLICY MUST STAY OUT OF ENVIRONMENT"
    policy.write_text(policy_text, encoding="utf-8")
    _configure(hermes_root, policy)
    client = MagicMock()
    client.chat.completions.create.return_value = _response()
    runner = _runner(client, env_type=env_type)
    environment = MagicMock()
    with patch("mini_swe_runner.create_environment", return_value=environment) as create:
        runner.run_task("ordinary task")

    serialized_args = json.dumps(
        {"args": create.call_args.args, "kwargs": create.call_args.kwargs},
        default=str,
    )
    assert policy_text not in serialized_args
    assert str(policy.resolve()) not in serialized_args
    assert "global_instructions" not in serialized_args
    if env_type == "docker":
        assert create.call_args.kwargs["include_host_context"] is False
        assert create.call_args.kwargs["persist_across_processes"] is False
        assert create.call_args.kwargs["persistent_filesystem"] is False
    elif env_type == "modal":
        assert create.call_args.kwargs["include_host_context"] is False
        assert create.call_args.kwargs["persistent_filesystem"] is False
    else:
        assert "include_host_context" not in create.call_args.kwargs


def test_task_loop_reuses_identical_frozen_system_prompt(hermes_root, tmp_path):
    policy = tmp_path / "HOST-POLICY.md"
    policy.write_text("FROZEN POLICY", encoding="utf-8")
    _configure(hermes_root, policy)
    tool_call = SimpleNamespace(
        id="call-1",
        type="function",
        function=SimpleNamespace(name="terminal", arguments='{"command":"pwd"}'),
    )
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _response(tool_calls=[tool_call]),
        _response(),
    ]
    runner = _runner(client)
    runner.env = MagicMock()
    runner.env.execute.return_value = {"output": "/tmp\n", "returncode": 0}
    runner._create_env = MagicMock()
    runner._cleanup_env = MagicMock()

    runner.run_task("two-iteration task")

    first, second = _system_messages(client)
    assert first.encode("utf-8") == second.encode("utf-8")


def test_sequential_tasks_observe_policy_change(hermes_root, tmp_path):
    policy = tmp_path / "HOST-POLICY.md"
    first_raw = b"FIRST POLICY\n"
    second_raw = b"SECOND POLICY\n"
    policy.write_bytes(first_raw)
    _configure(hermes_root, policy)
    client = MagicMock()
    client.chat.completions.create.side_effect = [_response(), _response()]
    runner = _runner(client)
    runner._create_env = MagicMock()
    runner._cleanup_env = MagicMock()

    runner.run_task("first task")
    policy.write_bytes(second_raw)
    runner.run_task("second task")

    first, second = _system_messages(client)
    assert hashlib.sha256(first_raw).hexdigest() in first
    assert hashlib.sha256(second_raw).hexdigest() in second
    assert first != second


def test_trajectory_redacts_policy_content_but_records_generation(hermes_root, tmp_path):
    policy = tmp_path / "HOST-POLICY.md"
    policy_text = "SECRET POLICY TRAJECTORY MARKER"
    policy.write_text(policy_text, encoding="utf-8")
    _configure(hermes_root, policy)
    client = MagicMock()
    client.chat.completions.create.return_value = _response()
    runner = _runner(client)
    runner._create_env = MagicMock()
    runner._cleanup_env = MagicMock()

    result = runner.run_task("ordinary task")

    serialized = json.dumps(result, ensure_ascii=False)
    assert policy_text not in serialized
    assert str(policy.resolve()) not in serialized
    digest = hashlib.sha256(policy_text.encode()).hexdigest()
    assert result["metadata"]["trusted_policy_present"] is True
    assert result["metadata"]["trusted_policy_sha256"] == digest
    assert digest in result["conversations"][0]["value"]
    assert "content applied to this task is intentionally omitted" in (
        result["conversations"][0]["value"]
    )


def test_batch_does_not_serialize_policy_load_error(hermes_root, tmp_path):
    policy = tmp_path / "MISSING-HOST-POLICY.md"
    _configure(hermes_root, policy)
    client = MagicMock()
    runner = _runner(client)
    runner._create_env = MagicMock()
    output = tmp_path / "trajectories.jsonl"

    with pytest.raises(GlobalInstructionsError):
        runner.run_batch(["must not start"], str(output))

    assert output.read_text(encoding="utf-8") == ""
    runner._create_env.assert_not_called()
    client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize(
    ("env_type", "environment_label"),
    [("docker", "Docker"), ("modal", "Modal")],
)
def test_remote_boundary_disclaims_host_access(
    hermes_root, env_type, environment_label
):
    client = MagicMock()
    client.chat.completions.create.return_value = _response()
    runner = _runner(client, env_type=env_type)
    runner._create_env = MagicMock()
    runner._cleanup_env = MagicMock()

    runner.run_task("ordinary task")

    system_prompt = _system_messages(client)[0]
    assert environment_label in system_prompt
    assert "does not inherit the host filesystem or credentials" in system_prompt
    assert "not copied or mounted" in system_prompt


def test_local_boundary_identifies_host_surface(hermes_root):
    client = MagicMock()
    client.chat.completions.create.return_value = _response()
    runner = _runner(client, env_type="local")
    runner._create_env = MagicMock()
    runner._cleanup_env = MagicMock()

    runner.run_task("ordinary task")

    assert "terminal executes directly on the local host surface" in _system_messages(client)[0]
