from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.profiles import (
    ProfileDispatchDeniedError,
    profiles_to_serve,
)


def _profile_homes(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    governed = root / "profiles" / "governed"
    governed.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root, governed


@pytest.mark.asyncio
async def test_multiplex_message_turn_rechecks_resolved_profile(
    tmp_path, monkeypatch
):
    _root, governed = _profile_homes(tmp_path, monkeypatch)
    assert "governed" in dict(profiles_to_serve(multiplex=True))
    (governed / "profile.yaml").write_text(
        "dispatch_mode: controller-only\n", encoding="utf-8"
    )
    runner = SimpleNamespace(
        config=SimpleNamespace(multiplex_profiles=True),
        _resolve_profile_home_for_source=lambda _source: governed,
        _run_agent_inner=AsyncMock(),
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat",
        chat_type="dm",
    )
    with pytest.raises(ProfileDispatchDeniedError, match="authenticated controller"):
        await GatewayRunner._run_agent(
            runner,
            "message",
            "context",
            [],
            source,
            "session",
        )

    runner._run_agent_inner.assert_not_awaited()


def test_api_agent_creation_rechecks_active_profile(tmp_path, monkeypatch):
    _root, governed = _profile_homes(tmp_path, monkeypatch)
    (governed / "profile.yaml").write_text(
        "dispatch_mode: controller-only\n", encoding="utf-8"
    )
    adapter = APIServerAdapter(PlatformConfig(enabled=True))

    with adapter._profile_scope("governed"):
        with pytest.raises(
            ProfileDispatchDeniedError, match="authenticated controller"
        ):
            adapter._create_agent()
