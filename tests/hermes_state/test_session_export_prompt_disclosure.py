"""Portable session exports never disclose persisted trusted prompt state."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from gateway.slash_commands import GatewaySlashCommandsMixin
from hermes_cli.session_export_md import redact_session_data
from hermes_state import AsyncSessionDB, SessionDB


HOST_POLICY = """<trusted-host-policy source=\"/Users/operator/AGENTS.md\">
ELFBOT_HOST_POLICY_V1
sha256: 0123456789abcdef
</trusted-host-policy>"""


def _assert_prompt_state_absent(exported):
    assert "system_prompt" not in exported
    assert "system_prompt_hash" not in exported
    assert HOST_POLICY not in json.dumps(exported)


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


@pytest.mark.parametrize("redact", [False, True], ids=["normal", "redacted"])
def test_export_session_omits_trusted_prompt_state(db, redact):
    db.create_session("single", "cli", system_prompt=HOST_POLICY)
    db.append_message("single", role="user", content="hello")

    exported = db.export_session("single")
    if redact:
        exported = redact_session_data(exported)

    _assert_prompt_state_absent(exported)
    assert exported["messages"][0]["content"] == "hello"


def test_export_session_lineage_omits_prompt_state_from_every_shape(db):
    db.create_session("root", "cli", system_prompt=HOST_POLICY)
    db.append_message("root", role="user", content="before compression")
    db.end_session("root", "compression")
    db.create_session(
        "tip",
        "cli",
        parent_session_id="root",
        system_prompt=HOST_POLICY + "\ntip",
    )
    db.append_message("tip", role="assistant", content="after compression")

    exported = db.export_session_lineage("tip")

    _assert_prompt_state_absent(exported)
    assert exported["lineage_session_ids"] == ["root", "tip"]
    assert len(exported["segments"]) == 2
    for segment in exported["segments"]:
        _assert_prompt_state_absent(segment)


def test_export_all_omits_prompt_state_from_every_session(db):
    for session_id, source in (("cli-session", "cli"), ("gateway-session", "telegram")):
        db.create_session(
            session_id, source, system_prompt=f"{HOST_POLICY}\n{session_id}"
        )
        db.append_message(session_id, role="user", content=session_id)

    exported = db.export_all()

    assert {session["id"] for session in exported} == {"cli-session", "gateway-session"}
    for session in exported:
        _assert_prompt_state_absent(session)


@pytest.mark.asyncio
async def test_gateway_save_json_delivers_no_trusted_prompt_state(db):
    db.create_session("gateway-save", "telegram", system_prompt=HOST_POLICY)
    db.append_message("gateway-save", role="user", content="share this conversation")
    delivered = {}

    class SessionStore:
        async def get_or_create_session(self, _source):
            return SimpleNamespace(session_id="gateway-save")

    class Adapter:
        async def send_document(self, *, file_path, **_kwargs):
            delivered["payload"] = json.loads(
                Path(file_path).read_text(encoding="utf-8")
            )

    class Runner(GatewaySlashCommandsMixin):
        async_session_store = SessionStore()
        _session_db = AsyncSessionDB(db)

        def get_adapter(self, _platform):
            return Adapter()

    event = MessageEvent(
        text="/save json",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1"),
    )

    result = await Runner()._handle_save_command(event)

    assert result == "Export complete."
    _assert_prompt_state_absent(delivered["payload"])
    assert delivered["payload"]["messages"][0]["content"] == "share this conversation"
