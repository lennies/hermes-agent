from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from toolsets import resolve_toolset, validate_toolset
from tools.registry import registry


def _context(home: Path, name: str) -> tuple[PluginContext, PluginManager]:
    manager = PluginManager(scope_key=str(home.resolve()))
    return (
        PluginContext(PluginManifest(name=name, key=name), manager),
        manager,
    )


def test_plugin_composite_toolset_is_profile_scoped_and_disposable(tmp_path):
    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    home_a.mkdir()
    home_b.mkdir()
    context, manager = _context(home_a, "closed-surface")

    handle = context.register_toolset(
        name="closed-review",
        description="One bounded review surface",
        tools=["terminal", "read_file"],
        includes=[],
    )
    assert handle is not None

    token = set_hermes_home_override(home_a)
    try:
        assert validate_toolset("closed-review") is True
        assert set(resolve_toolset("closed-review")) == {"terminal", "read_file"}
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(home_b)
    try:
        assert validate_toolset("closed-review") is False
        assert resolve_toolset("closed-review") == []
    finally:
        reset_hermes_home_override(token)

    handle.dispose()
    assert "closed-review" not in manager._plugin_toolset_names
    assert registry.get_registered_toolset(
        "closed-review", scope=str(home_a.resolve())
    ) is None


def test_plugin_composite_toolset_rejects_core_collision(tmp_path):
    context, _manager = _context(tmp_path, "collision")
    with pytest.raises(ValueError, match="built-in toolset"):
        context.register_toolset(
            name="terminal",
            description="Must not replace core",
            tools=["read_file"],
        )
