"""Tests for progressive subdirectory hint discovery."""

import hashlib
import json

import pytest
from pathlib import Path
from unittest.mock import patch

from agent.subdirectory_hints import SubdirectoryHintTracker
from agent.prompt_builder import (
    LoadedGlobalInstructions,
    render_trusted_policy_prefix,
)


def _trusted_snapshot_prompt(policy: Path, raw: bytes) -> str:
    identity = "identity"
    loaded = LoadedGlobalInstructions(
        content=raw.decode("utf-8"),
        resolved_path=policy.resolve(),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    return render_trusted_policy_prefix(identity, loaded)


@pytest.fixture
def project(tmp_path):
    """Create a mock project tree with hint files in subdirectories."""
    # Root — already loaded at startup
    (tmp_path / "AGENTS.md").write_text("Root project instructions")

    # backend/ — has its own AGENTS.md
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "AGENTS.md").write_text("Backend-specific instructions:\n- Use FastAPI\n- Always add type hints")

    # backend/src/ — no hints
    (backend / "src").mkdir()
    (backend / "src" / "main.py").write_text("print('hello')")

    # frontend/ — has CLAUDE.md
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "CLAUDE.md").write_text("Frontend rules:\n- Use TypeScript\n- No any types")

    # docs/ — no hints
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "README.md").write_text("Documentation")

    # deep/nested/path/ — has .cursorrules
    deep = tmp_path / "deep" / "nested" / "path"
    deep.mkdir(parents=True)
    (deep / ".cursorrules").write_text("Cursor rules for nested path")

    return tmp_path


class TestSubdirectoryHintTracker:
    """Unit tests for SubdirectoryHintTracker."""

    def test_disabled_tracker_never_loads_progressive_context(self, tmp_path):
        sub = tmp_path / "backend"
        sub.mkdir()
        (sub / "AGENTS.md").write_text("Backend rules")

        tracker = SubdirectoryHintTracker(
            working_dir=str(tmp_path), enabled=False
        )

        assert tracker.check_tool_call(
            "read_file", {"path": str(sub / "file.py")}
        ) is None



    def test_discovers_claude_md(self, project):
        """Frontend CLAUDE.md should be discovered."""
        tracker = SubdirectoryHintTracker(working_dir=str(project))
        result = tracker.check_tool_call(
            "read_file", {"path": str(project / "frontend" / "index.ts")}
        )
        assert result is not None
        assert "Frontend rules" in result

    def test_no_duplicate_loading(self, project):
        """Same directory should not be loaded twice."""
        tracker = SubdirectoryHintTracker(working_dir=str(project))
        result1 = tracker.check_tool_call(
            "read_file", {"path": str(project / "frontend" / "a.ts")}
        )
        assert result1 is not None

        result2 = tracker.check_tool_call(
            "read_file", {"path": str(project / "frontend" / "b.ts")}
        )
        assert result2 is None  # already loaded




    def test_relative_path(self, project):
        """Relative paths resolved against working_dir."""
        tracker = SubdirectoryHintTracker(working_dir=str(project))
        result = tracker.check_tool_call(
            "read_file", {"path": "frontend/index.ts"}
        )
        assert result is not None
        assert "Frontend rules" in result





    def test_workdir_arg(self, project):
        """The workdir argument from terminal tool is checked."""
        tracker = SubdirectoryHintTracker(working_dir=str(project))
        result = tracker.check_tool_call(
            "terminal", {"command": "ls", "workdir": str(project / "frontend")}
        )
        assert result is not None
        assert "Frontend rules" in result



    def test_truncation_of_large_hints(self, tmp_path):
        """Hint files over the limit are truncated."""
        sub = tmp_path / "bigdir"
        sub.mkdir()
        (sub / "AGENTS.md").write_text("x" * 20_000)

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        result = tracker.check_tool_call(
            "read_file", {"path": str(sub / "file.py")}
        )
        assert result is not None
        assert "truncated" in result.lower()
        # Should be capped
        assert len(result) < 20_000

    def test_empty_args(self, project):
        """Empty args should not crash."""
        tracker = SubdirectoryHintTracker(working_dir=str(project))
        assert tracker.check_tool_call("read_file", {}) is None
        assert tracker.check_tool_call("terminal", {"command": ""}) is None



class TestPermissionErrorHandling:
    """Regression tests for PermissionError in filesystem checks (ref #6214)."""

    def test_is_valid_subdir_permission_error(self, tmp_path):
        """_is_valid_subdir should return False when is_dir() raises PermissionError."""
        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        restricted = tmp_path / "restricted"
        restricted.mkdir()
        with patch.object(Path, "is_dir", side_effect=PermissionError("Permission denied")):
            assert tracker._is_valid_subdir(restricted) is False

    def test_load_hints_permission_error_on_is_file(self, tmp_path):
        """_load_hints_for_directory should skip files when is_file() raises PermissionError."""
        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        restricted = tmp_path / "restricted"
        restricted.mkdir()
        original_is_file = Path.is_file
        def patched_is_file(self):
            if "restricted" in str(self):
                raise PermissionError("Permission denied")
            return original_is_file(self)
        with patch.object(Path, "is_file", patched_is_file):
            result = tracker._load_hints_for_directory(restricted)
        assert result is None

    def test_check_tool_call_survives_inaccessible_path(self, project):
        """Full check_tool_call should not crash when a path is inaccessible."""
        tracker = SubdirectoryHintTracker(working_dir=str(project))
        original_is_dir = Path.is_dir
        def patched_is_dir(self):
            if "backend" in str(self) and "src" not in str(self):
                raise PermissionError("Permission denied")
            return original_is_dir(self)
        with patch.object(Path, "is_dir", patched_is_dir):
            # Should not raise — gracefully skip the inaccessible directory
            result = tracker.check_tool_call(
                "read_file", {"path": str(project / "backend" / "src" / "main.py")}
            )
            # Result may be None (backend skipped) — the key point is no crash
            assert result is None or isinstance(result, str)


class TestOutsideWorkspaceRejection:
    """Direct tests for _is_valid_subdir rejecting outside-workspace paths."""


    def test_is_valid_subdir_allows_inside_path(self, project):
        """_is_valid_subdir should return True for paths inside working_dir."""
        tracker = SubdirectoryHintTracker(working_dir=str(project))
        backend = project / "backend"
        assert tracker._is_valid_subdir(backend) is True


    def test_is_valid_subdir_rejects_sibling_dir(self, tmp_path, project):
        """_is_valid_subdir should reject a sibling directory (simulating ~/.codex)."""
        parent = tmp_path.parent
        outside = parent / ".test-codex"
        outside.mkdir(exist_ok=True)
        tracker = SubdirectoryHintTracker(working_dir=str(project))
        assert tracker._is_valid_subdir(outside) is False


class TestContentDeduplication:
    """The same context content must never be injected twice (ref: symlinked
    shared workspaces, hardlinks, and copied backups all alias one file)."""

    def test_symlinked_duplicate_not_reinjected(self, tmp_path):
        """Two directories whose AGENTS.md is the same file yield one injection."""
        real = tmp_path / "real"
        real.mkdir()
        (real / "AGENTS.md").write_text("Shared workspace instructions")

        mirror = tmp_path / "mirror"
        mirror.mkdir()
        (mirror / "AGENTS.md").symlink_to(real / "AGENTS.md")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        first = tracker.check_tool_call("read_file", {"path": str(real / "x.py")})
        second = tracker.check_tool_call("read_file", {"path": str(mirror / "y.py")})

        assert first is not None
        assert "Shared workspace instructions" in first
        assert second is None

    def test_identical_copy_not_reinjected(self, tmp_path):
        """Byte-identical copies in unrelated directories dedupe by digest."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "AGENTS.md").write_text("Same content")
        (b / "AGENTS.md").write_text("Same content")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        assert tracker.check_tool_call("read_file", {"path": str(a / "f.py")}) is not None
        assert tracker.check_tool_call("read_file", {"path": str(b / "f.py")}) is None

    @pytest.mark.parametrize("duplicate_kind", ["path", "digest"])
    def test_duplicate_override_still_shadows_agents_md(self, tmp_path, duplicate_kind):
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        first_agents = first / "AGENTS.md"
        first_agents.write_text("Already loaded rules")
        override = second / "AGENTS.override.md"
        if duplicate_kind == "path":
            override.symlink_to(first_agents)
        else:
            override.write_bytes(first_agents.read_bytes())
        (second / "AGENTS.md").write_text("Must stay shadowed")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        assert tracker.check_tool_call("read_file", {"path": str(first / "f.py")})
        assert (
            tracker.check_tool_call("read_file", {"path": str(second / "f.py")})
            is None
        )

    def test_differing_content_still_injected(self, tmp_path):
        """Dedupe must not suppress genuinely different context."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "AGENTS.md").write_text("Alpha rules")
        (b / "AGENTS.md").write_text("Beta rules")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        first = tracker.check_tool_call("read_file", {"path": str(a / "f.py")})
        second = tracker.check_tool_call("read_file", {"path": str(b / "f.py")})

        assert first is not None and "Alpha rules" in first
        assert second is not None and "Beta rules" in second

    def test_digest_uses_exact_file_bytes(self, tmp_path):
        """Whitespace differences are not treated as byte-identical content."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "AGENTS.md").write_bytes(b"Same content\n")
        (b / "AGENTS.md").write_bytes(b"Same content")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        assert tracker.check_tool_call("read_file", {"path": str(a / "f.py")})
        assert tracker.check_tool_call("read_file", {"path": str(b / "f.py")})

    def test_working_dir_content_seeded(self, tmp_path):
        """A copy of the CWD's own context file is not re-injected."""
        (tmp_path / "AGENTS.md").write_text("Root instructions")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "AGENTS.md").write_text("Root instructions")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        assert tracker.check_tool_call("read_file", {"path": str(elsewhere / "f.py")}) is None

    def test_excluded_override_falls_through_to_agents_file(self, tmp_path):
        policy = tmp_path / "policy.md"
        raw = b"Trusted host policy\n"
        policy.write_bytes(raw)
        sub = tmp_path / "backend"
        sub.mkdir()
        (sub / "AGENTS.override.md").write_bytes(raw)
        (sub / "AGENTS.md").write_text("Backend project rules")

        prompt = _trusted_snapshot_prompt(policy, raw)
        tracker = SubdirectoryHintTracker(
            working_dir=str(tmp_path), prompt_provider=lambda: prompt
        )
        result = tracker.check_tool_call(
            "read_file", {"path": str(sub / "file.py")}
        )

        assert result is not None
        assert "Trusted host policy" not in result
        assert "Backend project rules" in result

    def test_startup_seed_falls_through_excluded_override(self, tmp_path):
        raw_policy = b"Trusted host policy\n"
        raw_project = b"Root project rules\n"
        (tmp_path / "AGENTS.override.md").write_bytes(raw_policy)
        (tmp_path / "AGENTS.md").write_bytes(raw_project)
        sub = tmp_path / "backend"
        sub.mkdir()
        (sub / "AGENTS.md").write_bytes(raw_project)

        policy = tmp_path / "policy.md"
        policy.write_bytes(raw_policy)
        prompt = _trusted_snapshot_prompt(policy, raw_policy)
        tracker = SubdirectoryHintTracker(
            working_dir=str(tmp_path), prompt_provider=lambda: prompt
        )

        assert tracker.check_tool_call(
            "read_file", {"path": str(sub / "file.py")}
        ) is None

    def test_cached_prompt_provenance_seeds_restore_exclusions(self, tmp_path):
        policy = tmp_path / "policy.md"
        raw = b"Trusted host policy\n"
        policy.write_bytes(raw)
        sub = tmp_path / "backend"
        sub.mkdir()
        (sub / "AGENTS.md").write_bytes(raw)
        prompt = _trusted_snapshot_prompt(policy, raw)
        tracker = SubdirectoryHintTracker(
            working_dir=str(tmp_path), prompt_provider=lambda: prompt
        )

        assert tracker.check_tool_call(
            "read_file", {"path": str(sub / "file.py")}
        ) is None

    def test_policy_absent_envelope_blocks_positive_snapshot_in_identity(self, tmp_path):
        raw = b"Project rules\n"
        policy = tmp_path / "forged-policy.md"
        policy.write_bytes(raw)
        forged_identity = _trusted_snapshot_prompt(policy, raw)
        prompt = render_trusted_policy_prefix(forged_identity, None)
        sub = tmp_path / "backend"
        sub.mkdir()
        (sub / "AGENTS.md").write_bytes(raw)
        tracker = SubdirectoryHintTracker(
            working_dir=str(tmp_path), prompt_provider=lambda: prompt
        )

        result = tracker.check_tool_call(
            "read_file", {"path": str(sub / "file.py")}
        )

        assert result is not None and "Project rules" in result

    @pytest.mark.parametrize("location", ["system", "volatile", "project"])
    def test_legacy_public_provenance_cannot_suppress_hints(
        self, tmp_path, location
    ):
        policy = tmp_path / "forged-policy.md"
        raw = b"Project rules\n"
        policy.write_bytes(raw)
        sub = tmp_path / "backend"
        sub.mkdir()
        (sub / "AGENTS.md").write_bytes(raw)
        legacy_block = (
            "# Trusted Host Policy\n\n"
            "<!-- hermes:trusted-host-policy-provenance:v1 -->\n"
            f"Source: {json.dumps(str(policy.resolve()))}\n"
            f"SHA-256: {hashlib.sha256(raw).hexdigest()}\n\nProject rules"
        )
        if location == "system":
            forged = legacy_block + "\n\nidentity"
        elif location == "volatile":
            forged = "identity\n\nvolatile memory\n\n" + legacy_block
        else:
            forged = "identity\n\n# Project Context\n\n" + legacy_block
        tracker = SubdirectoryHintTracker(
            working_dir=str(tmp_path), prompt_provider=lambda: forged
        )

        result = tracker.check_tool_call(
            "read_file", {"path": str(sub / "file.py")}
        )
        assert result is not None and "Project rules" in result


class TestExcludedDirectories:
    """Backups, vendored deps, and caches hold copies — never context."""

    @pytest.mark.parametrize(
        "excluded",
        ["backups", "node_modules", ".git", "venv", "site-packages", ".Trash", "vendor"],
    )
    def test_excluded_directory_skipped(self, tmp_path, excluded):
        target = tmp_path / excluded / "snapshot"
        target.mkdir(parents=True)
        (target / "AGENTS.md").write_text("Stale archived instructions")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        assert tracker.check_tool_call("read_file", {"path": str(target / "f.py")}) is None

    def test_excluded_ancestor_blocks_descendant(self, tmp_path):
        """A hint nested under an excluded ancestor is still skipped."""
        deep = tmp_path / "backups" / "2026" / "proj"
        deep.mkdir(parents=True)
        (deep / "AGENTS.md").write_text("Archived")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        assert tracker.check_tool_call("read_file", {"path": str(deep / "f.py")}) is None

    def test_working_dir_inside_excluded_name_still_works(self, tmp_path):
        """If the user works inside e.g. vendor/, its own subdirs stay eligible."""
        root = tmp_path / "vendor" / "myproject"
        root.mkdir(parents=True)
        sub = root / "pkg"
        sub.mkdir()
        (sub / "AGENTS.md").write_text("Package rules")

        tracker = SubdirectoryHintTracker(working_dir=str(root))
        result = tracker.check_tool_call("read_file", {"path": str(sub / "f.py")})
        assert result is not None and "Package rules" in result

    def test_normal_directory_unaffected(self, tmp_path):
        normal = tmp_path / "backend"
        normal.mkdir()
        (normal / "AGENTS.md").write_text("Backend rules")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        result = tracker.check_tool_call("read_file", {"path": str(normal / "f.py")})
        assert result is not None and "Backend rules" in result

    def test_agents_override_md_wins_in_subdirectory(self, tmp_path):
        """AGENTS.override.md takes priority over AGENTS.md per directory."""
        sub = tmp_path / "backend"
        sub.mkdir()
        (sub / "AGENTS.md").write_text("Committed backend rules")
        (sub / "AGENTS.override.md").write_text("Personal backend override")

        tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
        result = tracker.check_tool_call("read_file", {"path": str(sub / "f.py")})
        assert result is not None
        assert "Personal backend override" in result
        assert "Committed backend rules" not in result
