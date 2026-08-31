"""Self-contained Kanban clones for workspace-only worker confinement."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _make_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    origin = "git@github.com:example/repository.git"
    _git(repo, "remote", "add", "origin", origin)
    return repo, origin


def test_isolated_clone_is_internal_exact_and_reusable(
    kanban_home, tmp_path
):
    source, origin = _make_repo(tmp_path)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="confined delivery",
            workspace_kind="isolated_clone",
            workspace_path=str(source),
            branch_name="delivery/test",
        )
        task = kb.get_task(conn, task_id)

    workspace, branch = kb._resolve_isolated_clone_workspace(task)

    assert workspace == (kb.workspaces_root() / task_id).resolve()
    assert branch == "delivery/test"
    assert (workspace / ".git").is_dir()
    assert _git(workspace, "rev-parse", "--show-toplevel") == str(workspace)
    assert _git(workspace, "rev-parse", "--git-common-dir") == ".git"
    assert _git(workspace, "branch", "--show-current") == branch
    assert _git(workspace, "config", "--get", "remote.origin.url") == origin
    assert (workspace / "README.md").read_text(encoding="utf-8") == "base\n"

    reused, reused_branch = kb._resolve_isolated_clone_workspace(task)
    assert reused == workspace
    assert reused_branch == branch


def test_isolated_clone_rejects_existing_branch_drift(kanban_home, tmp_path):
    source, _origin = _make_repo(tmp_path)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="confined delivery",
            workspace_kind="isolated_clone",
            workspace_path=str(source),
            branch_name="delivery/test",
        )
        task = kb.get_task(conn, task_id)

    workspace, _branch = kb._resolve_isolated_clone_workspace(task)
    _git(workspace, "checkout", "-b", "wrong-branch")

    with pytest.raises(ValueError, match="exact branch"):
        kb._resolve_isolated_clone_workspace(task)


def test_isolated_clone_rejects_source_without_origin(kanban_home, tmp_path):
    source, _origin = _make_repo(tmp_path)
    _git(source, "remote", "remove", "origin")
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="confined delivery",
            workspace_kind="isolated_clone",
            workspace_path=str(source),
        )
        task = kb.get_task(conn, task_id)

    with pytest.raises(ValueError, match="canonical origin"):
        kb._resolve_isolated_clone_workspace(task)


def test_isolated_clone_rejects_http_origin_with_embedded_credentials(
    kanban_home, tmp_path
):
    source, _origin = _make_repo(tmp_path)
    _git(
        source,
        "remote",
        "set-url",
        "origin",
        "https://delivery-token@example.invalid/repository.git",
    )
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="confined delivery",
            workspace_kind="isolated_clone",
            workspace_path=str(source),
        )
        task = kb.get_task(conn, task_id)

    with pytest.raises(ValueError, match="embedded credentials"):
        kb._resolve_isolated_clone_workspace(task)
