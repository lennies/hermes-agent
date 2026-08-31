"""Exact-task dispatch contract for deterministic controller-only profiles."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.profiles import create_profile, set_profile_dispatch_mode


@pytest.fixture
def controller_board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    create_profile("delivery-maintainer", no_alias=True, no_skills=True)
    set_profile_dispatch_mode("delivery-maintainer", "controller-only")
    return home


def test_controller_dispatch_spawns_only_the_exact_authorized_task(controller_board):
    calls: list[tuple[str, str]] = []

    def authorize(stage, task):
        calls.append((stage, task.id))
        return True

    spawned: list[str] = []

    def spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 424242

    with kb.connect() as conn:
        target = kb.create_task(
            conn, title="target", assignee="delivery-maintainer"
        )
        sibling = kb.create_task(
            conn, title="sibling", assignee="delivery-maintainer"
        )

        generic = kb.dispatch_once(conn, dry_run=True)
        assert generic.spawned == []
        assert set(generic.skipped_nonspawnable) == {target, sibling}

        result = kb.dispatch_controller_task(
            conn,
            task_id=target,
            expected_assignee="delivery-maintainer",
            authorize_dispatch=authorize,
            spawn_fn=spawn,
        )
        target_task = kb.get_task(conn, target)
        sibling_task = kb.get_task(conn, sibling)

    assert [item[0] for item in result.spawned] == [target]
    assert spawned == [target]
    assert target_task is not None and target_task.status == "running"
    assert target_task.current_run_id is not None
    assert target_task.claim_lock is not None
    assert target_task.worker_pid == 424242
    assert sibling_task is not None and sibling_task.status == "ready"
    assert sibling_task.current_run_id is None
    assert calls == [
        ("pre-claim", target),
        ("pre-spawn", target),
    ]


def test_controller_dispatch_denial_is_fail_closed_before_claim(controller_board):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="target", assignee="delivery-maintainer"
        )
        result = kb.dispatch_controller_task(
            conn,
            task_id=task_id,
            expected_assignee="delivery-maintainer",
            authorize_dispatch=lambda _stage, _task: False,
            spawn_fn=lambda *_args, **_kwargs: pytest.fail("must not spawn"),
        )
        task = kb.get_task(conn, task_id)

    assert result.spawned == []
    assert result.skipped_unauthorized == [task_id]
    assert task is not None and task.status == "ready"
    assert task.current_run_id is None
    assert task.claim_lock is None


def test_controller_dispatch_never_falls_back_to_generic_spawn(controller_board):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="target", assignee="delivery-maintainer"
        )
        with pytest.raises(
            ValueError,
            match="explicit confined spawn callback",
        ):
            kb.dispatch_controller_task(
                conn,
                task_id=task_id,
                expected_assignee="delivery-maintainer",
                authorize_dispatch=lambda *_args: True,
            )
        task = kb.get_task(conn, task_id)

    assert task is not None and task.status == "ready"
    assert task.current_run_id is None
    assert task.claim_lock is None


def test_controller_dispatch_rechecks_authority_at_spawn_boundary(controller_board):
    stages: list[str] = []

    def authorize(stage, _task):
        stages.append(stage)
        return len(stages) < 2

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="target", assignee="delivery-maintainer"
        )
        result = kb.dispatch_controller_task(
            conn,
            task_id=task_id,
            expected_assignee="delivery-maintainer",
            authorize_dispatch=authorize,
            spawn_fn=lambda *_args, **_kwargs: pytest.fail("must not spawn"),
        )
        task = kb.get_task(conn, task_id)
        run = conn.execute(
            "SELECT status, outcome, summary FROM task_runs WHERE task_id = ?",
            (task_id,),
        ).fetchone()

    assert stages == ["pre-claim", "pre-spawn"]
    assert result.spawned == []
    assert result.skipped_unauthorized == [task_id]
    assert task is not None and task.status == "ready"
    assert task.current_run_id is None
    assert task.claim_lock is None
    assert run is not None
    assert run["status"] == "dispatch_authorization_revoked"
    assert run["outcome"] == "dispatch_authorization_revoked"
    assert "authorization revoked" in run["summary"]


def test_controller_dispatch_refuses_assignee_drift(controller_board):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="target", assignee="delivery-maintainer"
        )
        result = kb.dispatch_controller_task(
            conn,
            task_id=task_id,
            expected_assignee="delivery-reviewer",
            authorize_dispatch=lambda *_args: True,
            spawn_fn=lambda *_args, **_kwargs: pytest.fail("must not spawn"),
        )
        task = kb.get_task(conn, task_id)

    assert result.spawned == []
    assert result.skipped_unauthorized == [task_id]
    assert task is not None and task.status == "ready"


def test_controller_dispatch_claim_cas_refuses_concurrent_reassignment(
    controller_board,
):
    stages: list[str] = []

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="target", assignee="delivery-maintainer"
        )

        def authorize(stage, _task):
            stages.append(stage)
            if stage == "pre-claim":
                with kb.connect() as concurrent:
                    assert kb.assign_task(concurrent, task_id, "other")
            return True

        result = kb.dispatch_controller_task(
            conn,
            task_id=task_id,
            expected_assignee="delivery-maintainer",
            authorize_dispatch=authorize,
            spawn_fn=lambda *_args, **_kwargs: pytest.fail("must not spawn"),
        )
        task = kb.get_task(conn, task_id)

    assert stages == ["pre-claim"]
    assert result.spawned == []
    assert result.skipped_unauthorized == [task_id]
    assert task is not None and task.status == "ready"
    assert task.assignee == "other"
    assert task.current_run_id is None
    assert task.claim_lock is None


def test_controller_dispatch_rechecks_profile_mode_before_spawn(controller_board):
    stages: list[str] = []

    def authorize(stage, _task):
        stages.append(stage)
        if stage == "pre-spawn":
            set_profile_dispatch_mode("delivery-maintainer", "disabled")
        return True

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="target", assignee="delivery-maintainer"
        )
        result = kb.dispatch_controller_task(
            conn,
            task_id=task_id,
            expected_assignee="delivery-maintainer",
            authorize_dispatch=authorize,
            spawn_fn=lambda *_args, **_kwargs: pytest.fail("must not spawn"),
        )
        task = kb.get_task(conn, task_id)
        run = conn.execute(
            "SELECT status, outcome FROM task_runs WHERE task_id = ?",
            (task_id,),
        ).fetchone()

    assert stages == ["pre-claim", "pre-spawn"]
    assert result.spawned == []
    assert result.skipped_unauthorized == [task_id]
    assert task is not None and task.status == "ready"
    assert task.current_run_id is None
    assert task.claim_lock is None
    assert run is not None
    assert run["status"] == "dispatch_authorization_revoked"
    assert run["outcome"] == "dispatch_authorization_revoked"


def test_controller_checkpoint_releases_exact_claim_to_same_stable_task(
    controller_board,
):
    digest = "sha256:" + ("a" * 64)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="controller", assignee="delivery-controller"
        )
        claimed = kb.claim_task(conn, task_id, claimer="controller:test")
        assert claimed is not None
        run_id = claimed.current_run_id
        assert run_id is not None

        assert kb.release_controller_claim(
            conn,
            task_id,
            expected_run_id=run_id,
            expected_claim_lock="controller:test",
            checkpoint_digest=digest,
        )
        released = kb.get_task(conn, task_id)
        run = conn.execute(
            "SELECT status, outcome, summary, metadata, ended_at "
            "FROM task_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        event = conn.execute(
            "SELECT kind, payload, run_id FROM task_events "
            "WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()

    assert released is not None
    assert released.status == "ready"
    assert released.current_run_id is None
    assert released.claim_lock is None
    assert run is not None
    assert run["status"] == "controller_checkpoint"
    assert run["outcome"] == "controller_checkpoint"
    assert run["summary"] == digest
    assert digest in run["metadata"]
    assert run["ended_at"] is not None
    assert event is not None
    assert event["kind"] == "controller_checkpoint"
    assert event["run_id"] == run_id
    assert digest in event["payload"]


def test_controller_checkpoint_mismatch_is_fail_closed(controller_board):
    digest = "sha256:" + ("b" * 64)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="controller", assignee="delivery-controller"
        )
        claimed = kb.claim_task(conn, task_id, claimer="controller:test")
        assert claimed is not None and claimed.current_run_id is not None

        assert not kb.release_controller_claim(
            conn,
            task_id,
            expected_run_id=claimed.current_run_id,
            expected_claim_lock="controller:wrong",
            checkpoint_digest=digest,
        )
        retained = kb.get_task(conn, task_id)

    assert retained is not None
    assert retained.status == "running"
    assert retained.current_run_id == claimed.current_run_id
    assert retained.claim_lock == "controller:test"


def test_controller_checkpoint_rejects_non_controller_and_invalid_digest(
    controller_board,
):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="worker", assignee="delivery-maintainer"
        )
        claimed = kb.claim_task(conn, task_id, claimer="controller:test")
        assert claimed is not None and claimed.current_run_id is not None
        assert not kb.release_controller_claim(
            conn,
            task_id,
            expected_run_id=claimed.current_run_id,
            expected_claim_lock="controller:test",
            checkpoint_digest="sha256:" + ("c" * 64),
        )
        with pytest.raises(ValueError, match="sha256"):
            kb.release_controller_claim(
                conn,
                task_id,
                expected_run_id=claimed.current_run_id,
                expected_claim_lock="controller:test",
                checkpoint_digest="not-a-digest",
            )
        with pytest.raises(ValueError, match="sha256"):
            kb.release_controller_claim(
                conn,
                task_id,
                expected_run_id=claimed.current_run_id,
                expected_claim_lock="controller:test",
                checkpoint_digest="sha256:" + ("A" * 64),
            )
        with pytest.raises(ValueError, match="positive integer"):
            kb.release_controller_claim(
                conn,
                task_id,
                expected_run_id=True,
                expected_claim_lock="controller:test",
                checkpoint_digest="sha256:" + ("e" * 64),
            )
