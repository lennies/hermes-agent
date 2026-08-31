"""Cross-process mutual exclusion for in-flight Hermes updates.

Three different surfaces can start an update of the same install tree:

* ``hermes update`` from a terminal,
* the dashboard's Update button (``POST /api/hermes/update`` →
  ``_spawn_hermes_action(["update"])``, detached),
* the desktop's Update button, which hands off to the Tauri
  ``hermes-setup --update`` and, on its failure screen, to install-mode
  bootstrap (``install.ps1`` / ``install.sh``).

Until now only the Tauri updater published an "update in progress" marker
(``UpdateMarkerGuard`` in ``apps/bootstrap-installer/src-tauri/src/update.rs``),
and only the Electron desktop consumed it (``electron/update-marker.ts``, to
gate local backend startup). Nothing stopped two *updaters* from running at
once — so a dashboard-spawned ``hermes update`` and an installer-driven
``git checkout`` could mutate the same checkout concurrently, rewriting source
under a live interpreter and leaving the tree half-updated.

This module makes that same marker the single lock for **all** update
entrypoints instead of adding a fourth mechanism. The first two lines remain
byte-compatible with deployed Rust and Electron readers:

    <install-root>/.hermes-update-in-progress
        body: "<pid>\\n<started_at_unix>\\n<ownership_token>"

New writers claim with create-new, fail closed on acquisition errors, and
remove only the exact inode and payload they created. A live PID is never
stolen based on age alone.

One layering wrinkle: the Tauri updater holds this marker for its WHOLE run and
then spawns ``hermes update`` as a child stage. Without a handoff the child
sees its own parent's live marker and refuses — the GUI update deadlocks
against itself on every attempt ("Hermes is still running", retry forever).
Two mechanisms recognize the orchestrating parent, and either suffices:

* The updater exports :data:`HANDOFF_PID_ENV` and :data:`HANDOFF_TOKEN_ENV`;
  both must exactly match the live marker before the child runs under it.
* A tokenless live holder that is a *process ancestor* of ours is accepted
  only as a legacy orchestrator. This is the load-bearing path for the fleet:
  the staged
  ``hermes-setup`` binary under ``~/.hermes`` is only refreshed by a full
  installer run (``copy_self_to_hermes_home`` deliberately no-ops during
  ``--update``), so every desktop whose staged updater predates the
  HANDOFF_PID_ENV export runs an old parent against a new child. Without the
  ancestry check those users get exit 2 ("Hermes is still running") on every
  GUI update forever, with no Hermes process actually running.
"""

from __future__ import annotations

import logging
import os
import secrets
import stat
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Keep in sync with UPDATE_MARKER_MAX_AGE_MS in
# apps/desktop/electron/update-marker.ts as a display/test boundary. Lease
# ownership itself is liveness-based; neither reader steals a live owner by age.
UPDATE_MARKER_MAX_AGE_SECONDS = 20 * 60

MARKER_NAME = ".hermes-update-in-progress"

# Set by an orchestrating updater (the Tauri `hermes-setup --update` flow) to
# its own pid before spawning `hermes update` as a child stage. The parent
# holds the marker for its whole run, so without this the child refuses its
# own parent's lock and the GUI update can never complete. See update_child_env
# in apps/bootstrap-installer/src-tauri/src/update.rs — keep the name in sync.
HANDOFF_PID_ENV = "HERMES_UPDATE_HANDOFF_PID"
HANDOFF_TOKEN_ENV = "HERMES_UPDATE_LEASE_TOKEN"

# Exit code meaning "another updater/instance owns this install right now".
# Already the de-facto contract: the Windows shim + venv-holder guards in
# _cmd_update_impl exit 2, and the Tauri updater matches on it
# (UPDATE_EXIT_CONCURRENT in apps/bootstrap-installer/src-tauri/src/update.rs)
# to show "Hermes is still running" instead of a generic failure. Naming it
# here keeps the concurrent-update refusal on that same understood contract.
UPDATE_EXIT_CONCURRENT = 2


def update_marker_path() -> Path:
    """Path of the shared update marker.

    Named-profile ``HERMES_HOME`` values resolve to their canonical install
    root, so every updater contends on one lease for one checkout/runtime.
    """
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / MARKER_NAME


def _pid_alive(pid: int) -> bool:
    """True when a process with ``pid`` currently exists.

    Delegates to :func:`gateway.status._pid_exists`, the project's existing
    no-kill probe. Do NOT hand-roll this with ``os.kill(pid, 0)``: on Windows
    that is not a no-op — CPython routes ``sig=0`` to
    ``GenerateConsoleCtrlEvent``, which Ctrl+C's the target's whole console
    process group (bpo-14484). A liveness check that killed the updater it was
    asking about would be a spectacular way to fix a concurrency bug.

    Any pid we cannot evaluate counts as live. Probe failure is not authority
    to delete another updater's lease or start a concurrent mutation.
    """
    if pid <= 0:
        return False
    try:
        from gateway.status import _pid_exists

        return bool(_pid_exists(pid))
    except Exception as exc:
        # Import failure or an unusable pid (for example, larger than the
        # platform's pid_t) is an unknown owner, not proof of death.
        logger.debug("Could not probe pid %s: %s", pid, exc)
        return True


def _handoff_pid() -> int | None:
    """Pid of the orchestrating updater that spawned us, if any.

    Read from :data:`HANDOFF_PID_ENV`. Malformed values count as absent —
    a broken handoff must fall back to the normal refusal, never crash.
    """
    raw = os.environ.get(HANDOFF_PID_ENV, "").strip()
    if not raw:
        return None
    try:
        pid = int(raw)
    except ValueError:
        return None
    return pid if pid > 0 else None


def _is_ancestor_pid(pid: int) -> bool:
    """True when ``pid`` is a live ancestor (parent chain) of this process.

    The orchestrating updater spawns ``hermes update`` as a (grand)child, so a
    live marker owned by one of our ancestors can only be the claim we are
    already running under — an unrelated concurrent updater is never in our
    parent chain. This heals the fleet of staged ``hermes-setup`` binaries
    that predate the HANDOFF_PID_ENV export and can never send it.

    Never includes our own pid, and any failure counts as "not an ancestor":
    an unprovable ancestry must fall back to the normal refusal.
    """
    if pid <= 0:
        return False
    try:
        import psutil

        return any(parent.pid == pid for parent in psutil.Process().parents())
    except Exception as exc:
        logger.debug("Could not walk process ancestry for pid %s: %s", pid, exc)
        return False


@dataclass(frozen=True)
class UpdateHolder:
    """An update marker that makes the install lease unavailable.

    ``pid`` is ``None`` when the marker exists but its owner cannot be proven.
    That state is deliberately blocking: malformed, partial, unreadable, and
    dead-owner markers require explicit operator recovery and are never stolen
    by an ordinary updater.
    """

    pid: int | None
    age_seconds: float | None
    token: str | None = None
    unverifiable: bool = False


@dataclass(frozen=True)
class _MarkerSnapshot:
    raw: bytes
    identity: tuple[int, int]


def _read_marker_snapshot(marker: Path) -> _MarkerSnapshot | None:
    descriptor = None
    try:
        descriptor = os.open(marker, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        metadata = os.fstat(descriptor)
        if not _marker_metadata_safe(metadata):
            return None
        raw = b""
        while len(raw) <= 1024:
            chunk = os.read(descriptor, 1025 - len(raw))
            if not chunk:
                break
            raw += chunk
        if len(raw) > 1024:
            return None
        return _MarkerSnapshot(raw=raw, identity=(metadata.st_dev, metadata.st_ino))
    except OSError:
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def read_live_update(*, path: Path | None = None) -> UpdateHolder | None:
    """Return the marker blocking the lock, or ``None`` only when absent.

    Mirrors ``readLiveUpdateMarker`` in ``electron/update-marker.ts``. A dead,
    malformed, partial, or unreadable marker is an unverifiable owner and is
    preserved. Ordinary acquisition never performs stale recovery: a
    check-then-unlink cleanup can delete a replacement claim created between
    those operations. Never raises.
    """
    marker = path or update_marker_path()
    snapshot = _read_marker_snapshot(marker)
    if snapshot is None:
        return (
            UpdateHolder(pid=None, age_seconds=None, unverifiable=True)
            if os.path.lexists(marker)
            else None
        )
    try:
        raw = snapshot.raw.decode("ascii")
    except UnicodeDecodeError:
        return UpdateHolder(pid=None, age_seconds=None, unverifiable=True)

    lines = raw.splitlines()
    try:
        pid = int(lines[0].strip())
    except (IndexError, ValueError):
        pid = -1
    try:
        started_at = float(lines[1].strip())
    except (IndexError, ValueError):
        started_at = float("-inf")

    if pid <= 0 or not (started_at > 0 and started_at != float("inf")):
        return UpdateHolder(pid=None, age_seconds=None, unverifiable=True)
    age = time.time() - started_at
    # A live updater remains authoritative even after the UI age ceiling.
    # Long dependency rebuilds are unusual but valid; stealing their lease is
    # worse than requiring operator recovery for a reused PID.
    if not _pid_alive(pid):
        return UpdateHolder(pid=None, age_seconds=None, unverifiable=True)

    token = lines[2].strip() if len(lines) > 2 and lines[2].strip() else None
    return UpdateHolder(pid=pid, age_seconds=age, token=token)


def describe_holder(holder: UpdateHolder | None) -> str:
    """One-line, user-facing explanation of who holds the update lock."""
    if holder is None or holder.unverifiable or holder.pid is None:
        return (
            "✗ The Hermes update lease exists but its owner cannot be verified.\n"
            "\n"
            "  Nothing was changed. Keep the install stopped and remove the\n"
            "  .hermes-update-in-progress marker only after confirming that no\n"
            "  terminal, dashboard, desktop, or installer update is running."
        )
    assert holder.age_seconds is not None
    minutes, seconds = divmod(int(max(holder.age_seconds, 0)), 60)
    elapsed = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    return (
        f"✗ Another Hermes update is already running (PID {holder.pid}, "
        f"started {elapsed} ago).\n"
        "\n"
        "  Two updates mutating the same checkout corrupt it: one rewrites\n"
        "  source while the other is mid-install. Wait for it to finish, or\n"
        "  close the window/dashboard tab that started it, then retry."
    )


def _marker_metadata_safe(metadata: os.stat_result) -> bool:
    """Validate marker ownership without rejecting legacy 0644 markers."""
    if not stat.S_ISREG(metadata.st_mode):
        return False
    geteuid = getattr(os, "geteuid", None)
    if geteuid is not None and metadata.st_uid != geteuid():
        return False
    return True


class UpdateLock:
    """Context manager owning the shared update marker for this process.

    ``acquired`` is False when another live update already holds it — callers
    decide whether that's a hard refusal (CLI/dashboard) or a wait. Releasing
    only removes the marker when *we* still own it, so a marker rewritten by a
    handoff partner or replacement owner is never deleted by stale cleanup.
    """

    def __init__(self, *, path: Path | None = None) -> None:
        self.path = path or update_marker_path()
        self.acquired = False
        self.holder: UpdateHolder | None = None
        self._owned_identity: tuple[int, int] | None = None
        self._owned_payload: bytes | None = None

    def acquire(self) -> bool:
        """Claim the lock. Returns False (and sets ``holder``) if it's taken.

        A token-bearing handoff must match both the explicit PID and token. A
        tokenless live ancestor is accepted only for deployed staged updaters
        that predate the token contract. The child never rewrites or releases
        the parent's marker.
        """
        existing = read_live_update(path=self.path)
        if existing is not None:
            handoff_token = os.environ.get(HANDOFF_TOKEN_ENV, "").strip()
            token_matches = bool(
                existing.token and handoff_token and secrets.compare_digest(existing.token, handoff_token)
            )
            if (existing.pid == _handoff_pid() and token_matches) or (
                existing.pid is not None
                and existing.token is None
                and _is_ancestor_pid(existing.pid)
            ):
                return True
            self.holder = existing
            return False
        descriptor = None
        created = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("Could not prepare update marker %s: %s", self.path, exc)
            return False
        try:
            descriptor = os.open(
                self.path,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            metadata = os.fstat(descriptor)
            if not _marker_metadata_safe(metadata):
                raise OSError("update marker is unsafe")
            created = True
            self._owned_identity = (metadata.st_dev, metadata.st_ino)
            token = secrets.token_hex(24)
            payload = f"{os.getpid()}\n{int(time.time())}\n{token}\n".encode("ascii")
            self._owned_payload = payload
            written = 0
            while written < len(payload):
                count = os.write(descriptor, payload[written:])
                if count <= 0:
                    raise OSError("update marker write failed")
                written += count
            os.fsync(descriptor)
            published = _read_marker_snapshot(self.path)
            if (
                published is None
                or published.identity != self._owned_identity
                or published.raw != self._owned_payload
            ):
                raise OSError("update marker publication could not be verified")
        except FileExistsError:
            # Another compliant owner won the atomic create after our first
            # read. Re-read it and refuse instead of overwriting its lease.
            self.holder = read_live_update(path=self.path)
            if self.holder is not None:
                return False
            logger.debug("Update marker appeared concurrently: %s", self.path)
            return False
        except OSError as exc:
            logger.error("Could not write update marker %s: %s", self.path, exc)
            if created:
                # Only clean up the claim opened by this process. Compliant
                # contenders never remove an unowned pathname, so once our
                # create-new succeeds no replacement owner can appear until
                # this exact claim is released.
                snapshot = _read_marker_snapshot(self.path)
                if (
                    snapshot is not None
                    and snapshot.identity == self._owned_identity
                    and snapshot.raw == self._owned_payload
                ):
                    self.acquired = True
                    self.release()
            return False
        finally:
            if descriptor is not None:
                os.close(descriptor)
        self.acquired = True
        return True

    def release(self) -> None:
        """Drop the marker if this process still owns it. Never raises."""
        if not self.acquired:
            return
        self.acquired = False
        descriptor = None
        try:
            descriptor = os.open(
                self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            metadata = os.fstat(descriptor)
            if self._owned_identity != (metadata.st_dev, metadata.st_ino):
                return
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                descriptor = None
                raw = handle.read(256)
            raw_bytes = raw.encode("utf-8")
        except OSError:
            return
        finally:
            if descriptor is not None:
                os.close(descriptor)
        if self._owned_payload != raw_bytes:
            return
        try:
            self.path.unlink()
        except OSError:
            pass
        finally:
            self._owned_identity = None
            self._owned_payload = None

    def __enter__(self) -> "UpdateLock":
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
