"""Process-level contention checks for every repo-owned update handoff."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
POSIX_HANDOFF = REPOSITORY / "scripts/desktop-update/posix.sh"
WINDOWS_HANDOFF = REPOSITORY / "scripts/desktop-update/windows.ps1"
ELECTRON_MARKER = REPOSITORY / "apps/desktop/electron/update-marker.ts"
RUST_UPDATER = REPOSITORY / "apps/bootstrap-installer/src-tauri/src/update.rs"


def _elfos_lease(root: Path) -> tuple[Path, bytes, tuple[int, int]]:
    marker = root / ".hermes-update-in-progress"
    payload = f"{os.getpid()}\n{int(time.time())}\n{'e' * 48}\n".encode("ascii")
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    return marker, payload, (metadata.st_dev, metadata.st_ino)


def _assert_lease_unchanged(
    marker: Path, payload: bytes, identity: tuple[int, int]
) -> None:
    metadata = marker.stat()
    assert (metadata.st_dev, metadata.st_ino) == identity
    assert marker.read_bytes() == payload


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


@pytest.mark.parametrize("named_profile", [False, True], ids=["root", "named-profile"])
def test_python_update_process_loses_to_elfos_install_global_lease(
    tmp_path, named_profile
):
    marker, payload, identity = _elfos_lease(tmp_path)
    hermes_home = (
        tmp_path / "profiles/delivery-maintainer" if named_profile else tmp_path
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from hermes_cli.update_lock import UpdateLock; "
            "raise SystemExit(7 if UpdateLock().acquire() else 0)",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "HOME": str(tmp_path),
            "PATH": "/usr/bin:/bin",
            "HERMES_HOME": str(hermes_home),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    _assert_lease_unchanged(marker, payload, identity)


def test_partial_publication_is_a_blocking_claim_and_only_one_owner_succeeds(tmp_path):
    """Pause the first writer after create-new but before its first byte.

    Every other language observes this same empty-path publication window.
    Ordinary acquisition must preserve it and lose, never classify it stale,
    unlink it, and become a second owner.
    """
    marker = tmp_path / ".hermes-update-in-progress"
    ready = tmp_path / "created"
    publish = tmp_path / "publish"
    release = tmp_path / "release"
    result = tmp_path / "first-result"
    program = """
import pathlib, sys, time
from hermes_cli import update_lock

marker, ready, publish, release, result = map(pathlib.Path, sys.argv[1:])
real_write = update_lock.os.write
paused = False

def barrier_write(fd, payload):
    global paused
    if not paused:
        paused = True
        ready.write_text("ready", encoding="utf-8")
        while not publish.exists():
            time.sleep(0.01)
    return real_write(fd, payload)

update_lock.os.write = barrier_write
lock = update_lock.UpdateLock(path=marker)
acquired = lock.acquire()
result.write_text("acquired" if acquired else "refused", encoding="utf-8")
while acquired and not release.exists():
    time.sleep(0.01)
lock.release()
"""
    first = subprocess.Popen(
        [
            sys.executable,
            "-c",
            program,
            str(marker),
            str(ready),
            str(publish),
            str(release),
            str(result),
        ],
        cwd=REPOSITORY,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.time() + 10
        while not ready.exists() and time.time() < deadline:
            time.sleep(0.01)
        assert ready.exists(), first.stderr.read().decode("utf-8", errors="replace")
        empty_identity = (marker.stat().st_dev, marker.stat().st_ino)
        assert marker.read_bytes() == b""

        contender = subprocess.run(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; from hermes_cli.update_lock import UpdateLock; "
                "raise SystemExit(9 if UpdateLock(path=Path(__import__('sys').argv[1])).acquire() else 0)",
                str(marker),
            ],
            cwd=REPOSITORY,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        assert contender.returncode == 0, contender.stderr.decode("utf-8", errors="replace")
        assert (marker.stat().st_dev, marker.stat().st_ino) == empty_identity

        publish.write_text("go", encoding="utf-8")
        deadline = time.time() + 10
        while not result.exists() and time.time() < deadline:
            time.sleep(0.01)
        assert result.read_text(encoding="utf-8") == "acquired"
        assert (marker.stat().st_dev, marker.stat().st_ino) == empty_identity
        assert len(marker.read_text(encoding="utf-8").splitlines()) == 3
    finally:
        release.write_text("go", encoding="utf-8")
        stdout, stderr = first.communicate(timeout=10)
        assert first.returncode == 0, (stdout + stderr).decode("utf-8", errors="replace")


def test_posix_handoff_loses_to_elfos_lease_without_touching_install(tmp_path):
    install_root = tmp_path / "hermes-agent"
    site_packages = install_root / "venv/lib/python3.13/site-packages"
    site_packages.mkdir(parents=True)
    (install_root / "sentinel").write_text("checkout\n", encoding="utf-8")
    (site_packages / "sentinel.py").write_text("runtime = 1\n", encoding="utf-8")
    marker, payload, identity = _elfos_lease(tmp_path)
    before = _tree_digest(install_root)

    result = subprocess.run(
        [
            "/bin/bash",
            str(POSIX_HANDOFF),
            "--daemonized",
            "--self-test-marker",
            "--install-root",
            str(install_root),
            "--desktop-pid",
            "0",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "HOME": str(tmp_path),
            "PATH": "/usr/bin:/bin",
            "TMPDIR": str(tmp_path),
        },
        timeout=20,
        check=False,
    )

    assert result.returncode == 2
    _assert_lease_unchanged(marker, payload, identity)
    assert _tree_digest(install_root) == before


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is unavailable")
def test_electron_writer_loses_to_elfos_lease_without_overwrite(tmp_path):
    marker, payload, identity = _elfos_lease(tmp_path)
    program = """
import { writeUpdateMarker } from %s;
const acquired = writeUpdateMarker(process.argv[1], process.pid);
process.exit(acquired ? 9 : 0);
""" % repr(ELECTRON_MARKER.as_uri())
    result = subprocess.run(
        ["node", "--experimental-strip-types", "--input-type=module", "-e", program, str(tmp_path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    _assert_lease_unchanged(marker, payload, identity)


def test_rust_and_windows_sources_use_create_new_and_identity_cleanup():
    rust = RUST_UPDATER.read_text(encoding="utf-8")
    powershell = WINDOWS_HANDOFF.read_text(encoding="utf-8")
    assert ".create_new(true)" in rust
    assert "owned_payload" in rust
    assert "owned_identity" in rust
    assert "marker_identity" in rust
    assert "update marker publication could not be verified" in rust
    assert "cannot remove exact stale marker" not in rust
    assert "GetLastError() != 87" in rust
    assert "if ok == 0 {\n            return true;" in rust
    assert "!= Some(libc::ESRCH)" in rust
    assert "std::fs::write(&path" not in rust
    assert "[System.IO.FileMode]::CreateNew" in powershell
    assert "$script:LeasePayload" in powershell
    assert "$script:LeaseStream" in powershell
    assert "[System.IO.FileOptions]::DeleteOnClose" in powershell
    assert "[System.IO.FileAccess]::ReadWrite" in powershell
    assert "WriteAllText($MarkerPath" not in powershell
