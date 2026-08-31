/**
 * In-app update mutual-exclusion marker (#50238).
 *
 * The Tauri updater writes HERMES_HOME/.hermes-update-in-progress for the whole
 * duration of an `--update` run (see apps/bootstrap-installer/src-tauri/src/
 * update.rs `UpdateMarkerGuard`). The marker body is pid, unix start time,
 * and an ownership token; deployed two-line markers remain readable.
 *
 * Why: if the user relaunches the desktop mid-update — the window vanished with
 * no progress and looks crashed — a fresh instance must NOT spawn its own local
 * backend. That backend re-locks the venv shim, the updater's straggler cleanup
 * (`force_kill_other_hermes`, taskkill /IM hermes.exe) kills it, the launch
 * fails with the 45s "backend didn't come up" timeout, and the user relaunches
 * into the same trap — an infinite respawn/kill loop. The desktop gates local
 * backend startup on this marker and parks until the update finishes.
 *
 * This module holds the PURE, side-effect-light logic (path, pid liveness,
 * parse + staleness) so it is unit-testable without booting Electron. The
 * polling/boot-progress wrapper lives in main.ts where the boot-progress and
 * log sinks are.
 */

import fs from 'fs'
import crypto from 'node:crypto'
import path from 'path'

const ownedMarkers = new Map()

// Retained as an API-compatible display/test boundary. Lease ownership is
// liveness-based: a live writer is never stolen merely because it ran long.
export const UPDATE_MARKER_MAX_AGE_MS = 20 * 60 * 1000

export function canonicalUpdateRoot(hermesHome) {
  const resolved = path.resolve(String(hermesHome || ''))

  return path.basename(path.dirname(resolved)) === 'profiles' ? path.dirname(path.dirname(resolved)) : resolved
}

export function markerPath(hermesHome) {
  return path.join(canonicalUpdateRoot(hermesHome), '.hermes-update-in-progress')
}

function readMarkerSnapshot(file) {
  let fd

  try {
    fd = fs.openSync(file, 'r')
    const stat = fs.fstatSync(fd)

    if (!stat.isFile()) {
      return null
    }

    return { raw: String(fs.readFileSync(fd, 'utf8')), dev: stat.dev, ino: stat.ino }
  } catch {
    return null
  } finally {
    if (typeof fd === 'number') {
      fs.closeSync(fd)
    }
  }
}

// True only if a host process with this pid is currently alive. Signal 0 does
// not deliver a signal — it just probes existence/permission. ESRCH => dead;
// EPERM => alive but owned by another user (still "alive" for our purposes).
// Injectable `kill` keeps it unit-testable.
export function isPidAlive(pid, kill: typeof process.kill = process.kill.bind(process)) {
  if (!Number.isInteger(pid) || pid <= 0) {
    return false
  }

  try {
    kill(pid, 0)

    return true
  } catch (err) {
    // Only ESRCH proves the process is gone. Permission errors and unknown
    // probe failures must preserve the lease and fail closed.
    return !(err && err.code === 'ESRCH')
  }
}

/**
 * Read + interpret the marker.
 *
 * Returns `{ pid, ageMs }` when an update is GENUINELY still running and an
 * `{ unverifiable: true }` blocking result for unreadable, malformed, partial,
 * or dead-owner markers. Returns `null` only when the pathname is absent.
 * Ordinary readers never perform stale cleanup: check-then-unlink can delete a
 * replacement claim created between those operations.
 *
 * Pure-ish: file I/O against the given path, plus an injectable pid probe and
 * clock for tests.
 */
export function readLiveUpdateMarker(
  hermesHome,
  {
    kill,
    now = Date.now,
    maxAgeMs = UPDATE_MARKER_MAX_AGE_MS
  }: {
    now?: () => number
    maxAgeMs?: number
    kill?: typeof process.kill
  } = {}
) {
  void maxAgeMs
  const file = markerPath(hermesHome)
  const snapshot = readMarkerSnapshot(file)

  if (!snapshot) {
    return fs.existsSync(file) ? { pid: null, ageMs: null, token: null, unverifiable: true } : null
  }

  const [pidLine, startedLine, tokenLine] = snapshot.raw.split('\n')
  const pid = Number.parseInt((pidLine || '').trim(), 10)
  const startedAt = Number.parseInt((startedLine || '').trim(), 10)
  const wellFormed = Number.isInteger(pid) && pid > 0 && Number.isFinite(startedAt) && startedAt > 0
  const ageMs = wellFormed ? now() - startedAt * 1000 : null
  const alive = wellFormed && isPidAlive(pid, kill)

  if (!alive) {
    return { pid: null, ageMs: null, token: null, unverifiable: true }
  }

  return { pid, ageMs, token: (tokenLine || '').trim() || null, unverifiable: false }
}

/**
 * Write the update-in-progress marker *from the desktop* before handing off
 * to the detached updater.
 *
 * The Tauri-based hermes-setup.exe takes several seconds to initialise its
 * window and reach the Rust `run_update` entry point where it writes the
 * marker itself. During that gap the desktop's `app.quit()` teardown kills
 * the backend child, the renderer's WebSocket drops, and the renderer
 * immediately calls `ensureBackend()` → `waitForUpdateToFinish()`. Because
 * the updater hasn't written the marker yet, the gate sees no live update
 * and spawns a *new* backend — which re-locks `.pyd` files in the venv.
 * When the updater finally reaches the venv-rebuild stage it finds those
 * files locked and the update bricks.
 *
 * Fix: the desktop writes the marker itself, using the spawned updater's
 * PID, immediately after `spawn()`. The updater's `UpdateMarkerGuard` will
 * later adopt the exact PID/token claim. Create-new means no writer can
 * replace a live owner; callers must cancel the spawned child when this
 * function returns false. When the updater finishes it deletes its exact
 * claim.
 * If the updater fails during the settle window, the caller releases this
 * exact in-process claim with `releaseWrittenUpdateMarker`. A marker left by
 * an abrupt process death remains fail-closed for explicit operator recovery.
 */
export function writeUpdateMarker(
  hermesHome,
  pid,
  {
    kill,
    now = Date.now,
    maxAgeMs = UPDATE_MARKER_MAX_AGE_MS,
    startedAt
  }: {
    now?: () => number
    maxAgeMs?: number
    kill?: typeof process.kill
    startedAt?: number
  } = {}
) {
  const file = markerPath(hermesHome)
  const nowMs = now()
  const owner = readLiveUpdateMarker(hermesHome, { kill, maxAgeMs, now: () => nowMs })

  if (owner) {
    return owner.pid === pid
  }

  const acquiredAt = typeof startedAt === 'number' && Number.isInteger(startedAt) ? startedAt : Math.floor(nowMs / 1000)

  const token = crypto.randomUUID().replaceAll('-', '')
  const payload = `${pid}\n${acquiredAt}\n${token}\n`
  let fd
  let ownedIdentity

  try {
    fs.mkdirSync(path.dirname(file), { recursive: true })
    fd = fs.openSync(file, 'wx', 0o600)
    const opened = fs.fstatSync(fd)

    if (!opened.isFile()) {
      return false
    }

    ownedIdentity = { dev: opened.dev, ino: opened.ino }
    fs.writeFileSync(fd, payload, 'utf8')
    fs.fsyncSync(fd)

    const published = readMarkerSnapshot(file)

    if (
      !published ||
      published.dev !== ownedIdentity.dev ||
      published.ino !== ownedIdentity.ino ||
      published.raw !== payload
    ) {
      return false
    }

    ownedMarkers.set(file, { pid, dev: published.dev, ino: published.ino, raw: published.raw })

    return true
  } catch {
    return false
  } finally {
    if (typeof fd === 'number') {
      fs.closeSync(fd)
    }
  }
}

/** Release only the exact marker this Electron process successfully wrote. */
export function releaseWrittenUpdateMarker(hermesHome, pid) {
  const file = markerPath(hermesHome)
  const owned = ownedMarkers.get(file)

  if (!owned || owned.pid !== pid) {
    return false
  }

  const current = readMarkerSnapshot(file)

  if (!current || current.dev !== owned.dev || current.ino !== owned.ino || current.raw !== owned.raw) {
    ownedMarkers.delete(file)

    return false
  }

  try {
    // Compliant contenders never remove an unowned marker, so the pathname
    // cannot acquire a replacement owner while our exact claim exists.
    fs.unlinkSync(file)
    ownedMarkers.delete(file)

    return true
  } catch {
    return false
  }
}

/**
 * Whether a NEW updater hand-off must be refused because a different,
 * already-alive updater currently owns the marker (#75778).
 *
 * This preflight gives immediate UX feedback; atomic create-new in
 * `writeUpdateMarker` remains the authoritative post-spawn race gate.
 *
 * Returns the foreign or unverifiable owner (with a ready-to-show message)
 * when the hand-off must be refused, or `null` only when no marker exists.
 */
export function updateHandoffConflict(
  hermesHome,
  opts: {
    now?: () => number
    maxAgeMs?: number
    kill?: typeof process.kill
  } = {}
) {
  const owner = readLiveUpdateMarker(hermesHome, opts)

  if (!owner) {
    return null
  }

  if (owner.unverifiable || owner.pid === null || owner.ageMs === null) {
    return {
      pid: null,
      ageMs: null,
      message:
        'The Hermes update lease exists but its owner cannot be verified. Confirm that no update is running before removing .hermes-update-in-progress.'
    }
  }

  const mins = Math.floor(owner.ageMs / 60_000)
  const secs = Math.floor((owner.ageMs % 60_000) / 1000)
  const elapsed = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`

  return {
    pid: owner.pid,
    ageMs: owner.ageMs,
    message: `An update is already running (PID ${owner.pid}, started ${elapsed} ago). Wait for it to finish, then try again.`
  }
}
