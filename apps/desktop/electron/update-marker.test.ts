/**
 * Tests for electron/update-marker.ts — the in-app update mutual-exclusion
 * marker that prevents a desktop relaunched mid-update from spawning a backend
 * the updater then kills in a loop (#50238).
 *
 * Run with: node --test electron/update-marker.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * Why this matters: the gate must (a) report every live updater regardless of
 * marker age, (b) fail closed on malformed/partial/dead-pid markers, and (c)
 * never delete an unowned marker during ordinary acquisition.
 */

import fs from 'fs'
import assert from 'node:assert/strict'
import os from 'os'
import path from 'path'

import { test } from 'vitest'

import {
  canonicalUpdateRoot,
  isPidAlive,
  markerPath,
  readLiveUpdateMarker,
  releaseWrittenUpdateMarker,
  UPDATE_MARKER_MAX_AGE_MS,
  updateHandoffConflict,
  writeUpdateMarker
} from './update-marker'

function tmpHome(tag) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `hermes-marker-${tag}-`))

  return dir
}

function writeMarker(home, pid, startedAtSec) {
  fs.writeFileSync(markerPath(home), `${pid}\n${startedAtSec}`)
}

const ALIVE: typeof process.kill = () => true // injected kill that "succeeds" => pid alive

const DEAD: typeof process.kill = () => {
  const err = new Error('no such process')

  ;(err as any).code = 'ESRCH'
  throw err
}

test('absent marker => no live update', () => {
  const home = tmpHome('absent')
  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE }), null)
})

test('live pid within age ceiling => live update reported', () => {
  const home = tmpHome('live')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor(now / 1000) - 5) // 5s old
  const res = readLiveUpdateMarker(home, { kill: ALIVE, now: () => now })
  assert.ok(res, 'a fresh, alive marker is a live update')
  assert.equal(res.pid, 4242)
  assert.ok(res.ageMs >= 0 && res.ageMs < 10_000)
  assert.ok(fs.existsSync(markerPath(home)), 'a live marker is NOT deleted')
})

test('dead pid => unverifiable owner and marker is preserved', () => {
  const home = tmpHome('dead')
  writeMarker(home, 999999, Math.floor(Date.now() / 1000))
  const owner = readLiveUpdateMarker(home, { kill: DEAD })
  assert.equal(owner?.unverifiable, true)
  assert.ok(fs.existsSync(markerPath(home)), 'ordinary readers never delete an unowned marker')
})

test('live marker past the UI age ceiling remains authoritative', () => {
  const home = tmpHome('expired')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor((now - UPDATE_MARKER_MAX_AGE_MS - 60_000) / 1000))
  const owner = readLiveUpdateMarker(home, { kill: ALIVE, now: () => now })
  assert.equal(owner?.pid, 4242, 'a live updater is never stolen based on age alone')
  assert.ok(fs.existsSync(markerPath(home)))
})

test('named profile resolves to the install-global update root', () => {
  const root = tmpHome('profile-root')
  const profile = path.join(root, 'profiles', 'delivery-maintainer')
  assert.equal(canonicalUpdateRoot(profile), root)
  assert.equal(markerPath(profile), path.join(root, '.hermes-update-in-progress'))
})

test('malformed marker => unverifiable owner and preserved', () => {
  const home = tmpHome('malformed')
  fs.writeFileSync(markerPath(home), 'not-a-pid\nnonsense')
  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE })?.unverifiable, true)
  assert.ok(fs.existsSync(markerPath(home)))
})

test('isPidAlive: own pid is alive, impossible pid is dead', () => {
  assert.equal(isPidAlive(process.pid), true)
  assert.equal(isPidAlive(-1), false)
  assert.equal(isPidAlive(0), false)
  assert.equal(isPidAlive(NaN), false)
})

test('isPidAlive: EPERM counts as alive (process owned by another user)', () => {
  const eperm = () => {
    const err = new Error('operation not permitted')

    ;(err as any).code = 'EPERM'
    throw err
  }

  assert.equal(isPidAlive(4242, eperm), true)
})

test('isPidAlive: unknown probe failures preserve the lease', () => {
  const unknown = () => {
    throw new Error('probe unavailable')
  }

  assert.equal(isPidAlive(4242, unknown), true)
})

test('writeUpdateMarker writes a marker that readLiveUpdateMarker accepts', () => {
  const home = tmpHome('write')
  const now = 1_000_000_000_000
  writeUpdateMarker(home, 4242, { now: () => now })
  // The marker should be readable and report the same pid.
  const res = readLiveUpdateMarker(home, { kill: ALIVE, now: () => now })
  assert.ok(res, 'marker written by writeUpdateMarker should be detected as live')
  assert.equal(res.pid, 4242)
  assert.ok(fs.existsSync(markerPath(home)), 'marker file should exist after write')
})

test('writeUpdateMarker never overwrites a live holder', () => {
  const home = tmpHome('write-handoff-age')
  const now = 1_000_000_000_000
  const startedAt = Math.floor(now / 1000) - 300

  writeMarker(home, 1010, startedAt)
  assert.equal(writeUpdateMarker(home, 2020, { kill: ALIVE, now: () => now }), false)

  const [pidLine, startedLine] = fs.readFileSync(markerPath(home), 'utf8').split('\n')
  assert.equal(Number.parseInt(pidLine, 10), 1010, 'the original owner is preserved')
  assert.equal(Number.parseInt(startedLine, 10), startedAt)
})

test('writeUpdateMarker uses the acquisition time passed to a detached script', () => {
  const home = tmpHome('write-script-acquired-at')
  const now = 1_000_000_000_000
  const startedAt = Math.floor(now / 1000) - 300

  writeUpdateMarker(home, 2020, { now: () => now, startedAt })

  const [, startedLine] = fs.readFileSync(markerPath(home), 'utf8').split('\n')
  assert.equal(Number.parseInt(startedLine, 10), startedAt)
})

test('releaseWrittenUpdateMarker removes only the exact claim written in this process', () => {
  const home = tmpHome('release-owned')
  assert.equal(writeUpdateMarker(home, 4242), true)
  assert.equal(releaseWrittenUpdateMarker(home, 9999), false)
  assert.ok(fs.existsSync(markerPath(home)))
  assert.equal(releaseWrittenUpdateMarker(home, 4242), true)
  assert.ok(!fs.existsSync(markerPath(home)))
})

test('releaseWrittenUpdateMarker preserves a claim changed by a handoff partner', () => {
  const home = tmpHome('release-replaced')
  assert.equal(writeUpdateMarker(home, 4242), true)
  const replacement = '7777\n1\nreplacement\n'
  fs.writeFileSync(markerPath(home), replacement)
  assert.equal(releaseWrittenUpdateMarker(home, 4242), false)
  assert.equal(fs.readFileSync(markerPath(home), 'utf8'), replacement)
})

test('writeUpdateMarker fails closed on bad path', () => {
  const badHome = path.join(os.tmpdir(), 'hermes-marker-nonexistent-' + Date.now())
  fs.writeFileSync(badHome, 'not a directory')
  assert.equal(writeUpdateMarker(badHome, 4242), false)
})

test('writeUpdateMarker + dead pid => remains a blocking operator-recovery lease', () => {
  const home = tmpHome('write-dead')
  writeUpdateMarker(home, 999999, { now: () => Date.now() })
  // PID 999999 is almost certainly not alive.
  const res = readLiveUpdateMarker(home, { kill: DEAD })
  assert.equal(res?.unverifiable, true)
  assert.ok(fs.existsSync(markerPath(home)), 'marker file is preserved')
})

// ---------------------------------------------------------------------------
// updateHandoffConflict (#75778)
//
// A retried "Update" click must not spawn a second updater over a still-live
// one — writeUpdateMarker unconditionally overwrites the marker, so an
// unchecked hand-off clobbers the original updater's claim while it is still
// alive and mutating the checkout.
// ---------------------------------------------------------------------------

test('no marker => hand-off is not blocked', () => {
  const home = tmpHome('conflict-none')
  assert.equal(updateHandoffConflict(home, { kill: ALIVE }), null)
})

test('a different live updater already owns the marker => hand-off is blocked', () => {
  const home = tmpHome('conflict-live')
  const now = 1_000_000_000_000
  writeMarker(home, 1010, Math.floor(now / 1000) - 6) // 6s old
  const conflict = updateHandoffConflict(home, { kill: ALIVE, now: () => now })
  assert.ok(conflict, 'a live foreign updater must block a new hand-off')
  assert.equal(conflict.pid, 1010)
  assert.match(conflict.message, /already running/)
  assert.match(conflict.message, /PID 1010/)
  assert.match(conflict.message, /6s/)
})

test('a dead-pid marker blocks a hand-off until operator recovery', () => {
  const home = tmpHome('conflict-dead')
  writeMarker(home, 999999, Math.floor(Date.now() / 1000))
  const conflict = updateHandoffConflict(home, { kill: DEAD })
  assert.ok(conflict)
  assert.match(conflict.message, /cannot be verified/)
})

test('a live marker past the UI age ceiling still blocks a hand-off', () => {
  const home = tmpHome('conflict-expired')
  const now = 1_000_000_000_000
  writeMarker(home, 1010, Math.floor((now - UPDATE_MARKER_MAX_AGE_MS - 60_000) / 1000))
  assert.ok(updateHandoffConflict(home, { kill: ALIVE, now: () => now }))
})

test('minutes-scale elapsed time is formatted as "Nm Ss"', () => {
  const home = tmpHome('conflict-minutes')
  const now = 1_000_000_000_000
  writeMarker(home, 1010, Math.floor(now / 1000) - 125) // 2m 5s old
  const conflict = updateHandoffConflict(home, { kill: ALIVE, now: () => now })
  assert.ok(conflict)
  assert.match(conflict.message, /2m 5s/)
})
