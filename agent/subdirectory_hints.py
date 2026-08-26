"""Progressive subdirectory hint discovery.

As the agent navigates into subdirectories via tool calls (read_file, terminal,
search_files, etc.), this module discovers and loads project context files
(AGENTS.md, CLAUDE.md, .cursorrules) from those directories.  Discovered hints
are appended to the tool result so the model gets relevant context at the moment
it starts working in a new area of the codebase.

This complements the startup context loading in ``prompt_builder.py`` which only
loads from the CWD.  Subdirectory hints are discovered lazily and injected into
the conversation without modifying the system prompt (preserving prompt caching).

Inspired by Block/goose's SubdirectoryHintTracker.
"""

import hashlib
import logging
import os
import shlex
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set

from agent.prompt_builder import (
    _scan_context_content,
    extract_trusted_policy_snapshot,
)

logger = logging.getLogger(__name__)

# Context files to look for in subdirectories, in priority order.
# Same filenames as prompt_builder.py but we load ALL found (not first-wins)
# since different subdirectories may use different conventions.
_HINT_FILENAMES = [
    "AGENTS.override.md",
    "AGENTS.md", "agents.md",
    "CLAUDE.md", "claude.md",
    ".cursorrules",
]

# Maximum chars per hint file to prevent context bloat
_MAX_HINT_CHARS = 8_000

# Tool argument keys that typically contain file paths
_PATH_ARG_KEYS = {"path", "file_path", "workdir"}

# Tools that take shell commands where we should extract paths
_COMMAND_TOOLS = {"terminal"}

# How many parent directories to walk up when looking for hints.
# Prevents scanning all the way to / for deeply nested paths.
_MAX_ANCESTOR_WALK = 5

# Directory names that never contain authoritative project context.
# Backups, vendored deps, VCS internals, and caches routinely hold *copies* of
# AGENTS.md; loading those duplicates real context and inflates the prompt.
_EXCLUDED_DIR_NAMES = frozenset({
    "node_modules", "venv", ".venv", "__pycache__",
    ".git", ".hg", ".svn",
    ".Trash", ".cache", ".tox", ".mypy_cache", ".pytest_cache",
    "site-packages", "dist-packages",
    "backups", "backup", ".backups",
    "vendor", "third_party",
})

def _is_ancestor_or_same(a: Path, b: Path) -> bool:
    """Check if *a* is the same as or an ancestor of *b* (parent directory check)."""
    try:
        b.relative_to(a)
        return True
    except ValueError:
        return False


class SubdirectoryHintTracker:
    """Track which directories the agent visits and load hints on first access.

    Usage::

        tracker = SubdirectoryHintTracker(working_dir="/path/to/project")

        # After each tool call:
        hints = tracker.check_tool_call("read_file", {"path": "backend/src/main.py"})
        if hints:
            tool_result += hints  # append to the tool result string
    """

    def __init__(
        self,
        working_dir: Optional[str] = None,
        *,
        prompt_provider: Optional[Callable[[], Optional[str]]] = None,
        enabled: bool = True,
    ):
        self.enabled = bool(enabled)
        self.working_dir = Path(working_dir or os.getcwd()).resolve()
        self._loaded_dirs: Set[Path] = set()
        self._loaded_resolved_paths: Set[Path] = set()
        self._startup_resolved_paths: Set[Path] = set()
        self._excluded_resolved_paths: Set[Path] = set()
        # Content digests already injected — prevents re-sending the same file
        # reachable through symlinks, hardlinks, or duplicated copies.
        self._loaded_digests: Set[str] = set()
        self._startup_digests: Set[str] = set()
        self._excluded_digests: Set[str] = set()
        self._prompt_provider = prompt_provider
        self._prompt_exclusions_seeded = False
        # Pre-mark the working dir as loaded (startup context handles it)
        self._loaded_dirs.add(self.working_dir)
        if self.enabled:
            self._seed_working_dir_digest()

    def _seed_working_dir_digest(self) -> None:
        """Record the CWD context file's digest so it is never re-injected.

        ``prompt_builder`` already loads the working directory's context file at
        startup.  Seeding its digest here means the same content reached through
        a different path (a symlink farm, a shared workspace) is recognised as a
        duplicate instead of being sent a second time.
        """
        if not self.enabled:
            return
        self._startup_resolved_paths.clear()
        self._startup_digests.clear()
        for filename in _HINT_FILENAMES:
            candidate = self.working_dir / filename
            try:
                if not candidate.is_file():
                    continue
                resolved = candidate.resolve(strict=True)
                raw = resolved.read_bytes()
                content = raw.decode("utf-8").strip()
            except (OSError, UnicodeDecodeError):
                continue
            if not content:
                continue
            digest = hashlib.sha256(raw).hexdigest()
            if (
                resolved in self._excluded_resolved_paths
                or digest in self._excluded_digests
            ):
                # An excluded override falls through to the next candidate,
                # exactly like startup prompt assembly.
                continue
            self._startup_resolved_paths.add(resolved)
            self._startup_digests.add(digest)
            break  # first effective match wins, mirroring startup loading

    def seed_exclusions_from_prompt_snapshot(self, prompt: str) -> None:
        """Replace exclusions from one verified byte-zero prompt snapshot."""
        if not self.enabled:
            return
        snapshot = extract_trusted_policy_snapshot(prompt)
        self._excluded_resolved_paths = (
            {snapshot.resolved_path} if snapshot is not None else set()
        )
        self._excluded_digests = (
            {snapshot.source_sha256} if snapshot is not None else set()
        )
        self._prompt_exclusions_seeded = True
        self._seed_working_dir_digest()

    def _seed_exclusions_from_cached_prompt(self) -> None:
        """Recover trusted-source provenance for restored/forked snapshots.

        Restore and fork paths can reuse a cached prompt without rebuilding it,
        so this recovers only the verified provenance frozen into that snapshot;
        it never rereads current config or hot-reloads the policy.
        """
        if self._prompt_exclusions_seeded or self._prompt_provider is None:
            return
        prompt = self._prompt_provider()
        if not prompt:
            return
        self.seed_exclusions_from_prompt_snapshot(prompt)

    def check_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> Optional[str]:
        """Check tool call arguments for new directories and load any hint files.

        Returns formatted hint text to append to the tool result, or None.
        """
        if not self.enabled:
            return None
        self._seed_exclusions_from_cached_prompt()
        dirs = self._extract_directories(tool_name, tool_args)
        if not dirs:
            return None

        all_hints = []
        for d in dirs:
            hints = self._load_hints_for_directory(d)
            if hints:
                all_hints.append(hints)

        if not all_hints:
            return None

        return "\n\n" + "\n\n".join(all_hints)

    def _extract_directories(
        self, tool_name: str, args: Dict[str, Any]
    ) -> list:
        """Extract directory paths from tool call arguments."""
        candidates: Set[Path] = set()

        # Direct path arguments
        for key in _PATH_ARG_KEYS:
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                self._add_path_candidate(val, candidates)

        # Shell commands — extract path-like tokens
        if tool_name in _COMMAND_TOOLS:
            cmd = args.get("command", "")
            if isinstance(cmd, str):
                self._extract_paths_from_command(cmd, candidates)

        return list(candidates)

    def _add_path_candidate(self, raw_path: str, candidates: Set[Path]):
        """Resolve a raw path and add its directory + ancestors to candidates.

        Walks up from the resolved directory toward the filesystem root,
        stopping at the first directory already in ``_loaded_dirs`` (or after
        ``_MAX_ANCESTOR_WALK`` levels).  This ensures that reading
        ``project/src/main.py`` discovers ``project/AGENTS.md`` even when
        ``project/src/`` has no hint files of its own.
        """
        try:
            p = Path(raw_path).expanduser()
            if not p.is_absolute():
                p = self.working_dir / p
            p = p.resolve()
            # Use parent if it's a file path (has extension or doesn't exist as dir)
            if p.suffix or (p.exists() and p.is_file()):
                p = p.parent
            # Walk up ancestors — stop at already-loaded or root
            for _ in range(_MAX_ANCESTOR_WALK):
                if p in self._loaded_dirs:
                    break
                if self._is_valid_subdir(p):
                    candidates.add(p)
                parent = p.parent
                if parent == p:
                    break  # filesystem root
                p = parent
        except (OSError, ValueError, RuntimeError):
            pass

    def _extract_paths_from_command(self, cmd: str, candidates: Set[Path]):
        """Extract path-like tokens from a shell command string."""
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            tokens = cmd.split()

        for token in tokens:
            # Skip flags
            if token.startswith("-"):
                continue
            # Must look like a path (contains / or .)
            if "/" not in token and "." not in token:
                continue
            # Skip URLs
            if token.startswith(("http://", "https://", "git@")):
                continue
            self._add_path_candidate(token, candidates)

    def _is_valid_subdir(self, path: Path) -> bool:
        """Check if path is a valid directory to scan for hints.

        Only allow subdirectories within the working directory tree.
        This prevents loading AGENTS.md from outside the active workspace
        (e.g. ~/.codex/AGENTS.md, ~/.claude/CLAUDE.md), which causes
        cross-agent context contamination and instruction mixup.
        """
        try:
            if not path.is_dir():
                return False
        except OSError:
            return False
        if path in self._loaded_dirs:
            return False
        # Reject paths outside the working directory tree.
        # path.resolve() may differ from working_dir.resolve() due to symlinks,
        # but path.is_relative_to(working_dir) handles both absolute and
        # symlinked paths correctly on Python 3.9+.
        try:
            if not path.is_relative_to(self.working_dir):
                return False
        except (OSError, ValueError):
            # Older Python or path resolution error — fall back to parent
            # check as a best-effort safeguard.
            if not _is_ancestor_or_same(self.working_dir, path):
                return False
        if self._is_excluded(path):
            return False
        return True

    def _is_excluded(self, path: Path) -> bool:
        """True when the path sits inside a directory that holds copies, not context.

        Directories the user is deliberately working inside are never excluded —
        if ``working_dir`` is itself under ``vendor/``, that segment is legitimate
        and only segments *below* the working dir are screened.
        """
        try:
            rel_parts = path.relative_to(self.working_dir).parts
        except ValueError:
            # Paths outside the working dir are already rejected by
            # _is_valid_subdir before this runs; treat as excluded defensively.
            return True
        return any(part in _EXCLUDED_DIR_NAMES for part in rel_parts)

    def _load_hints_for_directory(self, directory: Path) -> Optional[str]:
        """Load hint files from a directory. Returns formatted text or None.

        Only loads hints from directories within the working directory tree.
        """
        if not self.enabled:
            return None
        self._loaded_dirs.add(directory)

        # Reject paths outside the working directory tree.
        try:
            if not directory.is_relative_to(self.working_dir):
                logger.debug(
                    "Skipping hint files in %s — outside working_dir %s",
                    directory, self.working_dir,
                )
                return None
        except (OSError, ValueError):
            if not _is_ancestor_or_same(self.working_dir, directory):
                logger.debug(
                    "Skipping hint files in %s — outside working_dir %s",
                    directory, self.working_dir,
                )
                return None

        found_hints = []
        for filename in _HINT_FILENAMES:
            hint_path = directory / filename
            try:
                if not hint_path.is_file():
                    continue
            except OSError:
                continue
            try:
                resolved = hint_path.resolve(strict=True)
                if resolved in self._excluded_resolved_paths:
                    logger.debug("Skipping excluded hint source at %s", hint_path)
                    continue
                if resolved in (
                    self._loaded_resolved_paths | self._startup_resolved_paths
                ):
                    logger.debug("Skipping duplicate hint source at %s", hint_path)
                    break
                raw = resolved.read_bytes()
                content = raw.decode("utf-8").strip()
                if not content:
                    continue
                # Skip content we've already injected. The same AGENTS.md is
                # routinely reachable through several paths (symlinked shared
                # workspaces, hardlinks, copied backups); re-sending it burns
                # context for zero new information.
                digest = hashlib.sha256(raw).hexdigest()
                if digest in self._excluded_digests:
                    logger.debug(
                        "Skipping excluded hint content at %s (digest %s)",
                        hint_path,
                        digest[:12],
                    )
                    continue
                if digest in (self._loaded_digests | self._startup_digests):
                    logger.debug(
                        "Skipping duplicate hint content at %s (digest %s)",
                        hint_path,
                        digest[:12],
                    )
                    break
                self._loaded_resolved_paths.add(resolved)
                self._loaded_digests.add(digest)
                # Same security scan as startup context loading
                content = _scan_context_content(content, filename)
                if len(content) > _MAX_HINT_CHARS:
                    content = (
                        content[:_MAX_HINT_CHARS]
                        + f"\n\n[...truncated {filename}: {len(content):,} chars total]"
                    )
                # Best-effort relative path for display
                rel_path = str(hint_path)
                try:
                    rel_path = str(hint_path.relative_to(self.working_dir))
                except (ValueError, RuntimeError):
                    try:
                        rel_path = str(hint_path.relative_to(Path.home()))
                        rel_path = "~/" + rel_path
                    except (ValueError, RuntimeError):
                        pass  # keep absolute
                found_hints.append((rel_path, content))
                # First match wins per directory (like startup loading)
                break
            except Exception as exc:
                logger.debug("Could not read %s: %s", hint_path, exc)

        if not found_hints:
            return None

        sections = []
        for rel_path, content in found_hints:
            sections.append(
                f"[Subdirectory context discovered: {rel_path}]\n{content}"
            )

        logger.debug(
            "Loaded subdirectory hints from %s: %s",
            directory,
            [h[0] for h in found_hints],
        )
        return "\n\n".join(sections)
