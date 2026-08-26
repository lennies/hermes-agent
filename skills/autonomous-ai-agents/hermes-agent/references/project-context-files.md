# Project Context Files

Hermes injects project-level instructions into the system prompt by reading context files from the working directory. The discovery order is **first match wins** — only one project context source is loaded per session. This is separate from the optional top-level `global_instructions_file`, which is read from the default Hermes root config and rendered first under `# Trusted Host Policy`.

| File (in priority order) | Discovery | Use when |
|---|---|---|
| `.hermes.md` / `HERMES.md` | Walks parents up to the git root, stops at git root | You want hierarchical project rules (root + per-package overrides) |
| `AGENTS.override.md` / `AGENTS.md` / `agents.md` | Git root through cwd; first filename wins per directory | You want portable agent instructions that work the same in Hermes, Claude Code, Codex, etc. |
| `CLAUDE.md` / `claude.md` | Cwd only | Same as AGENTS.md, Claude-flavored |
| `.cursorrules` / `.cursor/rules/*.mdc` | Cwd only | Migrating from Cursor |

`SOUL.md` (in the active profile's `$HERMES_HOME`) is independent — it sets the agent's identity, not project rules. Global instructions are also independent: every named profile reads the same configured source from the default root, and a profile-local attempt to set the key is rejected.

### Pick the right one

- **Use `.hermes.md`** when you want Hermes-specific behavior that lives above the cwd (root + subtree), or when you want rules to inherit from a parent directory. The parent walk stops at the git root, so a home-level `.hermes.md` won't leak into every project (a git repo's root is the boundary).
- **Use `AGENTS.md`** when the same project will also be worked on by other agents (Codex, Claude Code, OpenCode). Those tools have compatible conventions for portable project instructions; Hermes loads the applicable root-to-cwd chain.
- **Don't put project rules in `~/.hermes/AGENTS.md`** and expect implicit global behavior. Use `global_instructions_file` in the default root config for host-wide policy, `SOUL.md` for profile identity, or a skill for reusable procedures.

### Size and truncation

Project context uses the configured/dynamic character cap and head-tail truncation. The global instructions source instead has a strict 500,000-byte implementation ceiling and is never silently truncated: missing, invalid, empty, non-regular, unreadable, changing-during-read, or oversized sources fail prompt construction.

If the global source is also discovered as project context, it is deduplicated
first by resolved path and then by exact byte digest. A duplicate
`AGENTS.override.md` still shadows `AGENTS.md`; only exclusion of the configured
global source falls through to the next filename. These exclusions also apply
to progressively discovered subdirectory context and are recovered only from
the hash-verified byte-zero snapshot frame on restore or fork.

### Security

All context files pass through the threat-pattern scanner before reaching the system prompt. Patterns matching prompt injection or promptware are replaced with a `[BLOCKED: ...]` placeholder. This means an `AGENTS.md` containing obvious injection attempts won't reach the model — the scanner blocks the content, not the file, so the rest of the file still loads.

### Disable for one session

Project-context suppression skips auto-injection of cwd context files. The configured trusted-host block is outside that gate. It is also kept when a local Hermes runtime uses a remote terminal backend; the backend-specific environment warning still describes where tools execute.

Global instructions are frozen into the session prompt. Fresh builds and
explicit rebuilds see current bytes; ordinary restore reuses the stored prompt
verbatim rather than hot-reloading and breaking the prompt cache. A bounded,
byte-zero snapshot frame records the frozen source path, original-byte digest,
and exact visible trusted-policy block; restore and progressive-context
deduplication accept only that hash-verified frame and never scan rendered
prompt prose for provenance-like text. When no policy is configured, a
code-owned absent envelope still reserves byte zero. Portable session imports
discard cached system prompts and rebuild from the destination host.

### Example: a small `.hermes.md`

```markdown
# My Project

Hermes: when working in this repo, follow these rules.

## Build
- Always run `make test` before declaring a change done.
- Use `uv run` for Python, not `pip install`.

## Style
- Prefer `pathlib.Path` over `os.path`.
- No `print()` in production code — use the `logger`.
```

That file at `/home/me/projects/myrepo/.hermes.md` is auto-loaded when Hermes runs in any subdirectory of `/home/me/projects/myrepo`, but not when it runs in `/home/me/other-project`.
