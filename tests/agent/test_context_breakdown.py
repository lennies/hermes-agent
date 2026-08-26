"""Tests for live session context breakdown."""

from unittest.mock import MagicMock, patch

from agent.context_breakdown import compute_session_context_breakdown
from agent.system_prompt import (
    extract_prompt_layout_snapshot,
    format_prompt_layout_snapshot,
)


def _framed_prompt(parts: dict[str, str]) -> str:
    body = "\n\n".join(parts[name] for name in ("stable", "context", "volatile") if parts[name])
    return f"{body}\n\n{format_prompt_layout_snapshot(parts)}"


def _make_agent(
    *,
    stable: str = "identity and guidance",
    context: str = "",
    volatile: str = "timestamp line",
    tools: list | None = None,
    context_length: int = 200_000,
    last_prompt_tokens: int = 0,
):
    agent = MagicMock()
    agent.model = "openai/gpt-5.4"
    agent.tools = tools or [
        {"type": "function", "function": {"name": "terminal", "description": "run"}},
        {"type": "function", "function": {"name": "mcp_demo_tool", "description": "mcp"}},
        {"type": "function", "function": {"name": "delegate_task", "description": "spawn"}},
    ]
    agent._memory_store = None
    agent._memory_enabled = True
    agent._user_profile_enabled = True
    agent.context_compressor = MagicMock(
        context_length=context_length,
        last_prompt_tokens=last_prompt_tokens,
    )
    return agent, {"stable": stable, "context": context, "volatile": volatile}


def test_breakdown_includes_major_categories():
    stable = "base guidance"
    context = "# Project Context\nFollow AGENTS.md"
    volatile_skills = "<available_skills>\n  demo:\n    - hello: hi\n</available_skills>"
    volatile_other = "Conversation started: Monday, August 25, 2026"
    volatile = f"{volatile_skills}\n\n{volatile_other}"
    history = [{"role": "user", "content": "hello there"}]
    agent, parts = _make_agent(stable=stable, context=context, volatile=volatile)
    parts.update(
        volatile_skills=volatile_skills,
        volatile_memory="",
        volatile_other=volatile_other,
    )

    with patch("agent.system_prompt.build_system_prompt_parts", return_value=parts):
        data = compute_session_context_breakdown(agent, history)

    ids = {item["id"] for item in data["categories"]}
    assert {"system_prompt", "tool_definitions", "rules", "skills", "mcp", "subagent_definitions", "conversation"} <= ids
    assert data["context_max"] == 200_000
    assert data["estimated_total"] > 0


def test_breakdown_uses_active_cached_prompt_without_live_rebuild():
    agent, _parts = _make_agent()
    parts = {
        "stable": "FROZEN STABLE V1",
        "context": "# Project Context\nFROZEN RULES V1",
        "volatile_skills": "<available_skills>FROZEN SKILL V1</available_skills>",
        "volatile_memory": "<memory-context>FROZEN MEMORY V1</memory-context>",
        "volatile_other": "Conversation started: Monday, August 25, 2026",
    }
    parts["volatile"] = "\n\n".join(
        parts[name]
        for name in ("volatile_skills", "volatile_memory", "volatile_other")
    )
    agent._cached_system_prompt_static = parts["stable"]
    agent._cached_system_prompt = _framed_prompt(parts)

    with patch(
        "agent.system_prompt.build_system_prompt_parts",
        side_effect=AssertionError("must not consult live policy/context"),
    ):
        data = compute_session_context_breakdown(agent, [])

    categories = {item["id"]: item["tokens"] for item in data["categories"]}
    assert categories["system_prompt"] > 0
    assert categories["rules"] > 0
    assert categories["skills"] > 0
    assert categories["memory"] > 0


def test_restored_breakdown_uses_frozen_tail_without_static_metadata():
    agent, _parts = _make_agent()
    agent._cached_system_prompt_static = None
    agent._cached_system_prompt_context = None
    agent._cached_system_prompt_volatile = None
    parts = {
        "stable": "FROZEN STABLE",
        "context": (
            "# Project Context\nFROZEN RULES\n"
            "<available_skills>PROJECT LOOKALIKE</available_skills>\n"
            "<memory-context>PROJECT LOOKALIKE</memory-context>\n"
            "Conversation started: PROJECT LOOKALIKE"
        ),
        "volatile_skills": "<available_skills>FROZEN SKILL</available_skills>",
        "volatile_memory": "<memory-context>FROZEN MEMORY</memory-context>",
        "volatile_other": "Conversation started: Monday, August 25, 2026",
    }
    parts["volatile"] = "\n\n".join(
        parts[name]
        for name in ("volatile_skills", "volatile_memory", "volatile_other")
    )
    agent._cached_system_prompt = _framed_prompt(parts)
    agent._memory_store = MagicMock()
    agent._memory_store.format_for_system_prompt.return_value = "CURRENT MEMORY"

    with patch(
        "agent.system_prompt.build_system_prompt_parts",
        side_effect=AssertionError("must not consult live policy/context"),
    ):
        data = compute_session_context_breakdown(agent, [])

    categories = {item["id"]: item["tokens"] for item in data["categories"]}
    assert categories["rules"] == (len(parts["context"]) + 3) // 4
    assert categories["skills"] == (len("<available_skills>FROZEN SKILL</available_skills>") + 3) // 4
    assert categories["memory"] == (len("<memory-context>FROZEN MEMORY</memory-context>") + 3) // 4
    agent._memory_store.format_for_system_prompt.assert_not_called()


def test_legacy_cached_prompt_is_reported_as_unclassified():
    agent, _parts = _make_agent()
    agent._cached_system_prompt = "LEGACY PROMPT WITHOUT LAYOUT"

    data = compute_session_context_breakdown(agent, [])

    categories = {item["id"]: item["tokens"] for item in data["categories"]}
    assert categories["unclassified_prompt"] > 0
    assert "rules" not in categories


def test_prompt_layout_round_trips_exact_frozen_segments():
    parts = {
        "stable": "STABLE 🧝",
        "context": "# Project Context\nRULES",
        "volatile_skills": "<available_skills>SKILLS</available_skills>",
        "volatile_memory": "<memory-context>MEMORY</memory-context>",
        "volatile_other": "Conversation started: NOW",
    }
    parts["volatile"] = "\n\n".join(
        parts[name]
        for name in ("volatile_skills", "volatile_memory", "volatile_other")
    )

    assert extract_prompt_layout_snapshot(_framed_prompt(parts)) == parts


def test_prompt_layout_tampering_fails_closed_to_unclassified():
    parts = {
        "stable": "FROZEN STABLE",
        "context": "FROZEN CONTEXT",
        "volatile": "FROZEN VOLATILE",
        "volatile_skills": "",
        "volatile_memory": "",
        "volatile_other": "FROZEN VOLATILE",
    }
    agent, _ = _make_agent()
    agent._cached_system_prompt = _framed_prompt(parts).replace(
        "FROZEN CONTEXT", "FORGED CONTEXT", 1
    )

    data = compute_session_context_breakdown(agent, [])

    categories = {item["id"]: item["tokens"] for item in data["categories"]}
    assert categories["unclassified_prompt"] > 0
    assert "rules" not in categories



# ── /context renderers (pure functions over the payload) ────────────────────

from agent.context_breakdown import (  # noqa: E402
    compute_context_details,
    render_context_breakdown_lines,
    render_context_category_lines,
    render_context_details_lines,
    render_context_grid,
)


def _payload(**overrides):
    base = {
        "categories": [
            {"id": "system_prompt", "label": "System prompt", "tokens": 10_000},
            {"id": "tool_definitions", "label": "Tool definitions", "tokens": 20_000},
            {"id": "skills", "label": "Skills", "tokens": 5_000},
            {"id": "conversation", "label": "Conversation", "tokens": 15_000},
        ],
        "context_max": 200_000,
        "context_percent": 25,
        "context_used": 50_000,
        "estimated_total": 50_000,
        "model": "openai/gpt-test",
    }
    base.update(overrides)
    return base


def test_grid_is_5x20_and_mostly_free():
    rows = render_context_grid(_payload())
    assert len(rows) == 5
    cells = " ".join(rows).split(" ")
    assert len(cells) == 100
    # 50k / 200k → 25 used cells, 75 free
    assert cells.count("·") == 75
    # Category glyphs proportional: 10k→5, 20k→10, 5k→2-3, 15k→7-8 cells
    assert cells.count("■") == 5
    assert cells.count("▣") == 10










def test_breakdown_lines_grid_toggle():
    with_grid = render_context_breakdown_lines(_payload(), grid=True)
    without = render_context_breakdown_lines(_payload(), grid=False)
    assert any("·" in line for line in with_grid[:5])
    assert not any("·" in line for line in without[:2])
    # Both include the window summary and the expand hint
    for lines in (with_grid, without):
        text = "\n".join(lines)
        assert "Context window: 50,000 / 200,000 tokens (25%)" in text
        assert "/context all" in text




def test_details_lines_caps_listing():
    details = {
        "skills": [
            {"name": f"skill-{i}", "index_tokens": 10, "skill_md_tokens": 100}
            for i in range(20)
        ],
        "toolsets": [],
    }
    lines = render_context_details_lines(details)
    assert any("… and 5 more" in line for line in lines)
