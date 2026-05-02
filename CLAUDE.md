# ATO-MCP Guidelines

The goal of this project is to maintain an on-device (with database build pulled from repo pre-baked) MCP server for searching the entire ATO database. 

The user is not expected to have a GPU and may be using a low performance enterprise laptop.

IMPORTANT: We should be looking to simplify the MCP surface and minimise extraneous context at all times.

# Workflow

Use `/r`, `/j`, `/b`, `/c`, and `/rj` from this repo root.

For long-running Bash commands such as builds or test suites, launch them with background execution when the runtime supports it.
Do NOT poll background tasks. Wait for completion before acting on dependent results.

# Documentation

All tagged documentation is managed by `proofd`. Canonical rule data lives outside the repo in the proofd knowledge base. `proofd sync` generates Claude Markdown snapshots under `.claude/rules/`.
Codex does not have Claude-style path-scoped rule auto-load, so repo bootstrap configures Codex hooks that inject proofd guidance on session start and targeted proofd context on relevant prompts.

Do not hand-edit `.claude/rules/*.md`. They are refreshed by `proofd sync`, typically during janitor, build, release, or finalization work. Use `"$HOME/.claude/agent-proofs/bin/proofd.py"` subcommands to create rules, add entries, split rules, record verifications, and regenerate the rule output.
Generated rule markdown is file-scoped and intentionally omits stored file lists. If you need source-reference files for a tag, use `"$HOME/.claude/agent-proofs/bin/proofd.py" entry-files --tag <TAG>`.

Tags are embedded in source code as language-appropriate comments containing `[TAG]` near the implementation site. Tags must be allocated by `proofd`; agents must not invent tag IDs themselves.

Useful commands:
- `"$HOME/.claude/agent-proofs/bin/proofd.py" sync`
- `"$HOME/.claude/agent-proofs/bin/proofd.py" lint`
- `"$HOME/.claude/agent-proofs/bin/proofd.py" entry-files --tag <TAG>`
- `"$HOME/.claude/agent-proofs/bin/proofd.py" select-matching <paths...>`
- `"$HOME/.claude/agent-proofs/bin/proofd.py" context <paths...>`