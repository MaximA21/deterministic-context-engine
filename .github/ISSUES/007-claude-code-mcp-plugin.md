---
title: Claude Code MCP plugin
labels: integration
---

## Summary

Build an MCP (Model Context Protocol) server that exposes the context engine as a tool for Claude Code, allowing Claude to use deterministic context management during coding sessions.

## Motivation

Claude Code sessions frequently exceed context limits on large codebases. An MCP plugin would let Claude offload context management to the engine, keeping critical code references and bug reports in context while evicting stale filler.

## Proposed Approach

1. Create `mcp_server.py` implementing the MCP server protocol
2. Expose tools:
   - `context_append(role, content)` — Add a chunk to the context log
   - `context_get()` — Return current active context within budget
   - `context_search(query)` — Search stored chunks by relevance
   - `context_stats()` — Return context usage, compaction history
3. Use `ChunkLog` as the backing store with goal-guided scoring
4. Package as an installable MCP server for Claude Code's `mcp_servers` config

## Configuration

```json
{
  "mcp_servers": {
    "context-engine": {
      "command": "python",
      "args": ["path/to/mcp_server.py"],
      "env": {}
    }
  }
}
```

## Acceptance Criteria

- [ ] MCP server implements the tool protocol
- [ ] All four tools work end-to-end with Claude Code
- [ ] Context persists across tool calls within a session
- [ ] Installation docs in README
- [ ] Example usage in a coding session
