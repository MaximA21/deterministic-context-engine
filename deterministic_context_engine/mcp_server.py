"""MCP server for the Deterministic Context Engine (JSON-RPC over stdio).

Implements the Model Context Protocol with tools:
  - context/append  — append a chunk to the context log
  - context/query   — get the current context window
  - context/status  — get engine metrics (tokens, chunks, compactions)
  - context/clear   — reset the context log

Speaks JSON-RPC 2.0 over stdin/stdout (one JSON object per line).
"""

from __future__ import annotations

import json
import sys
from typing import Any

from engine import ChunkLog


SERVER_NAME = "deterministic-context-engine"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2024-11-05"


class MCPServer:
    """Minimal MCP server operating over stdio."""

    def __init__(
        self,
        max_tokens: int = 128_000,
        soft_threshold: float = 0.7,
        hard_threshold: float = 0.9,
        scoring_mode: str = "tfidf",
    ) -> None:
        goal_guided = scoring_mode == "tfidf"
        mode = scoring_mode if scoring_mode != "tfidf" else None

        self._log = ChunkLog(
            db_path=":memory:",
            max_tokens=max_tokens,
            soft_threshold=soft_threshold,
            hard_threshold=hard_threshold,
            auto_priority=False,
            goal_guided=goal_guided,
            scoring_mode=mode,
        )
        self._tools = {
            "context/append": self._tool_append,
            "context/query": self._tool_query,
            "context/status": self._tool_status,
            "context/clear": self._tool_clear,
        }

    # --- Tool handlers ---

    def _tool_append(self, params: dict[str, Any]) -> dict[str, Any]:
        role = params.get("role", "user")
        content = params.get("content", "")
        priority = params.get("priority", 1.0)
        if not content:
            return {"error": "content is required"}
        chunk_hash = self._log.append(role, content, priority=priority)
        self._log.next_turn()
        return {
            "chunk_hash": chunk_hash,
            "current_tokens": self._log.current_tokens(),
        }

    def _tool_query(self, params: dict[str, Any]) -> dict[str, Any]:
        messages = self._log.get_context()
        return {
            "messages": messages,
            "count": len(messages),
            "tokens": self._log.current_tokens(),
        }

    def _tool_status(self, params: dict[str, Any]) -> dict[str, Any]:
        chunk_count = self._log._conn.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]
        evicted = sum(
            1 for d in self._log.decisions if d.action.startswith("compact")
        )
        return {
            "current_tokens": self._log.current_tokens(),
            "max_tokens": self._log.max_tokens,
            "chunks_active": chunk_count,
            "chunks_evicted": evicted,
            "compaction_events": self._log.compaction_count,
            "turn": self._log.turn(),
        }

    def _tool_clear(self, params: dict[str, Any]) -> dict[str, Any]:
        self._log._conn.execute("DELETE FROM chunks")
        self._log._conn.execute("DELETE FROM decisions")
        self._log._conn.commit()
        self._log._decisions.clear()
        self._log._turn = 0
        self._log._compaction_count = 0
        self._log._accumulated_keywords.clear()
        self._log._last_user_message = ""
        return {"status": "cleared"}

    # --- JSON-RPC dispatch ---

    def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
        }

    def _handle_tools_list(self, params: dict[str, Any]) -> dict[str, Any]:
        tool_defs = [
            {
                "name": "context/append",
                "description": "Append a chunk to the context log",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "enum": ["user", "assistant", "system"],
                            "default": "user",
                        },
                        "content": {"type": "string"},
                        "priority": {"type": "number", "default": 1.0},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "context/query",
                "description": "Get the current context window messages",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "context/status",
                "description": "Get engine status: tokens, chunks, compactions",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "context/clear",
                "description": "Reset the context log",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]
        return {"tools": tool_defs}

    def _handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        handler = self._tools.get(name)
        if handler is None:
            return {
                "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                "isError": True,
            }
        result = handler(arguments)
        return {
            "content": [{"type": "text", "text": json.dumps(result)}],
            "isError": False,
        }

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single JSON-RPC request. Returns response or None for notifications."""
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        dispatch = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
        }

        # Notifications (no id) that we acknowledge silently
        if method == "notifications/initialized":
            return None
        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        handler = dispatch.get(method)
        if handler is None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }

        result = handler(params)
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def run(self) -> None:
        """Read JSON-RPC messages from stdin, write responses to stdout."""
        sys.stderr.write(
            f"{SERVER_NAME} v{SERVER_VERSION} — MCP server ready (stdio)\n"
        )
        sys.stderr.flush()

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                error_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }
                sys.stdout.write(json.dumps(error_resp) + "\n")
                sys.stdout.flush()
                continue

            response = self.handle_request(request)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()


def run_server(
    max_tokens: int = 128_000,
    soft_threshold: float = 0.7,
    hard_threshold: float = 0.9,
    scoring_mode: str = "tfidf",
) -> None:
    """Entry point for the MCP server."""
    server = MCPServer(
        max_tokens=max_tokens,
        soft_threshold=soft_threshold,
        hard_threshold=hard_threshold,
        scoring_mode=scoring_mode,
    )
    server.run()
