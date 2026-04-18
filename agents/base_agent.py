"""
BaseAgent — shared Claude API wrapper for all Mobius-Nova subagents.

Every specialist agent inherits from this class and gets:
  • Anthropic client initialisation (reads ANTHROPIC_API_KEY from env)
  • Structured message calling with auto-retry
  • Tool-use plumbing (JSON-schema tool definitions)
  • Structured result logging
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Returned by every agent.run() call."""

    agent_name: str
    task: str
    success: bool
    output: Any = None          # Parsed result (dict, str, etc.)
    raw_text: str = ""          # Raw LLM text
    tool_calls: list[dict] = field(default_factory=list)
    error: str = ""
    attempts: int = 1
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "agent": self.agent_name,
            "task": self.task,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "attempts": self.attempts,
            "elapsed_s": round(self.elapsed_s, 2),
        }


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------

class BaseAgent:
    """Thin wrapper around the Anthropic Messages API.

    Subclasses override:
        • AGENT_NAME   – human-readable label
        • SYSTEM_PROMPT – role definition injected into every call
        • tools()       – list of JSON-schema tool dicts (optional)
        • _parse_result() – extract structured output from raw LLM text
    """

    AGENT_NAME: str = "BaseAgent"
    SYSTEM_PROMPT: str = "You are a helpful AI assistant."

    # Model routing: use Opus for the orchestrator, Sonnet for leaf agents
    DEFAULT_MODEL: str = "claude-sonnet-4-6"
    MAX_TOKENS: int = 4096
    MAX_RETRIES: int = 3
    RETRY_DELAY_S: float = 2.0

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self.model = model or self.DEFAULT_MODEL
        self.verbose = verbose
        self._log = logger.getChild(self.AGENT_NAME)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, task: str, context: dict | None = None) -> AgentResult:
        """Execute the agent on *task*, returning a structured AgentResult."""
        start = time.monotonic()
        attempts = 0
        last_error = ""

        prompt = self._build_prompt(task, context or {})

        for attempt in range(1, self.MAX_RETRIES + 1):
            attempts = attempt
            try:
                response = self._call_api(prompt)
                raw_text = self._extract_text(response)
                tool_calls = self._extract_tool_calls(response)
                output = self._parse_result(raw_text, tool_calls)

                elapsed = time.monotonic() - start
                result = AgentResult(
                    agent_name=self.AGENT_NAME,
                    task=task,
                    success=True,
                    output=output,
                    raw_text=raw_text,
                    tool_calls=tool_calls,
                    attempts=attempts,
                    elapsed_s=elapsed,
                )
                if self.verbose:
                    self._log.info("✅ %s completed in %.1fs", task[:60], elapsed)
                return result

            except anthropic.APIStatusError as exc:
                last_error = f"API error {exc.status_code}: {exc.message}"
                self._log.warning("Attempt %d/%d failed: %s", attempt, self.MAX_RETRIES, last_error)
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY_S * attempt)

            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                self._log.warning("Attempt %d/%d failed: %s", attempt, self.MAX_RETRIES, last_error)
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY_S * attempt)

        elapsed = time.monotonic() - start
        return AgentResult(
            agent_name=self.AGENT_NAME,
            task=task,
            success=False,
            error=last_error,
            attempts=attempts,
            elapsed_s=elapsed,
        )

    # ------------------------------------------------------------------
    # Override hooks
    # ------------------------------------------------------------------

    def tools(self) -> list[dict]:
        """Return JSON-schema tool definitions. Override in subclasses."""
        return []

    def _parse_result(self, raw_text: str, tool_calls: list[dict]) -> Any:
        """Parse raw LLM output into structured data. Override if needed."""
        # Try JSON first; fall back to plain text
        try:
            return json.loads(raw_text)
        except (json.JSONDecodeError, ValueError):
            return raw_text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, task: str, context: dict) -> str:
        if not context:
            return task
        ctx_str = json.dumps(context, indent=2)
        return f"{task}\n\n<context>\n{ctx_str}\n</context>"

    def _call_api(self, prompt: str) -> anthropic.types.Message:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.MAX_TOKENS,
            "system": self.SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }
        tool_defs = self.tools()
        if tool_defs:
            kwargs["tools"] = tool_defs

        if self.verbose:
            self._log.debug("→ %s | model=%s | tokens=%d", self.AGENT_NAME, self.model, self.MAX_TOKENS)

        return self.client.messages.create(**kwargs)

    @staticmethod
    def _extract_text(response: anthropic.types.Message) -> str:
        parts = [b.text for b in response.content if hasattr(b, "text")]
        return "\n".join(parts)

    @staticmethod
    def _extract_tool_calls(response: anthropic.types.Message) -> list[dict]:
        return [
            {"name": b.name, "input": b.input}
            for b in response.content
            if b.type == "tool_use"
        ]
