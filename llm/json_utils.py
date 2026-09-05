"""Defensive JSON extraction for LLM output.

Small models wrap JSON in prose, in `````json`` fences, or emit
Python literals (``True``/``None``) and trailing commas. Every helper here is
pure and deterministic, which is why the sanity tests can cover it without a
network call.

The contract used throughout the agent is:

1. Ask for JSON with an explicit schema in the prompt.
2. Parse leniently with :func:`loads_lenient`.
3. On failure, re-ask **once** with :data:`RETRY_INSTRUCTION` plus the parser
   error (see :meth:`llm.client.LLMClient.complete_json`).
4. On a second failure, raise :class:`JSONParseError`; the calling node records
   an ``error`` decision event and the orchestrator degrades gracefully rather
   than crashing the run.
"""

from __future__ import annotations

import json
import re
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

RETRY_INSTRUCTION = (
    "Your last output was invalid JSON and could not be parsed. "
    "Parser error: {error}\n"
    "Respond again with ONLY the JSON value that matches the schema. "
    "No prose, no markdown fences, no comments, no trailing commas. "
    "Start your response with the opening brace or bracket."
)


class JSONParseError(ValueError):
    """Raised when LLM output cannot be coerced into JSON after repair."""

    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw
        self.short_raw = raw[:600]


_FENCE_RE = re.compile(r"^\s*```(?:json|JSON|python|js)?\s*\n?|\n?```\s*$")
_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_LINE_COMMENT_RE = re.compile(r"(^|\s)//[^\n\"']*$", re.MULTILINE)
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

_SMART_QUOTES = {
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "«": '"',
    "»": '"',
}


def strip_code_fences(text: str) -> str:
    """Remove markdown fences and reasoning tags that wrap the payload."""
    if not text:
        return ""
    cleaned = _THINK_RE.sub("", text).strip()
    if cleaned.startswith("```"):
        cleaned = _FENCE_RE.sub("", cleaned)
        # A closing fence may remain if the opening one had a language tag.
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    return cleaned.strip()


def find_json_span(text: str) -> str | None:
    """Return the first balanced JSON object or array found in ``text``.

    Scans with a depth counter while honouring string literals and escapes, so
    a brace inside ``"expected_outcome": "shows {0} results"`` does not confuse
    the extractor. Prefers whichever of ``{`` / ``[`` appears first.
    """
    if not text:
        return None
    start = None
    opener = ""
    for index, char in enumerate(text):
        if char in "{[":
            start = index
            opener = char
            break
    if start is None:
        return None

    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    # Unterminated: return the remainder so the repair pass can try to close it.
    return text[start:]


def repair_common_json_issues(text: str) -> str:
    """Apply conservative textual fixes for the mistakes small models make.

    Deliberately conservative: we never try to rewrite single-quoted strings
    into double-quoted ones, because that mangles apostrophes in prose. If the
    payload is broken beyond these fixes we re-ask the model instead.
    """
    if not text:
        return text
    out = text
    for bad, good in _SMART_QUOTES.items():
        out = out.replace(bad, good)
    out = _BLOCK_COMMENT_RE.sub("", out)
    out = _LINE_COMMENT_RE.sub(r"\1", out)
    out = _TRAILING_COMMA_RE.sub(r"\1", out)
    # Python literals leaking out of a code-trained model.
    out = re.sub(r"(?<![\w\"])True(?![\w\"])", "true", out)
    out = re.sub(r"(?<![\w\"])False(?![\w\"])", "false", out)
    out = re.sub(r"(?<![\w\"])None(?![\w\"])", "null", out)
    out = re.sub(r"(?<![\w\"])NaN(?![\w\"])", "null", out)
    # Close an unterminated object/array by balancing the brackets we opened.
    out = _balance_brackets(out)
    return out.strip()


def _balance_brackets(text: str) -> str:
    """Append the closing brackets needed to balance a truncated payload."""
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if stack and ((char == "}" and stack[-1] == "{") or (char == "]" and stack[-1] == "[")):
                stack.pop()
    if not stack and not in_string:
        return text
    tail = '"' if in_string else ""
    # Drop a dangling ``"key":`` or trailing comma before closing.
    body = (text + tail).rstrip()
    body = re.sub(r",\s*$", "", body)
    body = re.sub(r'"[^"]*"\s*:\s*$', "", body).rstrip().rstrip(",")
    for opener in reversed(stack):
        body += "}" if opener == "{" else "]"
    return body


def loads_lenient(text: str) -> Any:
    """Parse JSON from noisy model output.

    Order of attempts: strict parse, fenced-block strip, balanced-span
    extraction, then textual repair. Raises :class:`JSONParseError` if all of
    them fail.
    """
    if text is None:
        raise JSONParseError("model returned no content", "")
    raw = str(text)
    if not raw.strip():
        raise JSONParseError("model returned an empty response", raw)

    attempts: list[str] = []
    stripped = strip_code_fences(raw)
    attempts.append(stripped)

    span = find_json_span(stripped)
    if span and span != stripped:
        attempts.append(span)

    attempts.append(repair_common_json_issues(span or stripped))

    last_error: Exception | None = None
    for candidate in attempts:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            continue

    raise JSONParseError(
        f"could not parse JSON: {last_error}" if last_error else "could not parse JSON",
        raw,
    )


def parse_model(text: str, model_cls: Type[T]) -> T:
    """Parse LLM output straight into a pydantic model."""
    data = loads_lenient(text)
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise JSONParseError(f"schema validation failed: {exc}", str(text)) from exc


def coerce_list(value: Any, key: str | None = None) -> list[Any]:
    """Normalise the several shapes a model uses for "a list of things".

    Accepts a bare list, ``{"<key>": [...]}``, ``{"items": [...]}``, or a
    single object that should have been a one-element list.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if key and isinstance(value.get(key), list):
            return value[key]
        for fallback in ("items", "results", "data", "list", "flows", "values"):
            if isinstance(value.get(fallback), list):
                return value[fallback]
        return [value]
    return [value]


def build_retry_messages(
    original_messages: list[dict[str, str]],
    bad_output: str,
    error: str,
) -> list[dict[str, str]]:
    """Append the assistant's bad answer plus the repair instruction.

    Keeping the invalid output in the transcript matters: without it the model
    has no idea what it is being asked to correct.
    """
    return [
        *original_messages,
        {"role": "assistant", "content": (bad_output or "")[:4000]},
        {"role": "user", "content": RETRY_INSTRUCTION.format(error=error)},
    ]
