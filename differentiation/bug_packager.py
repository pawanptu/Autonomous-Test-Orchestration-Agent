"""Bug packaging: turning a confirmed defect into something an engineer can act on.

A line in a report is not a bug report. When the Healer concludes that a failure
is a ``GENUINE_DEFECT`` this module writes a self-contained folder per defect::

    reports/runs/<run_id>/bugs/BUG-001/
        repro.py        standalone async Playwright script, runnable on its own
        screenshot.png  the frame captured at the moment of failure
        ticket.md       paste-ready GitHub Issues / Jira body
        bug.json        the PackagedBug model, redacted

Why a runnable repro is the deliverable
---------------------------------------
The expensive part of a bug report is not the prose, it is the twenty minutes an
engineer spends re-creating the state. The repro script is therefore compiled
from the flow's own steps, carries no project imports, no fixtures and no test
runner, and **ends by asserting the flow's expected outcome** - so running it
while the defect is present fails loudly, and running it after a fix passes.
That makes the artifact a verification tool, not just documentation.

Credential safety
-----------------
The repro script is written to disk and handed to a human, so it is the single
most dangerous file this agent produces. Three defences apply, in order:

1. values that look like credentials are emitted as ``os.environ.get(...)``
   lookups rather than literals in the first place;
2. the rendered source is passed through :func:`security.assert_no_secret_literals`
   and, on a hit, through :func:`security.redact_text` before the literal is
   swapped for an environment lookup - the value itself is never written;
3. every file, without exception, is redacted again on the way to disk.

Degraded paths, honestly
------------------------
* If the model is unavailable, rate-limited or returns unparsable JSON, a
  deterministic ticket is built from the flow definition and the failure
  evidence. The bug is never dropped, and the description says which path
  produced it.
* If the flow's expected outcome carries no assertable phrase (it was empty),
  the repro script replays the steps and prints a ``MANUAL CHECK REQUIRED``
  banner instead of asserting something it cannot derive. It does not invent an
  assertion that would always fail and pretend that is a reproduction.
* If a file cannot be written (permissions, disk), the corresponding path field
  on :class:`~graph.state.PackagedBug` stays ``None`` and the run continues.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urljoin

from browser.selectors import (
    escape_py_string,
    expression_for,
    extract_quoted,
    guess_role,
    normalize_intent,
)
from config import get_settings
from graph.state import (
    DefectClass,
    GeneratedTest,
    HealerAction,
    PackagedBug,
    RiskLevel,
    TestFlow,
    TestResult,
    TestStep,
)
from llm.client import LLMClient, ModelRole
from llm.json_utils import JSONParseError, coerce_list
from llm.prompts import BUG_SYSTEM, bug_user
from logging_setup import get_logger
from security import (
    assert_no_secret_literals,
    redact_secrets,
    redact_text,
    sanitize_url,
)

log = get_logger("aivor.bugs")

EmitFn = Callable[[str, str, str | None, float | None], None]
"""``(summary, detail, risk, confidence)`` - one call per packaged bug."""

BUG_DIRNAME = "bugs"
REPRO_FILENAME = "repro.py"
SCREENSHOT_FILENAME = "screenshot.png"
TICKET_FILENAME = "ticket.md"
BUG_JSON_FILENAME = "bug.json"

_BUG_ID_RE = re.compile(r"BUG-(\d+)", re.IGNORECASE)

Severity = str  # "critical" | "major" | "minor"; PackagedBug enforces the literal.

_SEVERITY_BY_RISK: dict[RiskLevel, Severity] = {
    RiskLevel.HIGH: "critical",
    RiskLevel.MEDIUM: "major",
    RiskLevel.LOW: "minor",
}

# Wording that reads as a credential even when the planner meant it innocently.
# A step matching one of these never gets a literal value written to disk.
_CREDENTIAL_TARGET_RE = re.compile(
    r"(?i)\b(pass(word|wd)?|pwd|secret|token|api[_-]?key|otp|one[- ]time code)\b"
)
_USERNAME_TARGET_RE = re.compile(r"(?i)\b(user ?name|user|e-?mail|login|account)\b")

# Words that carry no discriminating power when we mine the expected outcome for
# keywords to assert on.
_OUTCOME_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "with", "that", "this", "then", "when", "shows", "show",
        "should", "displays", "display", "page", "user", "text", "message",
        "your", "after", "from", "into", "have", "has", "was", "were", "are",
        "is", "be", "been", "its", "it", "for", "not", "but", "all", "any",
    }
)


# --------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------
def _enum_value(value: Any) -> str:
    """Read ``.value`` off an enum, tolerating a plain string from loose JSON."""
    if isinstance(value, Enum):
        return str(value.value)
    return str(value or "")


def _pystr(value: Any) -> str:
    """Render ``value`` as a double-quoted Python string literal, fully escaped."""
    return '"' + escape_py_string("" if value is None else str(value)) + '"'


def _safe_doc(text: str) -> str:
    """Make ``text`` safe to embed inside a triple-quoted docstring."""
    cleaned = redact_text(str(text or "")).replace('"""', "'''").replace("\\", "/")
    return cleaned.strip()


def _clamp(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _looks_like_url(text: str) -> bool:
    candidate = (text or "").strip()
    return candidate.startswith(("http://", "https://", "/")) or bool(
        re.match(r"^[\w.-]+\.[a-z]{2,}(/|$)", candidate, re.I)
    )


def _looks_like_css(text: str) -> bool:
    candidate = (text or "").strip()
    return bool(candidate) and (
        candidate.startswith(("#", ".", "[")) or ">" in candidate or "::" in candidate
    )


# --------------------------------------------------------------------------
# Bug identity
# --------------------------------------------------------------------------
def next_bug_id(existing: Sequence[str]) -> str:
    """Return the next ``BUG-NNN`` id, continuing after the highest one seen.

    Ids are allocated from the *set of strings already in play* rather than a
    counter, so re-entering the packager after a second heal pass - or resuming
    a run whose ``bugs/`` directory already holds folders - never re-uses an id
    and never silently overwrites an existing artifact directory. Entries that
    are not bug ids are ignored rather than raising.
    """
    highest = 0
    for item in existing or ():
        match = _BUG_ID_RE.search(str(item or ""))
        if not match:
            continue
        try:
            highest = max(highest, int(match.group(1)))
        except ValueError:  # pragma: no cover - the regex guarantees digits
            continue
    return f"BUG-{highest + 1:03d}"


def _existing_bug_ids(bugs_dir: Path) -> list[str]:
    """Bug ids already materialised on disk. Missing directory means none."""
    try:
        return [p.name for p in bugs_dir.iterdir() if p.is_dir() and _BUG_ID_RE.fullmatch(p.name)]
    except (OSError, ValueError):
        return []


# --------------------------------------------------------------------------
# Repro script synthesis
# --------------------------------------------------------------------------
_EXPRESSION_RE = re.compile(
    r"page\.(?:get_by_role|get_by_test_id|get_by_label|get_by_placeholder|"
    r"get_by_text|get_by_alt_text|get_by_title|locator)\((?:[^()\"']|\"[^\"]*\"|'[^']*')*\)"
)


def extract_locator_expressions(test_source: str | None) -> list[str]:
    """Pull the locator expressions the Generator actually resolved, in order.

    These beat anything we could re-derive from the plan's prose, because they
    were probed against the live DOM before the test ran. Parsing is textual and
    deliberately forgiving: a miss just means we fall back to the heuristic
    locator, never an exception.
    """
    if not test_source:
        return []
    seen: list[str] = []
    for match in _EXPRESSION_RE.finditer(test_source):
        expression = match.group(0)
        if expression not in seen:
            seen.append(expression)
    return seen


def _match_expression(step: TestStep, expressions: Sequence[str]) -> str | None:
    """Pick the resolved expression that best matches this step's intent.

    Scored on token overlap with the step's normalised target. We require a
    majority of the intent's tokens to appear so that "the Login button" does
    not silently borrow the locator for "the Logout link".
    """
    intent = normalize_intent(step.target) or normalize_intent(step.description)
    tokens = [t for t in re.split(r"[^\w]+", intent.lower()) if len(t) >= 3]
    if not tokens or not expressions:
        return None
    best: tuple[float, str] | None = None
    for expression in expressions:
        haystack = expression.lower()
        hits = sum(1 for token in tokens if token in haystack)
        score = hits / len(tokens)
        if score >= 0.6 and (best is None or score > best[0]):
            best = (score, expression)
    return best[1] if best else None


def _heuristic_expression(step: TestStep) -> str:
    """Derive a readable locator from the step's plain-language target.

    Mirrors :mod:`browser.selectors` strategy ordering so the repro script reads
    like the generated test even when we have no resolved expression to borrow.
    """
    raw = (step.target or step.description or "").strip()
    intent = normalize_intent(raw)
    if not intent:
        # Nothing discriminating survived normalisation; fall back to the raw
        # text so the reader at least sees what the step meant.
        intent = raw
    if _looks_like_css(raw):
        return expression_for("css", raw)
    if not intent:
        return 'page.locator("body")'
    if step.action in ("fill", "press", "select", "check"):
        return expression_for("label", intent)
    if step.action in ("assert_text", "assert_visible", "assert_not_visible", "wait_for"):
        return expression_for("text", intent)
    return expression_for("role", intent, role=guess_role(intent, step.action) or "button")


def _fill_value_expression(step: TestStep) -> tuple[str, bool]:
    """Return the Python expression for a ``fill`` value, plus a credential flag.

    A value that the planner marked as a password - or that the redactor can
    still see as a registered secret - is emitted as ``credential("password")``,
    which reads ``LOGIN_PASSWORD`` from the environment at run time. The literal
    never reaches the file.
    """
    value = step.value or ""
    target_blob = f"{step.target} {step.description}"
    if _CREDENTIAL_TARGET_RE.search(target_blob) or assert_no_secret_literals(value):
        return 'credential("password")', True
    if _USERNAME_TARGET_RE.search(target_blob) and not value:
        return 'credential("username")', True
    if not value:
        return _pystr(""), False
    return _pystr(redact_text(value)), False


def _goto_url(step: TestStep, flow: TestFlow, base_url: str) -> str:
    """Resolve the absolute URL a ``goto`` step should open, sanitised."""
    candidates = [step.value or "", step.target or "", flow.url or "", base_url or ""]
    root = (base_url or flow.url or "").strip()
    for candidate in candidates:
        text = (candidate or "").strip()
        if not _looks_like_url(text):
            continue
        if text.startswith(("http://", "https://")):
            return sanitize_url(text)
        if root:
            return sanitize_url(urljoin(root, text))
        return sanitize_url(text)
    return sanitize_url(root)


def _expected_keywords(expected: str, limit: int = 4) -> list[str]:
    """Mine the expected outcome for words that must appear on the healthy page."""
    normalised = normalize_intent(expected) or (expected or "").lower()
    words: list[str] = []
    for token in re.split(r"[^\w'-]+", normalised):
        cleaned = token.strip("'-").lower()
        if len(cleaned) < 3 or cleaned in _OUTCOME_STOPWORDS or cleaned in words:
            continue
        words.append(cleaned)
        if len(words) >= limit:
            break
    return words


def _translate_step(
    step: TestStep,
    index: int,
    *,
    flow: TestFlow,
    base_url: str,
    expressions: Sequence[str],
) -> list[str]:
    """Translate one plan step into indented Playwright source lines.

    Returns the comment line plus the call lines. An unknown action degrades to
    a ``TODO`` comment rather than emitting code we cannot justify - a repro
    script that silently skips a step is worse than one that says it did.
    """
    description = _safe_doc(step.description or step.target or step.action) or step.action
    lines = [f"    # Step {index}: {description}"]

    action = step.action
    locator = _match_expression(step, expressions) or _heuristic_expression(step)

    if action == "goto":
        url = _goto_url(step, flow, base_url)
        lines.append(
            f"    await page.goto({_pystr(url)}, wait_until=\"domcontentloaded\", "
            "timeout=NAV_TIMEOUT_MS)"
        )
        return lines

    if action == "click":
        lines.append(f"    await {locator}.click()")
        return lines

    if action == "fill":
        expression, is_credential = _fill_value_expression(step)
        lines.append(f"    await {locator}.fill({expression})")
        if is_credential:
            lines.append("    # Value comes from the environment; it is never stored here.")
        return lines

    if action == "select":
        lines.append(f"    await {locator}.select_option({_pystr(redact_text(step.value or ''))})")
        return lines

    if action == "check":
        lines.append(f"    await {locator}.check()")
        return lines

    if action == "press":
        key = (step.value or "Enter").strip() or "Enter"
        if step.target:
            lines.append(f"    await {locator}.press({_pystr(key)})")
        else:
            lines.append(f"    await page.keyboard.press({_pystr(key)})")
        return lines

    if action == "wait_for":
        if _looks_like_url(step.target):
            lines.append('    await page.wait_for_load_state("networkidle")')
        else:
            lines.append(
                f"    await {locator}.wait_for(state=\"visible\", timeout=ACTION_TIMEOUT_MS)"
            )
        return lines

    if action == "assert_text":
        needle = step.value or (extract_quoted(step.target) or [""])[0] or step.target
        lines.append(
            f"    await expect({locator}).to_contain_text({_pystr(redact_text(needle))}, "
            "timeout=ACTION_TIMEOUT_MS)"
        )
        return lines

    if action == "assert_visible":
        lines.append(f"    await expect({locator}).to_be_visible(timeout=ACTION_TIMEOUT_MS)")
        return lines

    if action == "assert_not_visible":
        lines.append(f"    await expect({locator}).to_be_hidden(timeout=ACTION_TIMEOUT_MS)")
        return lines

    if action == "assert_url":
        fragment = (step.value or step.target or "").strip()
        lines.append(f"    fragment = {_pystr(sanitize_url(fragment))}")
        lines.append(
            "    assert fragment in page.url, "
            '"expected the URL to contain " + fragment + ", got " + page.url'
        )
        return lines

    if action == "screenshot":
        lines.append(f'    await page.screenshot(path="repro_step_{index}.png", full_page=True)')
        return lines

    lines.append(f"    # TODO unsupported action {action!r}; perform this step by hand.")
    return lines


def _final_assertion_lines(flow: TestFlow) -> list[str]:
    """Emit the assertion that makes running the script a real reproduction.

    Preference order: a quoted phrase from the expected outcome (asserted as
    visible text), then a keyword check against the page body. When the flow
    carries no expected outcome at all we print a manual-check banner - see the
    module docstring on degraded paths.
    """
    expected = (flow.expected_outcome or "").strip()
    lines = [
        "",
        "    # ---- the assertion that reproduces the defect --------------------",
        "    # This is the line that fails while the bug is present. Do not weaken",
        "    # it to make the script pass; that would hide the defect.",
    ]
    if not expected:
        lines.append(
            '    print("[repro] MANUAL CHECK REQUIRED: the plan carried no expected "'
        )
        lines.append('          "outcome, so this script replays the steps only.")')
        return lines

    phrases = extract_quoted(expected)
    if phrases:
        lines.append(
            f"    await expect(page.get_by_text({_pystr(redact_text(phrases[0]))})).to_be_visible("
        )
        lines.append("        timeout=ACTION_TIMEOUT_MS")
        lines.append("    )")
        return lines

    keywords = _expected_keywords(expected)
    if keywords:
        lines.append('    body_text = (await page.locator("body").inner_text()).lower()')
        lines.append("    missing = [k for k in EXPECTED_KEYWORDS if k not in body_text]")
        lines.append("    assert not missing, (")
        lines.append('        "Expected outcome not observed: " + EXPECTED_OUTCOME')
        lines.append('        + " (missing from the page: " + ", ".join(missing) + ")"')
        lines.append("    )")
        return lines

    lines.append(
        '    print("[repro] MANUAL CHECK REQUIRED: the expected outcome could not be "'
    )
    lines.append('          "reduced to an assertable phrase. Verify it by eye: " + EXPECTED_OUTCOME)')
    return lines


def _render_script(
    *,
    flow: TestFlow,
    result: TestResult,
    base_url: str,
    bug_id: str,
    expressions: Sequence[str],
    include_credentials: bool,
) -> str:
    """Render the full repro module source. Pure string assembly, never raises."""
    try:
        cfg = get_settings()
        viewport_w, viewport_h = cfg.viewport_width, cfg.viewport_height
        action_timeout, nav_timeout = cfg.action_timeout_ms, cfg.nav_timeout_ms
    except Exception:  # pragma: no cover - settings must never block a bug artifact
        viewport_w, viewport_h = 1280, 900
        action_timeout, nav_timeout = 8_000, 25_000

    expected = _safe_doc(flow.expected_outcome)
    observed = _safe_doc(result.error_message or _enum_value(result.status) or "the flow failed")
    label = bug_id or "the packaged defect"
    resolved_base = sanitize_url(base_url or flow.url or "")

    head: list[str] = [
        '"""Standalone reproduction for ' + f"{label} - {_safe_doc(flow.name) or flow.id}." + "",
        "",
        "Auto-generated by the Autonomous Test Orchestration Agent from the failing",
        f"flow {_safe_doc(flow.name) or flow.id!r} (flow id: {flow.id}). It is deliberately",
        "self-contained: no project imports, no fixtures, no test runner.",
        "",
        "Run it with::",
        "",
        "    pip install playwright",
        "    playwright install chromium",
        "    python repro.py",
        "",
        f"Expected: {expected or '(the plan carried no expected outcome)'}",
        f"Observed: {observed}",
        "",
        "Exit code 1 means the defect still reproduces. Exit code 0 means the",
        "expected outcome held and the bug may already be fixed.",
        "",
        "No credential is stored in this file. Set LOGIN_USERNAME and LOGIN_PASSWORD",
        "in your environment if this flow needs to sign in.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import asyncio",
        "import os",
        "import sys",
        "",
        "from playwright.async_api import async_playwright, expect",
        "",
        f"BASE_URL = {_pystr(resolved_base)}",
        'HEADLESS = os.environ.get("HEADLESS", "1").strip().lower() not in ("0", "false", "no")',
        f'VIEWPORT = {{"width": {viewport_w}, "height": {viewport_h}}}',
        f"ACTION_TIMEOUT_MS = {action_timeout}",
        f"NAV_TIMEOUT_MS = {nav_timeout}",
        "",
        f"EXPECTED_OUTCOME = {_pystr(redact_text(flow.expected_outcome or ''))}",
        f"EXPECTED_KEYWORDS = {_expected_keywords(flow.expected_outcome or '')!r}",
        "",
    ]

    if include_credentials:
        head.extend(
            [
                'LOGIN_USERNAME = os.environ.get("LOGIN_USERNAME", "")',
                'LOGIN_PASSWORD = os.environ.get("LOGIN_PASSWORD", "")',
                "",
                "",
                "def credential(kind: str) -> str:",
                '    """Read a credential from the environment.',
                "",
                "    The value is never written to this file. An unset variable is",
                "    reported loudly rather than silently typed as an empty string,",
                "    because a blank login looks like an application defect.",
                '    """',
                '    variable = "LOGIN_USERNAME" if kind == "username" else "LOGIN_PASSWORD"',
                '    value = LOGIN_USERNAME if kind == "username" else LOGIN_PASSWORD',
                "    if not value:",
                '        print("[repro] " + variable + " is not set. Export it and re-run, "',
                '              "otherwise this step submits an empty value.")',
                "    return value",
                "",
            ]
        )

    body: list[str] = [
        "",
        "async def reproduce(page) -> None:",
        '    """Replay the failing flow and assert the outcome the app should produce."""',
    ]
    steps = list(flow.steps)
    if not steps:
        body.append(
            "    # The plan recorded no steps for this flow; open the entry point and"
        )
        body.append("    # assert the expected outcome directly.")
        body.append(
            '    await page.goto(BASE_URL, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)'
        )
    else:
        opens_with_goto = steps[0].action == "goto"
        if not opens_with_goto:
            body.append("    # The plan assumed an already-open page; start from the entry point.")
            body.append(
                '    await page.goto(BASE_URL, wait_until="domcontentloaded", '
                "timeout=NAV_TIMEOUT_MS)"
            )
        for index, step in enumerate(steps, start=1):
            try:
                body.extend(
                    _translate_step(
                        step, index, flow=flow, base_url=base_url, expressions=expressions
                    )
                )
            except Exception as exc:  # one bad step must not cost us the artifact
                log.warning("could not translate step %d of flow %s: %s", index, flow.id, exc)
                body.append(f"    # Step {index} could not be translated automatically.")

    body.extend(_final_assertion_lines(flow))

    tail: list[str] = [
        "",
        "",
        "async def main() -> None:",
        '    """Drive the reproduction and report whether the defect is still present."""',
        f'    print("[repro] {label}: {_safe_doc(flow.name) or flow.id}")',
        "    reproduced = False",
        "    async with async_playwright() as pw:",
        "        browser = await pw.chromium.launch(headless=HEADLESS)",
        "        context = await browser.new_context(viewport=VIEWPORT)",
        "        page = await context.new_page()",
        "        page.set_default_timeout(ACTION_TIMEOUT_MS)",
        "        try:",
        "            await reproduce(page)",
        '            print("[repro] The expected outcome held; the defect did NOT reproduce.")',
        "        except Exception as exc:",
        "            reproduced = True",
        '            print("[repro] DEFECT REPRODUCED: " + type(exc).__name__ + ": " + str(exc))',
        "            try:",
        '                await page.screenshot(path="repro_failure.png", full_page=True)',
        '                print("[repro] Wrote repro_failure.png next to this script.")',
        "            except Exception:",
        '                print("[repro] Could not capture a failure screenshot.")',
        "        finally:",
        "            await context.close()",
        "            await browser.close()",
        "    sys.exit(1 if reproduced else 0)",
        "",
        "",
        'if __name__ == "__main__":',
        "    asyncio.run(main())",
        "",
    ]

    return "\n".join(head + body + tail)


def _minimal_script(*, flow: TestFlow, base_url: str, bug_id: str, reason: str) -> str:
    """The honest degraded repro: open the page, state what to check by hand.

    Used only when the step-by-step translation produced source that would not
    compile. Shipping a script that cannot run would be worse than shipping one
    that admits what it could not automate.
    """
    return "\n".join(
        [
            '"""Minimal reproduction stub for ' + f"{bug_id or 'the packaged defect'}." + "",
            "",
            "The full step-by-step translation could not be generated for this flow",
            f"({_safe_doc(reason)}), so this script opens the entry point and states",
            "what to verify by hand. It does not fake an assertion it cannot derive.",
            '"""',
            "",
            "from __future__ import annotations",
            "",
            "import asyncio",
            "import os",
            "",
            "from playwright.async_api import async_playwright",
            "",
            f"BASE_URL = {_pystr(sanitize_url(base_url or flow.url or ''))}",
            'HEADLESS = os.environ.get("HEADLESS", "1").strip().lower() not in ("0", "false", "no")',
            f"EXPECTED_OUTCOME = {_pystr(redact_text(flow.expected_outcome or ''))}",
            f"STEPS = {[_safe_doc(s.description or s.target or s.action) for s in flow.steps]!r}",
            "",
            "",
            "async def main() -> None:",
            '    """Open the flow entry point and print the steps to verify by hand."""',
            "    async with async_playwright() as pw:",
            "        browser = await pw.chromium.launch(headless=HEADLESS)",
            "        page = await (await browser.new_context()).new_page()",
            '        await page.goto(BASE_URL, wait_until="domcontentloaded")',
            '        print("[repro] MANUAL CHECK REQUIRED. Perform these steps:")',
            "        for index, step in enumerate(STEPS, start=1):",
            '            print("  " + str(index) + ". " + step)',
            '        print("[repro] Expected outcome: " + EXPECTED_OUTCOME)',
            "        await browser.close()",
            "",
            "",
            'if __name__ == "__main__":',
            "    asyncio.run(main())",
            "",
        ]
    )


def build_repro_script(
    *,
    flow: TestFlow,
    result: TestResult,
    base_url: str,
    test_source: str | None = None,
    bug_id: str = "",
) -> str:
    """Compose a standalone async Playwright script that reproduces the defect.

    The script is assembled from the flow's own steps so that it reads like the
    journey a human would take, enriched where possible with the locator
    expressions the Generator already proved against the live DOM. It always
    ends by asserting the flow's expected outcome, so running it while the bug
    is present fails - that is the point of the artifact.

    Security: the rendered source is checked with
    :func:`security.assert_no_secret_literals`. If a registered secret survived
    into the text, the source is redacted and the offending literal is replaced
    by an ``os.environ`` lookup; the value itself is never written. The result
    is compiled before it is returned, and a source that will not compile is
    replaced by the honest minimal stub rather than shipped broken.
    """
    expressions = extract_locator_expressions(test_source)
    needs_credentials = bool(flow.requires_auth) or any(
        _CREDENTIAL_TARGET_RE.search(f"{s.target} {s.description}")
        or (s.value and assert_no_secret_literals(s.value))
        for s in flow.steps
    )

    source = _render_script(
        flow=flow,
        result=result,
        base_url=base_url,
        bug_id=bug_id,
        expressions=expressions,
        include_credentials=needs_credentials,
    )

    hits = assert_no_secret_literals(source)
    if hits:
        # A registered credential reached the text anyway (an echoed step value,
        # a URL with userinfo). Scrub the value, then hand the call site an
        # environment lookup so the script stays runnable without it.
        log.warning(
            "repro script for flow %s contained %d secret literal(s); replacing with "
            "environment lookups",
            flow.id,
            len(hits),
        )
        source = redact_text(source)
        source = source.replace('"***REDACTED***"', 'credential("password")')
        source = source.replace("***REDACTED***", "REDACTED")
        if not needs_credentials:
            source = _render_script(
                flow=flow,
                result=result,
                base_url=base_url,
                bug_id=bug_id,
                expressions=expressions,
                include_credentials=True,
            )
            source = redact_text(source)
            source = source.replace('"***REDACTED***"', 'credential("password")')
            source = source.replace("***REDACTED***", "REDACTED")

    source = redact_text(source)
    try:
        compile(source, f"<repro:{flow.id}>", "exec")
    except SyntaxError as exc:
        log.warning("generated repro for flow %s did not compile (%s); using the stub", flow.id, exc)
        return redact_text(
            _minimal_script(flow=flow, base_url=base_url, bug_id=bug_id, reason=str(exc.msg))
        )
    return source


# --------------------------------------------------------------------------
# Ticket rendering
# --------------------------------------------------------------------------
def render_ticket_markdown(bug: PackagedBug, *, run_id: str) -> str:
    """Render the paste-ready GitHub Issues / Jira body for one packaged bug.

    Structure is fixed on purpose - title, description, numbered steps,
    expected, actual, evidence, impact, provenance footer - because a reviewer
    scanning ten of these should not have to re-learn the layout each time. The
    footer names the agent and the run id so nobody mistakes an auto-filed
    ticket for a hand-written one.
    """
    risk = _enum_value(bug.risk) or RiskLevel.MEDIUM.value
    severity = str(bug.severity or "major")
    lines: list[str] = [
        f"# {redact_text(bug.title or 'Untitled defect')}",
        "",
        f"**Severity:** {severity}  |  **Business risk:** {risk.upper()}  "
        f"|  **Flow:** {redact_text(bug.flow_name or bug.flow_id)}  "
        f"|  **Classification:** {_enum_value(bug.classification)} "
        f"(confidence {_clamp(bug.confidence):.2f})",
        "",
        redact_text(bug.description or "").strip() or "_No description was produced._",
        "",
        "## Steps to reproduce",
        "",
    ]

    steps = [redact_text(str(s)).strip() for s in bug.steps_to_reproduce if str(s).strip()]
    if steps:
        lines.extend(f"{index}. {step}" for index, step in enumerate(steps, start=1))
    else:
        lines.append("1. Run the attached `repro.py` (the plan recorded no readable steps).")
    lines.extend(
        [
            "",
            "## Expected",
            "",
            redact_text(bug.expected or "").strip() or "_Not stated by the plan._",
            "",
            "## Actual",
            "",
            redact_text(bug.actual or "").strip() or "_Not captured._",
            "",
            "## Evidence",
            "",
        ]
    )

    evidence: list[str] = []
    if bug.repro_script_path:
        evidence.append(
            f"- `{REPRO_FILENAME}` - standalone Playwright reproduction "
            f"(`{redact_text(bug.repro_script_path)}`). Run `python {REPRO_FILENAME}`; "
            "it exits 1 while the defect is present."
        )
    else:
        evidence.append("- No reproduction script could be written for this defect.")
    if bug.screenshot_path:
        evidence.append(
            f"- `{SCREENSHOT_FILENAME}` - the frame captured at the moment of failure "
            f"(`{redact_text(bug.screenshot_path)}`)."
        )
    else:
        evidence.append("- No screenshot was captured for this failure.")
    evidence.append(f"- `{BUG_JSON_FILENAME}` - the machine-readable record of this bug.")
    if bug.directory:
        evidence.append(f"- Artifact directory: `{redact_text(bug.directory)}`")
    lines.extend(evidence)

    lines.extend(
        [
            "",
            "## Impact",
            "",
            _impact_sentence(risk=risk, severity=severity, flow_name=bug.flow_name or bug.flow_id),
            "",
            "## Labels",
            "",
            ", ".join(f"`{redact_text(str(label))}`" for label in bug.labels) or "_none_",
            "",
            "---",
            "",
            f"Auto-filed by the Autonomous Test Orchestration Agent - run `{run_id}`, "
            f"bug `{bug.bug_id}`, {bug.created_at}. The failure was classified as a genuine "
            "application defect by the Healer; no assertion was weakened to make the test pass. "
            "Confirm the reproduction before assigning.",
            "",
        ]
    )
    return "\n".join(lines)


def _impact_sentence(*, risk: str, severity: str, flow_name: str) -> str:
    """One sentence a non-engineer can act on, keyed off the risk rubric."""
    name = redact_text(flow_name or "the affected flow")
    if risk == RiskLevel.HIGH.value:
        return (
            f"**{name}** is a business-critical journey (money, authentication, or user data), "
            f"so this {severity} defect blocks real users from completing it and should be "
            "triaged before the next release."
        )
    if risk == RiskLevel.MEDIUM.value:
        return (
            f"**{name}** is a primary but non-transactional journey; this {severity} defect "
            "degrades the experience and will generate support contacts if it ships."
        )
    return (
        f"**{name}** is a low-risk journey; this {severity} defect is unlikely to block revenue "
        "but still contradicts the documented expected behaviour."
    )


# --------------------------------------------------------------------------
# Ticket content - model first, deterministic fallback
# --------------------------------------------------------------------------
@dataclass
class _TicketDraft:
    """The prose half of a bug, from whichever path produced it."""

    title: str
    description: str
    steps: list[str] = field(default_factory=list)
    expected: str = ""
    actual: str = ""
    severity: Severity = "major"
    labels: list[str] = field(default_factory=list)
    source: str = "llm"
    note: str = ""


def _coerce_severity(value: Any, default: Severity = "major") -> Severity:
    text = str(value or "").strip().lower()
    if text in ("critical", "blocker", "sev1", "s1", "highest", "urgent"):
        return "critical"
    if text in ("minor", "low", "trivial", "cosmetic", "sev4", "s4"):
        return "minor"
    if text in ("major", "high", "medium", "normal", "sev2", "sev3", "s2", "s3"):
        return "major"
    return default


def _deterministic_draft(
    *, flow: TestFlow, result: TestResult, action: HealerAction, risk: RiskLevel, reason: str
) -> _TicketDraft:
    """Build a usable ticket without the model.

    The Healer prompt already asks for a ``bug_title`` / ``bug_description``, and
    the Healer node stashes whatever it got in ``HealerAction.signals``; we use
    those when present and otherwise assemble the ticket from the flow itself.
    Either way the bug is filed - a model outage must not lose a real defect.
    """
    signals = action.signals if isinstance(action.signals, dict) else {}
    flow_label = flow.name or flow.id
    title = str(signals.get("bug_title") or "").strip() or (
        f"{flow_label}: expected outcome not observed"
    )
    observed = (
        redact_text(result.error_message or "")
        or f"the test finished with status {_enum_value(result.status) or 'error'}"
    )
    expected = flow.expected_outcome or "the flow completes as described in the plan"

    steps: list[str] = []
    for index, step in enumerate(flow.steps, start=1):
        text = step.description or f"{step.action} {step.target}".strip()
        if step.action == "goto":
            text = f"Open {sanitize_url(step.value or step.target or flow.url)}"
        steps.append(redact_text(text) or f"Step {index}")
    steps.append(f"Observe: {redact_text(expected)}")

    body = str(signals.get("bug_description") or "").strip()
    if not body:
        body = "\n".join(
            [
                "## Summary",
                "",
                f"The {flow_label} flow does not produce its expected outcome. The Healer "
                f"classified the failure as a genuine application defect with "
                f"{_clamp(action.confidence):.2f} confidence: "
                f"{redact_text(action.rationale) or 'no rationale was recorded'}.",
            ]
        )

    note = (
        "This ticket was assembled deterministically from the test plan and the captured "
        f"failure evidence because the ticket-writing model was unavailable ({reason}). "
        "The reproduction script, screenshot and evidence below are unaffected - they come "
        "from the actual run, not from a model."
    )

    return _TicketDraft(
        title=title[:120],
        description=f"{body}\n\n> **Note:** {note}",
        steps=steps,
        expected=redact_text(expected),
        actual=observed,
        severity=_SEVERITY_BY_RISK.get(risk, "major"),
        labels=["bug", "auto-filed", f"risk:{risk.value}", "deterministic-ticket"],
        source="deterministic",
        note=note,
    )


async def _draft_ticket(
    *,
    llm: LLMClient,
    flow: TestFlow,
    result: TestResult,
    action: HealerAction,
    risk: RiskLevel,
) -> _TicketDraft:
    """Ask the codegen model for the ticket prose, falling back deterministically.

    Everything sent to the model is redacted first: a prompt is an outbound
    network payload, so it is treated exactly like a file on disk. Ticket prose
    is mechanical translation, not judgment, which is why it runs on
    :attr:`~llm.client.ModelRole.CODEGEN` and never on the 70B model.
    """
    try:
        flow_payload = _as_dict(redact_secrets(flow.model_dump(mode="json")))
        flow_payload["url"] = sanitize_url(str(flow_payload.get("url") or ""))
        result_payload = _as_dict(redact_secrets(result.model_dump(mode="json")))
        result_payload.pop("dom_snippet", None)
        if result_payload.get("traceback"):
            result_payload["traceback"] = str(result_payload["traceback"])[:1500]
        result_payload["final_url"] = sanitize_url(str(result_payload.get("final_url") or ""))
        healer_payload = _as_dict(redact_secrets(action.model_dump(mode="json")))

        payload = await llm.complete_json(
            ModelRole.CODEGEN,
            [
                {"role": "system", "content": BUG_SYSTEM},
                {
                    "role": "user",
                    "content": bug_user(
                        flow=flow_payload,
                        result=result_payload,
                        healer=healer_payload,
                        risk=risk.value,
                    ),
                },
            ],
            task=f"bug_packager:{flow.id}",
        )
    except JSONParseError as exc:
        log.warning("bug ticket for flow %s was unparsable JSON: %s", flow.id, exc)
        return _deterministic_draft(
            flow=flow, result=result, action=action, risk=risk, reason="invalid JSON twice"
        )
    except Exception as exc:  # noqa: BLE001 - a model outage must never lose a defect
        log.warning("bug ticket for flow %s could not be generated: %s", flow.id, exc)
        return _deterministic_draft(
            flow=flow,
            result=result,
            action=action,
            risk=risk,
            reason=f"{type(exc).__name__}: {exc}",
        )

    data = payload
    if isinstance(data, list):
        data = next((item for item in data if isinstance(item, dict)), {})
    if not isinstance(data, dict):
        return _deterministic_draft(
            flow=flow,
            result=result,
            action=action,
            risk=risk,
            reason="the model returned a non-object payload",
        )

    fallback = _deterministic_draft(
        flow=flow, result=result, action=action, risk=risk, reason="unused"
    )
    steps = [
        redact_text(str(item.get("step") if isinstance(item, dict) else item)).strip()
        for item in coerce_list(data.get("steps_to_reproduce"), "steps_to_reproduce")
    ]
    steps = [s for s in steps if s and s.lower() != "none"]
    labels = [
        redact_text(str(label)).strip()[:40]
        for label in coerce_list(data.get("labels"), "labels")
        if str(label).strip()
    ]
    for required in ("auto-filed", f"risk:{risk.value}"):
        if required not in labels:
            labels.append(required)

    return _TicketDraft(
        title=redact_text(str(data.get("title") or fallback.title)).strip()[:120] or fallback.title,
        description=redact_text(str(data.get("description") or fallback.description)).strip()
        or fallback.description,
        steps=steps or fallback.steps,
        expected=redact_text(str(data.get("expected") or fallback.expected)).strip(),
        actual=redact_text(str(data.get("actual") or fallback.actual)).strip(),
        severity=_coerce_severity(data.get("severity"), _SEVERITY_BY_RISK.get(risk, "major")),
        labels=labels[:8],
        source="llm",
    )


def _as_dict(value: Any) -> dict[str, Any]:
    """Normalise a redacted payload back into a mutable dict for the prompt."""
    return dict(value) if isinstance(value, dict) else {}


# --------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------
def _risk_for(action: HealerAction, risk_lookup: dict[str, RiskLevel]) -> RiskLevel:
    """Risk of the affected flow: the ranking stage wins, the action back-fills."""
    candidate: Any = risk_lookup.get(action.flow_id) or action.risk
    if isinstance(candidate, RiskLevel):
        return candidate
    try:
        return RiskLevel(str(candidate).strip().lower())
    except ValueError:
        return RiskLevel.MEDIUM


def _index_results(results: Sequence[TestResult]) -> dict[str, TestResult]:
    """One result per flow, preferring the failing attempt.

    A healed flow can have several results; the bug is about the failure, so the
    first failing attempt is the evidence we keep.
    """
    chosen: dict[str, TestResult] = {}
    for item in results:
        existing = chosen.get(item.flow_id)
        if existing is None or (item.failed and not existing.failed):
            chosen[item.flow_id] = item
    return chosen


def _write_text(path: Path, text: str) -> Path | None:
    """Write a redacted UTF-8 file. Returns ``None`` if the write failed.

    Redaction happens here as well as at every call site on purpose: this is the
    last line before bytes hit the disk, and defence in depth is cheap.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(redact_text(text), encoding="utf-8")
        return path
    except OSError as exc:
        log.error("could not write %s: %s", path.name, exc)
        return None


def _copy_screenshot(source: str | None, destination: Path) -> Path | None:
    """Copy the failure screenshot into the bug directory, if one exists."""
    if not source:
        return None
    try:
        origin = Path(source)
        if not origin.is_file():
            log.info("screenshot %s is not on disk; the bug ships without one", origin.name)
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, destination)
        return destination
    except (OSError, ValueError) as exc:
        log.warning("could not copy the failure screenshot: %s", exc)
        return None


def _minimal_bug(
    *, bug_id: str, action: HealerAction, risk: RiskLevel, reason: str
) -> PackagedBug:
    """Last-resort record so a defect is never silently dropped."""
    return PackagedBug(
        bug_id=bug_id,
        flow_id=action.flow_id,
        flow_name=action.flow_name,
        title=f"{action.flow_name or action.flow_id}: confirmed defect (packaging degraded)",
        description=(
            "The Healer confirmed a genuine application defect for this flow, but the bug "
            f"artifacts could not be assembled ({redact_text(reason)}). The classification and "
            "its evidence are still recorded here so the defect is not lost."
        ),
        confidence=_clamp(action.confidence),
        risk=risk,
        severity=_SEVERITY_BY_RISK.get(risk, "major"),
        expected="",
        actual=redact_text(action.rationale or ""),
        labels=["bug", "auto-filed", f"risk:{risk.value}", "packaging-degraded"],
    )


async def _package_one(
    *,
    llm: LLMClient,
    bug_id: str,
    run_id: str,
    bugs_dir: Path,
    flow: TestFlow,
    result: TestResult,
    action: HealerAction,
    risk: RiskLevel,
    base_url: str,
    test_source: str | None,
) -> tuple[PackagedBug, list[str]]:
    """Produce one bug's model and its on-disk artifacts. Returns (bug, artifacts)."""
    draft = await _draft_ticket(llm=llm, flow=flow, result=result, action=action, risk=risk)

    severity = draft.severity
    if risk is RiskLevel.HIGH and severity == "minor":
        # A HIGH-risk flow is, by the risk rubric, money / auth / user data. A
        # defect there cannot be "minor" no matter how cosmetic it looks to a
        # model that never saw the rubric, so the risk ranking overrides it.
        log.info("bug %s: severity forced from minor to critical by HIGH flow risk", bug_id)
        severity = "critical"
        draft.description += (
            "\n\n> **Severity override:** the model rated this *minor*; the affected flow is "
            "ranked HIGH business risk, so severity was raised to *critical* by the risk rubric."
        )

    bug = PackagedBug(
        bug_id=bug_id,
        flow_id=flow.id,
        flow_name=flow.name or action.flow_name,
        title=draft.title,
        description=draft.description,
        classification=DefectClass.GENUINE_DEFECT,
        confidence=_clamp(action.confidence),
        risk=risk,
        severity=severity,  # type: ignore[arg-type]
        steps_to_reproduce=draft.steps,
        expected=draft.expected,
        actual=draft.actual,
        labels=draft.labels or ["bug", "auto-filed"],
    )

    directory = bugs_dir / bug_id
    artifacts: list[str] = []
    try:
        directory.mkdir(parents=True, exist_ok=True)
        bug.directory = str(directory)
    except OSError as exc:
        log.error("could not create the artifact directory for %s: %s", bug_id, exc)
        bug.description += (
            f"\n\n> **Note:** the artifact directory could not be created ({redact_text(str(exc))}), "
            "so this bug has no repro script or screenshot on disk."
        )
        return bug, artifacts

    repro = _write_text(
        directory / REPRO_FILENAME,
        build_repro_script(
            flow=flow,
            result=result,
            base_url=base_url,
            test_source=test_source,
            bug_id=bug_id,
        ),
    )
    if repro is not None:
        bug.repro_script_path = str(repro)
        artifacts.append("repro")

    shot = _copy_screenshot(result.screenshot_path, directory / SCREENSHOT_FILENAME)
    if shot is not None:
        bug.screenshot_path = str(shot)
        artifacts.append("screenshot")

    ticket = _write_text(directory / TICKET_FILENAME, render_ticket_markdown(bug, run_id=run_id))
    if ticket is not None:
        bug.ticket_path = str(ticket)
        artifacts.append("ticket")

    try:
        (directory / BUG_JSON_FILENAME).write_text(
            json.dumps(
                redact_secrets(bug.model_dump(mode="json")),
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        artifacts.append("bug.json")
    except (OSError, TypeError, ValueError) as exc:
        log.error("could not write %s for %s: %s", BUG_JSON_FILENAME, bug_id, exc)

    return bug, artifacts


async def package_bugs(
    *,
    llm: LLMClient,
    run_directory: Path,
    run_id: str,
    flows: Sequence[TestFlow],
    results: Sequence[TestResult],
    healer_actions: Sequence[HealerAction],
    risk_lookup: dict[str, RiskLevel] | None = None,
    tests: Sequence[GeneratedTest] | None = None,
    base_url: str = "",
    emit: EmitFn | None = None,
) -> list[PackagedBug]:
    """Package every Healer-confirmed genuine defect into its own artifact folder.

    One bug per :class:`~graph.state.HealerAction` classified
    ``GENUINE_DEFECT`` - no de-duplication, because two flows failing the same
    way is itself information a triager wants. Bugs are numbered in risk order
    so ``BUG-001`` is the one to read first, and the returned list is ordered
    the same way (high risk first, then bug id).

    Failure handling is deliberate: an LLM outage degrades to a deterministic
    ticket, a disk error degrades to a bug record with fewer artifacts, and an
    unexpected exception degrades to a minimal record. Nothing the Healer
    confirmed is ever dropped on the floor.
    """
    report: EmitFn = emit or (lambda summary, detail="", risk=None, confidence=None: None)

    defects = [
        action
        for action in healer_actions
        if _enum_value(action.classification) == DefectClass.GENUINE_DEFECT.value
    ]
    if not defects:
        log.info("no genuine defects to package")
        return []

    flows_by_id = {flow.id: flow for flow in flows}
    results_by_id = _index_results(results)
    source_by_id = {t.flow_id: t.source for t in (tests or []) if t.source}
    risks = dict(risk_lookup or {})

    bugs_dir = run_directory / BUG_DIRNAME
    try:
        bugs_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - the models are still returned
        log.error("could not create %s: %s", bugs_dir, exc)

    # Highest business risk first, stable within a level, so the id order and
    # the report order agree.
    ordered = sorted(
        enumerate(defects),
        key=lambda pair: (_risk_for(pair[1], risks).sort_key, pair[0]),
    )
    allocated: list[str] = _existing_bug_ids(bugs_dir)
    packaged: list[PackagedBug] = []

    for _, action in ordered:
        risk = _risk_for(action, risks)
        bug_id = next_bug_id(allocated)
        allocated.append(bug_id)

        flow = flows_by_id.get(action.flow_id) or TestFlow(
            id=action.flow_id or "unknown-flow",
            name=action.flow_name or action.flow_id,
            expected_outcome="",
            url=base_url,
        )
        result = results_by_id.get(action.flow_id) or TestResult(
            flow_id=action.flow_id,
            flow_name=action.flow_name,
            error_message=action.rationale or "no execution result was recorded",
        )

        try:
            bug, artifacts = await _package_one(
                llm=llm,
                bug_id=bug_id,
                run_id=run_id,
                bugs_dir=bugs_dir,
                flow=flow,
                result=result,
                action=action,
                risk=risk,
                base_url=base_url or flow.url or "",
                test_source=source_by_id.get(action.flow_id),
            )
        except Exception as exc:  # noqa: BLE001 - one bad bug must not kill the run
            log.error(
                "packaging %s for flow %s failed: %s", bug_id, action.flow_id, exc, exc_info=True
            )
            bug = _minimal_bug(bug_id=bug_id, action=action, risk=risk, reason=str(exc))
            artifacts = []

        packaged.append(bug)
        summary = (
            f"Bug packaged: {bug.bug_id} with {' + '.join(artifacts)}"
            if artifacts
            else f"Bug packaged: {bug.bug_id} (record only, artifacts unavailable)"
        )
        report(
            summary,
            f"{bug.title} | severity={bug.severity} risk={_enum_value(bug.risk)} "
            f"flow={bug.flow_name or bug.flow_id} dir={bug.directory or 'n/a'}",
            _enum_value(bug.risk),
            bug.confidence,
        )

    packaged.sort(key=lambda b: (b.risk.sort_key, b.bug_id))
    log.info("packaged %d genuine defect(s) into %s", len(packaged), bugs_dir)
    return packaged
