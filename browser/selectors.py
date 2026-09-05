"""Live selector resolution and validation.

The Planner describes targets in plain language ("the Add to Basket button")
because a plan written against imagined CSS is worthless. This module turns
that description into a Playwright locator that is **proved to exist on the
live page** before a single line of test code is written. Nothing downstream
ever sees a hallucinated selector.

How it works
------------
1. :func:`build_candidates` proposes an ordered list of locator strategies for
   the intent, biased by the step's action (a ``fill`` wants a textbox, a
   ``click`` wants a button or link) and enriched with the element inventory
   the crawler captured for that page.
2. :func:`validate_candidates` probes each candidate against the live DOM,
   recording match count and visibility.
3. The first candidate that resolves to exactly one visible element wins. If
   several match, the most specific strategy wins (test id > role+name > label
   > placeholder > text > css).
4. If nothing resolves, the step is marked ``valid=False``. The Generator then
   writes a defensive step and the Healer is told the selector was never
   resolvable in the first place - which is a very different signal from "it
   used to work".

Every helper that does not need a browser is a pure function so the sanity
tests can cover the fallback ordering and the code-emission escaping without
launching Chromium.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Sequence

from graph.state import SelectorCandidate, SelectorValidation
from logging_setup import get_logger

log = get_logger("aivor.selectors")

# Strategies ordered from most to least specific. Ties are broken by this
# ordering, and it is also the fallback order used when a candidate fails.
STRATEGY_PRIORITY: tuple[str, ...] = (
    "testid",
    "role",
    "label",
    "placeholder",
    "alt",
    "title",
    "text",
    "css",
)

# Words that carry no discriminating power in a target description.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "for", "on", "in", "at", "to", "with",
        "field", "input", "box", "textbox", "button", "btn", "link",
        "element", "control", "component", "icon", "label", "area",
        "first", "second", "third", "main", "primary", "page",
    }
)

# Action -> plausible ARIA roles, most likely first.
ACTION_ROLES: dict[str, tuple[str, ...]] = {
    "click": ("button", "link", "menuitem", "tab", "checkbox", "radio", "option"),
    "fill": ("textbox", "searchbox", "combobox", "spinbutton"),
    "select": ("combobox", "listbox"),
    "check": ("checkbox", "radio", "switch"),
    "press": ("textbox", "searchbox"),
    "wait_for": ("heading", "alert", "status", "region", "list"),
    "assert_visible": ("heading", "alert", "status", "list", "listitem", "table", "img"),
    "assert_not_visible": ("alert", "status", "listitem"),
    "assert_text": ("heading", "alert", "status", "paragraph", "cell"),
}

# Noun in the intent -> role, when the action alone is not decisive.
NOUN_ROLES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bbutton\b|\bsubmit\b|\bsign in\b|\blog ?in\b|\badd to\b", re.I), "button"),
    (re.compile(r"\blink\b|\bnavigat|\bmenu item\b", re.I), "link"),
    (re.compile(r"\bheading\b|\btitle\b|\bheader\b", re.I), "heading"),
    (re.compile(r"\bcheckbox\b|\bagree\b|\bremember me\b", re.I), "checkbox"),
    (re.compile(r"\bdropdown\b|\bselect\b|\bcombo\b", re.I), "combobox"),
    (re.compile(r"\bsearch\b", re.I), "searchbox"),
    (re.compile(r"\bpassword\b|\bemail\b|\busername\b|\binput\b|\bfield\b", re.I), "textbox"),
    (re.compile(r"\berror\b|\bwarning\b|\balert\b|\bmessage\b", re.I), "alert"),
    (re.compile(r"\blist\b|\bresults\b|\bgrid\b", re.I), "list"),
    (re.compile(r"\bimage\b|\bphoto\b|\bthumbnail\b", re.I), "img"),
)

_QUOTED_RE = re.compile(r"[\"'‘’“”]([^\"'‘’“”]{2,60})[\"'‘’“”]")


# ==========================================================================
# Pure helpers (covered by tests/test_selector_validation_helpers.py)
# ==========================================================================
def escape_py_string(value: str) -> str:
    """Escape a value for embedding inside a double-quoted Python literal."""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def extract_quoted(text: str) -> list[str]:
    """Pull out quoted phrases, which are almost always the literal label."""
    return [m.strip() for m in _QUOTED_RE.findall(text or "") if m.strip()]


def normalize_intent(text: str) -> str:
    """Reduce a target description to its discriminating words.

    ``"the 'Add to Basket' button"`` -> ``"add to basket"``.
    ``"the search input"``           -> ``"search"``.
    Returns an empty string when nothing discriminating remains.
    """
    if not text:
        return ""
    quoted = extract_quoted(text)
    if quoted:
        return quoted[0].strip().lower()
    cleaned = re.sub(r"[^\w\s-]", " ", text.lower())
    words = [w for w in cleaned.split() if w and w not in _STOPWORDS]
    return " ".join(words).strip()


def guess_role(intent: str, action: str = "click") -> str | None:
    """Best-guess ARIA role for a target, from its wording then its action."""
    for pattern, role in NOUN_ROLES:
        if pattern.search(intent or ""):
            return role
    # An unrecognised action falls back to the click vocabulary rather than to
    # nothing: the Planner's action aliasing already normalises toward "click",
    # and returning None here would leave a step with no candidate at all.
    roles = ACTION_ROLES.get(action or "click") or ACTION_ROLES["click"]
    return roles[0] if roles else None


def fallback_order(action: str) -> tuple[str, ...]:
    """Strategy order to try for a given action.

    ``fill``/``select`` lead with label and placeholder because form controls
    are usually labelled; everything else leads with role+name.
    """
    if action in ("fill", "press", "select", "check"):
        return ("testid", "label", "placeholder", "role", "css", "text", "title", "alt")
    if action in ("assert_text", "assert_visible", "assert_not_visible", "wait_for"):
        return ("testid", "role", "text", "label", "css", "alt", "title", "placeholder")
    return STRATEGY_PRIORITY


def strategy_rank(strategy: str) -> int:
    """Lower is more specific. Unknown strategies sort last."""
    try:
        return STRATEGY_PRIORITY.index(strategy)
    except ValueError:
        return len(STRATEGY_PRIORITY)


# How durable each strategy is across releases of the application under test.
# This is a statement about *maintenance cost*, not about whether the locator
# currently resolves: a CSS path resolves perfectly today and breaks the moment
# somebody reorders a div.
STRATEGY_FRAGILITY: dict[str, tuple[str, str]] = {
    "testid": (
        "stable",
        "a dedicated test attribute exists precisely so that markup can change "
        "without breaking this test",
    ),
    "role": (
        "stable",
        "accessible role and name track what the control *is*, which changes far "
        "less often than how it is styled",
    ),
    "label": (
        "stable",
        "the visible label is part of the product's contract with its users",
    ),
    "placeholder": (
        "moderate",
        "placeholder text is copy, and copy is edited without ceremony",
    ),
    "alt": ("moderate", "alt text is copy and is frequently revised"),
    "title": ("moderate", "title attributes are copy and are often removed outright"),
    "text": (
        "fragile",
        "matches on visible copy, so any wording change - including a translation "
        "or a typo fix - breaks this test",
    ),
    "css": (
        "fragile",
        "depends on document structure and class names, both of which change "
        "whenever the markup or the styling is refactored",
    ),
}

#: Class-name fragments that mark a CSS selector as machine-generated, and so
#: guaranteed to change on the next build of the target application.
_GENERATED_CLASS_RE = re.compile(
    r"(?:^|[.\s_-])(?:css|sc|jsx|styled|emotion|mui|chakra|tw)-[a-z0-9]{4,}"
    r"|[.#][a-z]*[0-9a-f]{6,}\b"
    r"|:nth-(?:child|of-type)\(",
    re.I,
)


def assess_fragility(strategy: str, expression: str, match_count: int = 1) -> dict[str, Any]:
    """Describe how likely one chosen locator is to break on the next release.

    Returned verbatim into the report so that a reader can tell a test that
    genuinely covers a flow from one that will need re-healing next sprint. The
    grade is deliberately independent of whether the locator resolved: a
    fragile locator that works today is exactly the thing worth flagging.
    """
    grade, rationale = STRATEGY_FRAGILITY.get(
        strategy, ("fragile", "unrecognised strategy; treated as fragile")
    )
    notes: list[str] = []

    if strategy == "css" and _GENERATED_CLASS_RE.search(expression or ""):
        grade = "fragile"
        notes.append(
            "the selector depends on a generated class name or positional index, "
            "which changes on every build"
        )
    if match_count > 1:
        grade = "fragile" if grade != "fragile" else grade
        notes.append(
            f"the locator matches {match_count} elements, so it is resolved by "
            "position and will silently target a different element if the order changes"
        )

    return {
        "strategy": strategy,
        "expression": expression,
        "grade": grade,
        "rationale": rationale,
        "notes": notes,
        "recommendation": (
            ""
            if grade == "stable"
            else "add a data-testid to this element to make the test durable"
        ),
    }


def fragility_summary(assessments: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-step fragility into a per-suite headline."""
    counts = {"stable": 0, "moderate": 0, "fragile": 0}
    for item in assessments:
        grade = str(item.get("grade") or "fragile")
        counts[grade] = counts.get(grade, 0) + 1
    total = sum(counts.values())
    return {
        "counts": counts,
        "total": total,
        "fragile_ratio": round(counts["fragile"] / total, 3) if total else 0.0,
        "verdict": (
            "durable"
            if total and counts["fragile"] == 0
            else "some selectors will need maintenance"
            if total
            else "no selectors resolved"
        ),
    }


def expression_for(strategy: str, value: str, *, role: str | None = None) -> str:
    """Render the Playwright Python source for one locator strategy.

    The returned string is what the generated test file will literally contain,
    so it must be valid Python with every value escaped.
    """
    safe = escape_py_string(value)
    if strategy == "role":
        role_name = escape_py_string(role or "button")
        if value:
            return f'page.get_by_role("{role_name}", name="{safe}")'
        return f'page.get_by_role("{role_name}")'
    if strategy == "testid":
        return f'page.get_by_test_id("{safe}")'
    if strategy == "label":
        return f'page.get_by_label("{safe}")'
    if strategy == "placeholder":
        return f'page.get_by_placeholder("{safe}")'
    if strategy == "alt":
        return f'page.get_by_alt_text("{safe}")'
    if strategy == "title":
        return f'page.get_by_title("{safe}")'
    if strategy == "text":
        return f'page.get_by_text("{safe}")'
    if strategy == "css":
        return f'page.locator("{safe}")'
    raise ValueError(f"unknown selector strategy: {strategy!r}")


def score_candidate(candidate: SelectorCandidate, intent: str) -> float:
    """Rank a probed candidate. Higher is better.

    Exactly one visible match is the ideal. Zero matches is disqualifying.
    Many matches is penalised because it makes the test ambiguous.
    """
    if candidate.match_count <= 0:
        return -1.0
    score = 10.0 - strategy_rank(candidate.strategy)
    if candidate.match_count == 1:
        score += 6.0
    else:
        score -= min(candidate.match_count - 1, 5) * 1.5
    if candidate.visible:
        score += 4.0
    if intent and intent.lower() in candidate.expression.lower():
        score += 1.5
    return score


def pick_best(candidates: Sequence[SelectorCandidate], intent: str) -> SelectorCandidate | None:
    """Choose the highest-scoring probed candidate, or ``None`` if all failed."""
    scored = [(score_candidate(c, intent), i, c) for i, c in enumerate(candidates)]
    scored = [entry for entry in scored if entry[0] >= 0]
    if not scored:
        return None
    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    return scored[0][2]


# ==========================================================================
# Candidate construction
# ==========================================================================
@dataclass
class Candidate:
    """A locator proposal: how to build it, and the source that recreates it."""

    strategy: str
    value: str
    role: str | None = None
    expression: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not self.expression:
            self.expression = expression_for(self.strategy, self.value, role=self.role)


def _inventory_matches(intent: str, inventory: Sequence[dict[str, Any]], keys: Sequence[str]) -> list[str]:
    """Values from the crawler's element inventory that overlap the intent.

    This is what makes resolution reliable: instead of guessing a label we look
    at the labels the page actually has and pick the closest one.
    """
    if not intent:
        return []
    tokens = {t for t in re.split(r"\W+", intent.lower()) if len(t) > 2}
    scored: list[tuple[int, str]] = []
    for item in inventory:
        for key in keys:
            raw = item.get(key)
            if not isinstance(raw, str) or not raw.strip():
                continue
            value = raw.strip()
            lowered = value.lower()
            if lowered == intent.lower():
                scored.append((100, value))
                continue
            overlap = len({t for t in re.split(r"\W+", lowered) if len(t) > 2} & tokens)
            if overlap:
                scored.append((overlap * 10 - abs(len(lowered) - len(intent)) // 20, value))
    scored.sort(key=lambda pair: -pair[0])
    seen: list[str] = []
    for _, value in scored:
        if value not in seen:
            seen.append(value)
    return seen[:3]


def build_candidates(
    *,
    intent: str,
    action: str,
    page_inventory: dict[str, Any] | None = None,
    max_candidates: int = 8,
) -> list[Candidate]:
    """Propose locators for one step target, best first.

    ``page_inventory`` is the crawler's record for the page: ``inputs``,
    ``buttons``, ``headings``, ``links``. When present, proposals are drawn
    from real page content rather than from guesswork.
    """
    normalized = normalize_intent(intent)
    role = guess_role(intent, action)
    inventory = page_inventory or {}
    inputs = inventory.get("inputs") or []
    buttons = inventory.get("buttons") or []
    headings = inventory.get("headings") or []

    candidates: list[Candidate] = []

    def add(strategy: str, value: str, *, role_override: str | None = None, note: str = "") -> None:
        # A role-only candidate ("any button on this page") is legitimate and is
        # the last-resort proposal when the target text gave us nothing to match
        # on. Every other strategy needs a value to be meaningful.
        if not value and strategy != "role":
            return
        try:
            proposal = Candidate(strategy=strategy, value=value, role=role_override or role, note=note)
        except ValueError:
            return
        if all(c.expression != proposal.expression for c in candidates):
            candidates.append(proposal)

    # 1. Anything explicitly quoted is the strongest signal we have.
    for quoted in extract_quoted(intent):
        add("role", quoted, note="quoted literal from the plan")
        add("text", quoted, note="quoted literal from the plan")
        add("label", quoted, note="quoted literal from the plan")

    # 2. Draw from what the crawler actually saw on the page.
    if action in ("fill", "press", "select", "check"):
        for value in _inventory_matches(normalized, inputs, ("test_id",)):
            add("testid", value, note="matched a data-testid on the page")
        for value in _inventory_matches(normalized, inputs, ("label",)):
            add("label", value, note="matched a real form label")
        for value in _inventory_matches(normalized, inputs, ("placeholder",)):
            add("placeholder", value, note="matched a real placeholder")
        for value in _inventory_matches(normalized, inputs, ("name", "id")):
            add("css", f"[name='{value}']" if value else "", note="matched the input name attribute")
    else:
        for value in _inventory_matches(normalized, buttons, ("test_id",)):
            add("testid", value, note="matched a data-testid on the page")
        for value in _inventory_matches(normalized, buttons, ("text", "aria_label", "value")):
            add("role", value, note="matched a real button/link accessible name")
        for value in _inventory_matches(normalized, headings, ("text",)):
            add("role", value, role_override="heading", note="matched a real heading")

    # 3. Generic strategies from the normalized wording.
    if normalized:
        add("testid", normalized.replace(" ", "-"))
        add("role", normalized)
        add("label", normalized)
        add("placeholder", normalized)
        add("text", normalized)
        add("alt", normalized)
        add("title", normalized)

    # 4. Well-known structural fallbacks for the common form fields.
    lowered = (intent or "").lower()
    if "password" in lowered:
        add("css", "input[type='password']", note="conventional password input")
    if "email" in lowered:
        add("css", "input[type='email']", note="conventional email input")
    if "search" in lowered:
        add("css", "input[type='search'], input[name*='search' i], input[id*='search' i]",
            note="conventional search input")
    if any(word in lowered for word in ("username", "user name", "login", "user id")):
        add("css", "input[name*='user' i], input[id*='user' i], input[type='text']",
            note="conventional username input")
    if any(word in lowered for word in ("submit", "sign in", "log in", "login", "continue")):
        add("css", "button[type='submit'], input[type='submit']", note="conventional submit control")

    # 5. Last resort: the role alone, if we could guess one.
    if role:
        add("role", "", note="role only, no name filter")

    order = fallback_order(action)
    candidates.sort(key=lambda c: order.index(c.strategy) if c.strategy in order else len(order))
    return candidates[:max_candidates]


# ==========================================================================
# Live probing
# ==========================================================================
def _locator_from(page: Any, candidate: Candidate) -> Any:
    """Build a real Playwright locator matching ``candidate.expression``."""
    strategy, value, role = candidate.strategy, candidate.value, candidate.role
    if strategy == "role":
        return page.get_by_role(role or "button", name=value) if value else page.get_by_role(role or "button")
    if strategy == "testid":
        return page.get_by_test_id(value)
    if strategy == "label":
        return page.get_by_label(value)
    if strategy == "placeholder":
        return page.get_by_placeholder(value)
    if strategy == "alt":
        return page.get_by_alt_text(value)
    if strategy == "title":
        return page.get_by_title(value)
    if strategy == "text":
        return page.get_by_text(value)
    if strategy == "css":
        return page.locator(value)
    raise ValueError(f"unknown strategy {strategy!r}")


async def probe_candidate(page: Any, candidate: Candidate, timeout_ms: int = 1500) -> SelectorCandidate:
    """Count and visibility-check one candidate against the live page."""
    probed = SelectorCandidate(
        strategy=candidate.strategy,  # type: ignore[arg-type]
        expression=candidate.expression,
        note=candidate.note,
    )
    try:
        locator = _locator_from(page, candidate)
        probed.match_count = await locator.count()
        if probed.match_count > 0:
            try:
                probed.visible = await locator.first.is_visible(timeout=timeout_ms)
            except Exception:
                probed.visible = False
    except Exception as exc:  # invalid selector syntax, detached frame, ...
        probed.match_count = 0
        probed.note = f"{probed.note} | probe error: {type(exc).__name__}".strip(" |")
    return probed


async def resolve_target(
    page: Any,
    *,
    step_index: int,
    intent: str,
    action: str,
    page_inventory: dict[str, Any] | None = None,
    max_candidates: int = 8,
) -> SelectorValidation:
    """Resolve one plain-language target against the live DOM.

    Returns a :class:`SelectorValidation` describing every candidate tried and
    which one won, so the report and the Healer can see the reasoning rather
    than just the answer.
    """
    validation = SelectorValidation(step_index=step_index, intent=intent)
    try:
        validation.page_url = page.url
    except Exception:  # pragma: no cover - closed page
        validation.page_url = ""

    if action in ("goto", "assert_url", "screenshot", "wait_for") and not intent:
        validation.valid = True
        validation.note = "no element target required for this action"
        return validation

    proposals = build_candidates(
        intent=intent,
        action=action,
        page_inventory=page_inventory,
        max_candidates=max_candidates,
    )
    if not proposals:
        validation.note = "no candidate strategy could be derived from the target text"
        return validation

    for proposal in proposals:
        probed = await probe_candidate(page, proposal)
        validation.candidates.append(probed)
        # Early exit on a perfect hit: exactly one visible element.
        if probed.match_count == 1 and probed.visible:
            break

    best = pick_best(validation.candidates, normalize_intent(intent))
    if best is None:
        validation.valid = False
        validation.note = (
            f"none of {len(validation.candidates)} candidate locators matched an "
            "element on the live page"
        )
        return validation

    validation.chosen = best.expression
    validation.chosen_strategy = best.strategy
    validation.valid = True
    if best.match_count > 1:
        validation.chosen = f"{best.expression}.first"
        validation.note = (
            f"{best.match_count} elements matched; pinned to .first "
            "(ambiguous target in the plan)"
        )
    elif not best.visible:
        validation.note = "matched exactly one element, but it is not currently visible"
    return validation


async def reprobe_expression(page: Any, expression: str) -> dict[str, Any]:
    """Re-check a locator *source expression* against the current page.

    Used by the Healer: "is the selector this test used still in the DOM?" is
    the single strongest signal for SCRIPT_ISSUE vs GENUINE_DEFECT.
    """
    result: dict[str, Any] = {
        "expression": expression,
        "present": False,
        "count": 0,
        "visible": False,
        "error": None,
    }
    parsed = parse_expression(expression)
    if parsed is None:
        result["error"] = "could not parse the locator expression"
        return result
    try:
        locator = _locator_from(page, parsed)
        result["count"] = await locator.count()
        result["present"] = result["count"] > 0
        if result["present"]:
            try:
                result["visible"] = await locator.first.is_visible(timeout=1500)
            except Exception:
                result["visible"] = False
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


_EXPR_RE = re.compile(
    r"page\.(?P<method>get_by_role|get_by_test_id|get_by_label|get_by_placeholder|"
    r"get_by_alt_text|get_by_title|get_by_text|locator)\((?P<args>.*)\)\s*(?:\.first)?\s*$",
    re.DOTALL,
)
_METHOD_TO_STRATEGY = {
    "get_by_role": "role",
    "get_by_test_id": "testid",
    "get_by_label": "label",
    "get_by_placeholder": "placeholder",
    "get_by_alt_text": "alt",
    "get_by_title": "title",
    "get_by_text": "text",
    "locator": "css",
}


def parse_expression(expression: str) -> Candidate | None:
    """Parse a locator source string back into a :class:`Candidate`.

    Only the shapes this module emits are supported; anything else returns
    ``None`` and the caller falls back to treating it as opaque.
    """
    if not expression:
        return None
    match = _EXPR_RE.match(expression.strip())
    if not match:
        return None
    strategy = _METHOD_TO_STRATEGY[match.group("method")]
    args = match.group("args")
    literals = re.findall(r'"((?:[^"\\]|\\.)*)"', args)
    literals = [lit.replace('\\"', '"').replace("\\\\", "\\") for lit in literals]
    if not literals:
        return None
    if strategy == "role":
        role = literals[0]
        name = literals[1] if len(literals) > 1 else ""
        return Candidate(strategy="role", value=name, role=role, expression=expression)
    return Candidate(strategy=strategy, value=literals[0], expression=expression)


async def collect_inventory(page: Any, max_items: int = 60) -> dict[str, Any]:
    """Snapshot the interactive surface of the current page.

    Runs one ``evaluate`` rather than dozens of round trips. The shape returned
    here is what :func:`build_candidates` consumes and what the Planner sees.
    """
    script = """
    (max) => {
      const vis = (el) => {
        const r = el.getBoundingClientRect();
        const s = getComputedStyle(el);
        return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
      };
      const label = (el) => {
        if (el.getAttribute('aria-label')) return el.getAttribute('aria-label');
        if (el.id) {
          const l = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
          if (l) return (l.innerText || '').trim();
        }
        const wrap = el.closest('label');
        if (wrap) return (wrap.innerText || '').trim();
        return '';
      };
      const inputs = [];
      for (const el of document.querySelectorAll('input, textarea, select')) {
        if (inputs.length >= max) break;
        if ((el.type || '').toLowerCase() === 'hidden') continue;
        inputs.push({
          tag: el.tagName.toLowerCase(),
          type: (el.type || '').toLowerCase(),
          name: el.name || '',
          id: el.id || '',
          placeholder: el.placeholder || '',
          label: label(el),
          test_id: el.getAttribute('data-testid') || el.getAttribute('data-test-id') || '',
          required: !!el.required,
          visible: vis(el),
        });
      }
      const buttons = [];
      for (const el of document.querySelectorAll(
             'button, a[href], [role="button"], input[type="submit"], input[type="button"]')) {
        if (buttons.length >= max) break;
        const text = (el.innerText || el.value || '').trim().slice(0, 80);
        if (!text && !el.getAttribute('aria-label')) continue;
        buttons.push({
          tag: el.tagName.toLowerCase(),
          text: text,
          aria_label: el.getAttribute('aria-label') || '',
          value: el.value || '',
          href: el.getAttribute('href') || '',
          test_id: el.getAttribute('data-testid') || el.getAttribute('data-test-id') || '',
          visible: vis(el),
        });
      }
      const headings = [];
      for (const el of document.querySelectorAll('h1, h2, h3, [role="heading"]')) {
        if (headings.length >= 20) break;
        const text = (el.innerText || '').trim().slice(0, 120);
        if (text) headings.push({ tag: el.tagName.toLowerCase(), text });
      }
      const forms = [];
      for (const el of document.querySelectorAll('form')) {
        if (forms.length >= 10) break;
        forms.push({
          action: el.getAttribute('action') || '',
          method: (el.getAttribute('method') || 'get').toLowerCase(),
          id: el.id || '',
          field_count: el.querySelectorAll('input, textarea, select').length,
          has_password: !!el.querySelector('input[type="password"]'),
        });
      }
      return { inputs, buttons, headings, forms };
    }
    """
    try:
        data = await page.evaluate(script, max_items)
    except Exception as exc:
        log.debug("inventory collection failed: %s", exc)
        return {"inputs": [], "buttons": [], "headings": [], "forms": []}
    return data if isinstance(data, dict) else {"inputs": [], "buttons": [], "headings": [], "forms": []}
