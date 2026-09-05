"""A thin library of canonical test-flow patterns.

The Planner is good at describing *this* application and weaker at remembering
the flows every application should be tested for. This module encodes eight
patterns that recur across almost every web app, matches them against what the
crawler actually found, and passes the survivors to the Planner as *hints* -
never as mandates. The coverage gate can then name a pattern that was triggered
by the site but has no corresponding flow.

Honest scope
------------
This is the thin version, and it is deliberately labelled as such in the README
rather than dressed up. It is a hand-curated, regex-triggered list. It does not
learn, it is not domain-aware, and it does not condition code generation. See
:data:`ROADMAP` for what a real pattern library would add.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from logging_setup import get_logger

log = get_logger("aivor.patterns")


@dataclass(frozen=True)
class FlowPattern:
    """One canonical flow shape, with the evidence that should trigger it."""

    key: str
    name: str
    category: str
    applies_when: str
    trigger_patterns: tuple[str, ...]
    suggested_steps: tuple[str, ...]
    expected_outcome: str
    risk_hint: str

    def matches(self, blob: str) -> bool:
        return any(re.search(pattern, blob, re.I) for pattern in self.trigger_patterns)

    def as_hint(self) -> dict[str, Any]:
        return {
            "pattern": self.key,
            "name": self.name,
            "category": self.category,
            "why_it_applies": self.applies_when,
            "suggested_steps": list(self.suggested_steps),
            "expected_outcome": self.expected_outcome,
            "risk_hint": self.risk_hint,
        }


PATTERNS: tuple[FlowPattern, ...] = (
    FlowPattern(
        key="auth_happy_path",
        name="Sign in with valid credentials",
        category="happy_path",
        applies_when="the crawl found a password field",
        trigger_patterns=(r"type=[\"']?password", r"\bsign ?in\b", r"\blog ?in\b"),
        suggested_steps=(
            "open the login page",
            "fill the username field from the injected secret",
            "fill the password field from the injected secret",
            "submit the form",
            "assert that a signed-in affordance such as a sign-out link is visible",
        ),
        expected_outcome="the session is established and a sign-out affordance appears",
        risk_hint="high",
    ),
    FlowPattern(
        key="auth_invalid_credentials",
        name="Reject invalid credentials",
        category="error_state",
        applies_when="the crawl found a password field",
        trigger_patterns=(r"type=[\"']?password", r"\bsign ?in\b", r"\blog ?in\b"),
        suggested_steps=(
            "open the login page",
            "fill the username field with an obviously fake address",
            "fill the password field with an obviously wrong value",
            "submit the form",
            "assert the visible error message and that the user is still signed out",
        ),
        expected_outcome="an inline error is shown and no session is created",
        risk_hint="high",
    ),
    FlowPattern(
        key="search_no_results",
        name="Search for something that cannot exist",
        category="edge_case",
        applies_when="the crawl found a search input",
        trigger_patterns=(r"\bsearch\b", r"name=[\"']?q[\"']?", r"type=[\"']?search"),
        suggested_steps=(
            "open the page carrying the search control",
            "type a nonsense string that cannot match anything",
            "submit the search",
            "assert the explicit empty-state message rather than an empty page",
        ),
        expected_outcome="a no-results message is displayed and the page does not error",
        risk_hint="medium",
    ),
    FlowPattern(
        key="empty_required_field",
        name="Submit a form with a required field left empty",
        category="edge_case",
        applies_when="the crawl found a form with a required input",
        trigger_patterns=(r"\brequired\b", r"<form", r"\bsubmit\b"),
        suggested_steps=(
            "open the page containing the form",
            "leave the required field empty",
            "submit the form",
            "assert the validation message and that nothing was persisted",
        ),
        expected_outcome="submission is blocked and a specific validation message appears",
        risk_hint="medium",
    ),
    FlowPattern(
        key="not_found_404",
        name="Request a URL that does not exist",
        category="error_state",
        applies_when="always applicable to any HTTP application",
        trigger_patterns=(r"https?://",),
        suggested_steps=(
            "navigate to a path under the target origin that cannot exist",
            "assert that a not-found page renders with recognisable copy",
            "assert that the page is not a blank screen or an unhandled stack trace",
        ),
        expected_outcome="a handled not-found page is rendered",
        risk_hint="low",
    ),
    FlowPattern(
        key="add_to_cart",
        name="Add an item to the cart and verify it persists",
        category="happy_path",
        applies_when="the crawl found cart or basket affordances",
        trigger_patterns=(r"\badd to (cart|basket|bag)\b", r"\b(cart|basket)\b", r"\bquantity\b"),
        suggested_steps=(
            "open a product or item detail page",
            "add the item to the cart",
            "navigate to the cart",
            "assert the item title and the quantity are both present",
        ),
        expected_outcome="the cart shows the added item with the correct quantity and total",
        risk_hint="high",
    ),
    FlowPattern(
        key="checkout_abandon",
        name="Start checkout and cancel it",
        category="error_state",
        applies_when="the crawl found a checkout surface",
        trigger_patterns=(r"\bcheckout\b", r"\bplace (your )?order\b", r"\bpayment\b"),
        suggested_steps=(
            "put the cart into a state where checkout is reachable",
            "begin checkout",
            "cancel or navigate away before completing",
            "assert the cart contents are unchanged and no order was created",
        ),
        expected_outcome="cancelling leaves the cart intact and creates no order",
        risk_hint="high",
    ),
    FlowPattern(
        key="pagination_boundary",
        name="Walk pagination to its boundary",
        category="edge_case",
        applies_when="the crawl found pagination controls",
        trigger_patterns=(r"\bnext\b", r"\bpage \d+\b", r"\bpaginat", r"\bpage-\d+"),
        suggested_steps=(
            "open the first page of a paginated listing",
            "advance to the next page",
            "assert the listing content changed and the page indicator advanced",
            "assert the previous control is now enabled",
        ),
        expected_outcome="pagination advances and the boundary controls are correct",
        risk_hint="medium",
    ),
)

PATTERNS_BY_KEY: dict[str, FlowPattern] = {p.key: p for p in PATTERNS}


def match_patterns(site_blob: str) -> list[FlowPattern]:
    """Patterns whose triggers appear in the crawled site text."""
    if not site_blob:
        return []
    return [pattern for pattern in PATTERNS if pattern.matches(site_blob)]


def hints_for_prompt(site_blob: str, limit: int = 6) -> list[dict[str, Any]]:
    """Compact hint dicts for the Planner prompt.

    These are suggestions. The Planner is free to ignore any of them - the
    application may genuinely not have the surface a regex thought it saw - and
    the prompt says so.
    """
    matched = match_patterns(site_blob)
    # Lead with the risky ones: those are the patterns worth spending a flow on.
    order = {"high": 0, "medium": 1, "low": 2}
    matched.sort(key=lambda p: order.get(p.risk_hint, 1))
    return [pattern.as_hint() for pattern in matched[:limit]]


def missing_pattern_keys(
    plan_flow_names: Sequence[str],
    matched: Sequence[FlowPattern],
) -> list[str]:
    """Triggered patterns with no plausibly-matching flow in the plan.

    Matching is deliberately loose - a name-token overlap - because the point
    is to surface an obvious omission for the coverage gate, not to police the
    Planner's wording.
    """
    plan_tokens = [
        {token for token in re.split(r"\W+", (name or "").lower()) if len(token) > 3}
        for name in plan_flow_names
    ]
    missing: list[str] = []
    for pattern in matched:
        wanted = {
            token
            for token in re.split(r"\W+", f"{pattern.name} {pattern.key}".lower())
            if len(token) > 3
        }
        if not any(len(wanted & tokens) >= 2 for tokens in plan_tokens):
            missing.append(pattern.key)
    return missing


ROADMAP = """\
Pattern library - roadmap
=========================

Implemented today (thin version)
--------------------------------
* Eight hand-curated canonical flow patterns with regex triggers.
* Matching against the crawled site text, ordered by risk hint.
* Hints injected into the Planner prompt as suggestions, not mandates.
* A gap check that names triggered patterns absent from the plan.

Not implemented, and deliberately not faked
-------------------------------------------
* Learned patterns. A real library would mine completed runs for flow shapes
  that repeatedly found defects and promote them into the library automatically,
  weighted by hit rate per application domain.
* Domain packs. Separate curated sets for e-commerce, SaaS admin consoles,
  banking, and CMS-style content sites, selected by a classifier over the crawl
  rather than by keyword.
* Pattern-conditioned generation. Today a pattern only nudges the Planner. A
  full version would carry a parameterised code template per pattern so the
  Generator emits a known-good skeleton and only fills in the resolved
  locators, which would cut codegen failures substantially.
* Negative-pattern mining. Recording which patterns produced flaky or
  low-value tests on a given target so they stop being suggested for it.
* Cross-run pattern coverage reporting: "this target has never been tested for
  session expiry" as a first-class report line.
"""
