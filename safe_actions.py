"""Safe mode: refusing to perform irreversible actions against a live application.

The agent clicks buttons it discovered seconds earlier, on an application it has
never seen, with no human in the loop, and it does so *twice* - once while
exploring and again when the generated test runs. On a real staging site that
makes "Delete account", "Place order" and "Send invoice" live ammunition.

Safe mode is on by default. It is a classifier plus a gate:

* :func:`classify_action` turns the text an operator or the LLM associated with
  a step ("click the Place Order button") into a :class:`DestructiveCategory`.
* :func:`evaluate_action` applies the run's :class:`SafetyPolicy` and returns an
  :class:`ActionDecision` that is either allowed, or blocked with a reason code.

Blocked, not silently skipped
-----------------------------
A blocked action produces an explicit ``action-blocked`` decision that flows
into the report and the generated test as a raised
:class:`DestructiveActionBlocked`. A test that quietly omitted its checkout step
and then passed would be worse than no test: it would report coverage of a flow
nobody exercised.

Authorising a category
----------------------
An operator who genuinely wants checkout coverage on a throwaway environment
sets ``SAFE_MODE=false`` (everything permitted) or names the specific categories
in ``AUTHORIZED_DESTRUCTIVE_ACTIONS`` (e.g. ``checkout,payment``). Naming
categories is preferred: it keeps ``delete`` and ``account_cancellation``
blocked while the flow under test proceeds.

Idempotency markers
-------------------
Values the agent types into forms are tagged with a run-scoped marker (see
:func:`data_marker`) so that records it does create are identifiable and
cleanable afterwards, and so a human looking at the target can tell agent
traffic from real user traffic.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final

from logging_setup import get_logger

log = get_logger("aivor.safety")


class DestructiveCategory(str, Enum):
    """The kinds of irreversible action safe mode knows how to recognise."""

    PAYMENT = "payment"
    CHECKOUT = "checkout"
    DELETE = "delete"
    ACCOUNT_CANCELLATION = "account_cancellation"
    PASSWORD_RESET = "password_reset"
    EMAIL_SEND = "email_send"
    IRREVERSIBLE_SUBMIT = "irreversible_submit"


#: Human-readable rationale per category, used in the blocked-action message.
CATEGORY_REASON: Final[dict[DestructiveCategory, str]] = {
    DestructiveCategory.PAYMENT: "it would attempt a real payment or card authorisation",
    DestructiveCategory.CHECKOUT: "it would place a real order",
    DestructiveCategory.DELETE: "it would destroy data that cannot be restored",
    DestructiveCategory.ACCOUNT_CANCELLATION: "it would close or deactivate an account",
    DestructiveCategory.PASSWORD_RESET: "it would invalidate a real credential",
    DestructiveCategory.EMAIL_SEND: "it would send mail to a real recipient",
    DestructiveCategory.IRREVERSIBLE_SUBMIT: "it submits an action that cannot be undone",
}

# Ordered most- to least-specific. The first pattern that matches wins, so
# "cancel subscription" classifies as account cancellation rather than as a
# generic irreversible submit.
_PATTERNS: Final[tuple[tuple[DestructiveCategory, re.Pattern[str]], ...]] = (
    (
        DestructiveCategory.PAYMENT,
        re.compile(
            r"\b(pay\s*now|make\s+(a\s+)?payment|confirm\s+payment|authori[sz]e\s+payment"
            r"|card\s+number|credit\s+card|cvv|cvc|billing\s+address|charge\s+(my|the)\s+card"
            r"|add\s+(a\s+)?(payment|card)|paypal|stripe|checkout\s+and\s+pay)\b",
            re.I,
        ),
    ),
    (
        DestructiveCategory.CHECKOUT,
        re.compile(
            r"\b(place\s+(your\s+|the\s+)?order|complete\s+(the\s+)?(order|purchase)"
            r"|confirm\s+(the\s+)?(order|purchase|booking)|buy\s+now|purchase\s+now"
            r"|submit\s+order|proceed\s+to\s+checkout|checkout)\b",
            re.I,
        ),
    ),
    (
        DestructiveCategory.ACCOUNT_CANCELLATION,
        re.compile(
            r"\b(close\s+(my\s+|the\s+)?account|delete\s+(my\s+|the\s+)?account"
            r"|deactivate\s+(my\s+|the\s+)?account|cancel\s+(my\s+|the\s+)?"
            r"(subscription|membership|plan|account)|terminate\s+(my\s+)?account"
            r"|unsubscribe)\b",
            re.I,
        ),
    ),
    (
        DestructiveCategory.PASSWORD_RESET,
        re.compile(
            r"\b(reset\s+(my\s+|the\s+)?password|forgot\s+(my\s+)?password"
            r"|change\s+(my\s+|the\s+)?password|send\s+(a\s+)?reset\s+link"
            r"|revoke\s+(the\s+)?(token|key|session)|rotate\s+(the\s+)?(key|secret))\b",
            re.I,
        ),
    ),
    (
        DestructiveCategory.EMAIL_SEND,
        re.compile(
            r"\b(send\s+(the\s+|an?\s+)?(email|e-mail|message|invite|invitation|invoice"
            r"|newsletter|notification)|email\s+(the\s+)?(customer|user|invoice)"
            r"|invite\s+(a\s+)?(user|member|colleague))\b",
            re.I,
        ),
    ),
    (
        DestructiveCategory.DELETE,
        re.compile(
            r"\b(delete|remove\s+permanently|permanently\s+remove|destroy|erase|purge"
            r"|drop\s+(the\s+)?(table|database)|empty\s+(the\s+)?(trash|bin)"
            r"|clear\s+all\s+data|wipe)\b",
            re.I,
        ),
    ),
    (
        DestructiveCategory.IRREVERSIBLE_SUBMIT,
        re.compile(
            r"\b(confirm\s+and\s+submit|submit\s+(the\s+)?(application|claim|return|refund"
            r"|request|report)|publish|deploy|transfer\s+funds|withdraw|send\s+money"
            r"|approve|reject\s+permanently|finali[sz]e)\b",
            re.I,
        ),
    ),
)

# Phrases that look destructive but are not, checked before the patterns above.
# "Add to cart" contains no destructive verb, but "remove from cart" does, and a
# cart is a scratch surface that is safe (and useful) to exercise.
_BENIGN: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bremove\s+(the\s+\w+\s+)?from\s+(the\s+)?(cart|basket|bag|wishlist)\b", re.I),
    re.compile(r"\bdelete\s+(the\s+)?(search|filter|query|draft\s+text)\b", re.I),
    re.compile(r"\bclear\s+(the\s+)?(search|filter|form|input|field)\b", re.I),
    re.compile(r"\bcancel\s+(the\s+)?(dialog|modal|edit|search|filter)\b", re.I),
)


@dataclass(frozen=True)
class SafetyPolicy:
    """The destructive-action rules in force for one run."""

    safe_mode: bool = True
    authorized: frozenset[DestructiveCategory] = frozenset()

    @classmethod
    def from_settings(cls, settings: object) -> SafetyPolicy:
        raw = getattr(settings, "authorized_destructive_actions", ()) or ()
        return cls(
            safe_mode=bool(getattr(settings, "safe_mode", True)),
            authorized=parse_categories(raw),
        )

    def permits(self, category: DestructiveCategory) -> bool:
        return not self.safe_mode or category in self.authorized


def parse_categories(raw: Iterable[str] | str) -> frozenset[DestructiveCategory]:
    """Parse operator-supplied category names, ignoring unknown entries.

    Unknown names are logged rather than raising: a typo in an environment
    variable should not take the service down, and the effect of ignoring it is
    to stay *more* restrictive, which is the safe direction.
    """
    if isinstance(raw, str):
        items = [chunk.strip() for chunk in raw.replace(";", ",").split(",")]
    else:
        items = [str(chunk).strip() for chunk in raw]
    out: set[DestructiveCategory] = set()
    for item in items:
        if not item:
            continue
        normalized = item.lower().replace("-", "_").replace(" ", "_")
        try:
            out.add(DestructiveCategory(normalized))
        except ValueError:
            log.warning(
                "ignoring unknown destructive-action category %r; valid values are: %s",
                item,
                ", ".join(sorted(c.value for c in DestructiveCategory)),
            )
    return frozenset(out)


@dataclass(frozen=True)
class ActionDecision:
    """Whether one action may be performed, and why not when it may not."""

    allowed: bool
    category: DestructiveCategory | None = None
    reason: str = ""
    detail: str = ""
    matched_text: str = ""

    @property
    def blocked(self) -> bool:
        return not self.allowed

    def audit(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "category": self.category.value if self.category else None,
            "reason": self.reason,
            "detail": self.detail,
            "matched_text": self.matched_text[:120],
        }


class DestructiveActionBlocked(RuntimeError):
    """Raised at execution time when safe mode refuses an action.

    Generated tests import this so a blocked step fails loudly with the reason
    attached, instead of being dropped from the test body.
    """

    def __init__(self, decision: ActionDecision) -> None:
        super().__init__(decision.detail)
        self.decision = decision


def classify_action(*texts: str | None) -> tuple[DestructiveCategory | None, str]:
    """Classify the combined text of an action.

    All fragments describing one step - its description, its target, and any
    value being typed - are considered together, because the destructive verb
    may live in any of them ("click", "the Place Order button").

    Returns the category and the substring that triggered it, or
    ``(None, "")`` when the action is not destructive.
    """
    blob = " ".join(t for t in texts if t).strip()
    if not blob:
        return None, ""
    for benign in _BENIGN:
        if benign.search(blob):
            return None, ""
    for category, pattern in _PATTERNS:
        match = pattern.search(blob)
        if match:
            return category, match.group(0)
    return None, ""


def evaluate_action(
    policy: SafetyPolicy,
    *texts: str | None,
) -> ActionDecision:
    """Apply ``policy`` to the action described by ``texts``."""
    category, matched = classify_action(*texts)
    if category is None:
        return ActionDecision(allowed=True, reason="not-destructive")
    if policy.permits(category):
        return ActionDecision(
            allowed=True,
            category=category,
            reason="authorized" if policy.safe_mode else "safe-mode-disabled",
            detail=(
                f"{category.value} action explicitly authorised by the operator"
                if policy.safe_mode
                else f"{category.value} action permitted because SAFE_MODE=false"
            ),
            matched_text=matched,
        )
    return ActionDecision(
        allowed=False,
        category=category,
        reason="destructive-action-blocked",
        detail=(
            f"safe mode blocked a {category.value} action ({matched!r}) because "
            f"{CATEGORY_REASON[category]}. Authorise it with "
            f"AUTHORIZED_DESTRUCTIVE_ACTIONS={category.value} on a throwaway "
            "environment, or set SAFE_MODE=false to permit every category."
        ),
        matched_text=matched,
    )


# --------------------------------------------------------------------------
# Exploration guard
# --------------------------------------------------------------------------
def is_safe_to_explore(policy: SafetyPolicy, text: str, href: str = "") -> bool:
    """Whether the crawler may click ``text`` while mapping the application.

    Exploration is held to the same bar as execution. This is the rule that
    stops the *discovery* pass from placing an order that the generated test
    then places a second time.
    """
    decision = evaluate_action(policy, text, href)
    return decision.allowed


# --------------------------------------------------------------------------
# Test-data markers
# --------------------------------------------------------------------------
MARKER_PREFIX: Final[str] = "atoa"
"""Short, stable tag identifying data this agent created.

Deliberately not a word an application is likely to use, and deliberately
lowercase alphanumeric so it survives fields that reject punctuation.
"""


def data_marker(run_id: str) -> str:
    """The marker embedded in values this run types into a target application.

    Keeping the run id in the marker means a human (or a cleanup job) looking at
    a stray record can trace it to the exact run that produced it.
    """
    suffix = re.sub(r"[^a-z0-9]", "", (run_id or "").lower())[-10:] or "unknown"
    return f"{MARKER_PREFIX}{suffix}"


def mark_value(value: str, run_id: str, *, field_type: str = "text") -> str:
    """Tag a value the agent is about to type so the record it creates is traceable.

    Email and telephone fields have formats that a bare marker would break, so
    they get format-preserving treatment. Anything the marker would corrupt -
    a number, a date, a password - is returned untouched.
    """
    marker = data_marker(run_id)
    kind = (field_type or "text").lower()
    if kind in ("number", "date", "datetime-local", "time", "month", "week", "range", "color"):
        return value
    if kind == "password":
        return value
    if kind == "email" or "@" in (value or ""):
        local, _, domain = (value or "").partition("@")
        local = local or "qa"
        domain = domain or "example.invalid"
        return f"{local}+{marker}@{domain}"
    if kind == "tel":
        return value
    if not value:
        return marker
    return f"{value} {marker}"


def is_marked(value: str) -> bool:
    """True when ``value`` carries this agent's test-data marker."""
    return bool(value) and MARKER_PREFIX in value.lower()


def describe_policy(policy: SafetyPolicy) -> dict[str, object]:
    """Report-safe description of the safety posture for one run."""
    return {
        "safe_mode": policy.safe_mode,
        "authorized_destructive_actions": sorted(c.value for c in policy.authorized),
        "blocked_by_default": sorted(
            c.value for c in DestructiveCategory if not policy.permits(c)
        ),
    }


def summarise_blocks(decisions: Sequence[ActionDecision]) -> dict[str, object]:
    """Aggregate blocked actions for the final report."""
    blocked = [d for d in decisions if d.blocked]
    by_category: dict[str, int] = {}
    for decision in blocked:
        if decision.category is not None:
            key = decision.category.value
            by_category[key] = by_category.get(key, 0) + 1
    return {
        "blocked_count": len(blocked),
        "by_category": by_category,
        "reasons": [d.detail for d in blocked[:20]],
    }
