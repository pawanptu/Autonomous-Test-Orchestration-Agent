"""Confidence scoring and the auto-apply threshold branch.

The Healer's confidence is not the model's number alone. It is the model's
number *blended with a deterministic score* computed from evidence this process
gathered itself: was the locator still in the DOM when we re-probed it, was the
failure a timeout or an assertion mismatch, did the same failure reproduce, is
there a captcha on screen.

Why blend rather than trust the model
-------------------------------------
A language model's self-reported confidence is poorly calibrated and easy to
talk up. The deterministic half is grounded in facts, and averaging the two
caps how far a confident-sounding rationale can move the decision. The
deterministic half is also unit-testable, which is what makes the 0.6 threshold
branch something we can prove rather than something we hope for.

The threshold branch is a genuinely different code path, not a log level:
  >= 0.6 and SCRIPT_ISSUE -> patch applied, test re-run once
  <  0.6                  -> patch NOT applied, queued for human review with
                             its evidence, surfaced in the UI and the report
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from config import CONFIDENCE_AUTO_APPLY_THRESHOLD
from graph.state import DefectClass

BASE_CONFIDENCE = 0.5

# Deterministic adjustments, straight from the rubric in llm/prompts.py.
ADJUSTMENTS: dict[str, tuple[float, str]] = {
    "selector_present_in_dom": (
        +0.18,
        "the original locator still resolves in the captured DOM, so the "
        "element exists and the test's expectation is the suspect part",
    ),
    "selector_absent_from_dom": (
        -0.05,
        "the original locator no longer resolves; this is consistent with both "
        "a renamed element and a genuinely missing one",
    ),
    "clear_timeout": (
        +0.12,
        "the failure is an unambiguous timeout rather than an assertion mismatch",
    ),
    "clear_assertion": (
        +0.10,
        "the failure is an unambiguous assertion mismatch with concrete "
        "expected and actual values",
    ),
    "locator_named_in_error": (
        +0.08,
        "the error message names the specific locator that failed",
    ),
    "reproduced_twice": (
        +0.15,
        "the same failure reproduced on a second attempt, so it is not flaky",
    ),
    "spa_race_suspected": (
        -0.20,
        "the page is a single-page app and the timing suggests a render race",
    ),
    "captcha_or_bot_wall": (
        -0.30,
        "a captcha or bot wall is present, so nothing observed downstream is "
        "trustworthy",
    ),
    "network_flaky": (
        -0.18,
        "the network returned errors or timed out at the transport level",
    ),
    "ambiguous_expected_text": (
        -0.15,
        "the expected text is templated or ambiguous, so a mismatch may be "
        "cosmetic rather than a defect",
    ),
    "auth_wall": (
        -0.22,
        "the failure happened behind a login wall, so the observed page is not "
        "the page under test",
    ),
    "console_errors_present": (
        +0.06,
        "the page logged JavaScript errors, which corroborates an application "
        "fault over a script fault",
    ),
}


@dataclass
class ConfidenceSignals:
    """Evidence-derived booleans. Everything defaults to "not observed"."""

    selector_present_in_dom: bool | None = None
    failure_kind: str = "unknown"  # timeout | assertion | navigation | exception | unknown
    locator_named_in_error: bool = False
    reproduced_twice: bool = False
    spa_race_suspected: bool = False
    captcha_or_bot_wall: bool = False
    network_flaky: bool = False
    ambiguous_expected_text: bool = False
    auth_wall: bool = False
    console_errors_present: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "selector_present_in_dom": self.selector_present_in_dom,
            "failure_kind": self.failure_kind,
            "locator_named_in_error": self.locator_named_in_error,
            "reproduced_twice": self.reproduced_twice,
            "spa_race_suspected": self.spa_race_suspected,
            "captcha_or_bot_wall": self.captcha_or_bot_wall,
            "network_flaky": self.network_flaky,
            "ambiguous_expected_text": self.ambiguous_expected_text,
            "auth_wall": self.auth_wall,
            "console_errors_present": self.console_errors_present,
        }


@dataclass
class ConfidenceScore:
    value: float
    reasons: list[str] = field(default_factory=list)

    def rationale(self, limit: int = 2) -> str:
        if not self.reasons:
            return "No distinguishing evidence was available, so confidence stayed at the 0.5 base."
        return "; ".join(self.reasons[:limit]) + "."


def score_signals(signals: ConfidenceSignals, base: float = BASE_CONFIDENCE) -> ConfidenceScore:
    """Apply the confidence rubric to evidence. Deterministic and testable."""
    value = base
    reasons: list[str] = []

    def apply(key: str) -> None:
        nonlocal value
        delta, reason = ADJUSTMENTS[key]
        value += delta
        reasons.append(f"{'+' if delta >= 0 else ''}{delta:.2f} {reason}")

    if signals.selector_present_in_dom is True:
        apply("selector_present_in_dom")
    elif signals.selector_present_in_dom is False:
        apply("selector_absent_from_dom")

    if signals.failure_kind == "timeout":
        apply("clear_timeout")
    elif signals.failure_kind == "assertion":
        apply("clear_assertion")

    for key, flag in (
        ("locator_named_in_error", signals.locator_named_in_error),
        ("reproduced_twice", signals.reproduced_twice),
        ("spa_race_suspected", signals.spa_race_suspected),
        ("captcha_or_bot_wall", signals.captcha_or_bot_wall),
        ("network_flaky", signals.network_flaky),
        ("ambiguous_expected_text", signals.ambiguous_expected_text),
        ("auth_wall", signals.auth_wall),
        ("console_errors_present", signals.console_errors_present),
    ):
        if flag:
            apply(key)

    return ConfidenceScore(value=round(max(0.0, min(1.0, value)), 3), reasons=reasons)


LLM_CONFIDENCE_WEIGHT = 0.4
"""How much the model's self-reported confidence counts against the evidence.

Deliberately below 0.5 so the evidence has the larger vote. At 0.5 a maximally
confident model (1.0) paired with damning evidence (0.2) lands exactly on the
0.6 threshold and auto-applies a patch, which is the wrong outcome: when the
page shows a captcha and the locator is gone, no amount of model conviction
should move a patch into production. At 0.4 that same pair scores 0.52 and is
correctly routed to human review.
"""


def blend(
    llm_confidence: float | None,
    deterministic: float,
    llm_weight: float = LLM_CONFIDENCE_WEIGHT,
) -> float:
    """Combine the model's self-report with the evidence-derived score.

    When the model gave no usable number, the deterministic score stands alone.
    ``llm_weight`` caps how much a confident-sounding rationale can move the
    decision away from the evidence.
    """
    try:
        model_value = float(llm_confidence)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return round(max(0.0, min(1.0, deterministic)), 3)
    model_value = max(0.0, min(1.0, model_value))
    weight = max(0.0, min(1.0, llm_weight))
    return round(max(0.0, min(1.0, model_value * weight + deterministic * (1 - weight))), 3)


@dataclass
class HealDecision:
    """The branch taken for one failing test."""

    auto_apply: bool
    needs_human_review: bool
    action: str
    reason: str
    threshold: float = CONFIDENCE_AUTO_APPLY_THRESHOLD


def should_auto_apply(
    classification: DefectClass | str,
    confidence: float,
    *,
    has_fix: bool = True,
    threshold: float = CONFIDENCE_AUTO_APPLY_THRESHOLD,
) -> bool:
    """The single predicate that gates every automatic patch.

    True requires all three: the failure is a SCRIPT_ISSUE, a concrete fix was
    proposed, and confidence is at or above the threshold. A genuine defect is
    never auto-"fixed", regardless of how confident the model is.
    """
    value = classification.value if isinstance(classification, DefectClass) else str(classification)
    if value != DefectClass.SCRIPT_ISSUE.value:
        return False
    if not has_fix:
        return False
    try:
        return float(confidence) >= threshold
    except (TypeError, ValueError):
        return False


def decide(
    classification: DefectClass | str,
    confidence: float,
    *,
    has_fix: bool = True,
    threshold: float = CONFIDENCE_AUTO_APPLY_THRESHOLD,
) -> HealDecision:
    """Resolve one failure into the branch the pipeline will actually take."""
    value = classification.value if isinstance(classification, DefectClass) else str(classification)
    try:
        numeric = float(confidence)
    except (TypeError, ValueError):
        numeric = 0.0

    if value == DefectClass.GENUINE_DEFECT.value:
        return HealDecision(
            auto_apply=False,
            needs_human_review=numeric < threshold,
            action="route_to_bug_packager",
            reason=(
                "Classified as a genuine application defect; the test is left "
                "failing and a bug is packaged. Assertions are never weakened "
                "to make a real failure pass."
            ),
            threshold=threshold,
        )

    if value == DefectClass.ENVIRONMENT.value:
        return HealDecision(
            auto_apply=False,
            needs_human_review=True,
            action="quarantine_environment",
            reason=(
                "Classified as an environment problem (captcha, bot wall, "
                "network or auth wall). Neither a patch nor a bug is "
                "appropriate; a human should confirm the target is testable."
            ),
            threshold=threshold,
        )

    if value == DefectClass.SCRIPT_ISSUE.value:
        if not has_fix:
            return HealDecision(
                auto_apply=False,
                needs_human_review=True,
                action="queue_for_review",
                reason=(
                    "Classified as a script issue but no concrete locator or "
                    "wait fix was proposed, so there is nothing to apply."
                ),
                threshold=threshold,
            )
        if numeric >= threshold:
            return HealDecision(
                auto_apply=True,
                needs_human_review=False,
                action="apply_patch_and_rerun",
                reason=(
                    f"Script issue at {numeric:.2f} confidence, at or above the "
                    f"{threshold:.2f} auto-apply threshold; the locator/wait fix "
                    "is applied and the test is re-run once."
                ),
                threshold=threshold,
            )
        return HealDecision(
            auto_apply=False,
            needs_human_review=True,
            action="queue_for_review",
            reason=(
                f"Script issue at {numeric:.2f} confidence, below the "
                f"{threshold:.2f} threshold; the patch is NOT applied and the "
                "finding is queued for human review with its evidence."
            ),
            threshold=threshold,
        )

    return HealDecision(
        auto_apply=False,
        needs_human_review=True,
        action="queue_for_review",
        reason=(
            "The failure could not be classified from the available evidence, "
            "so nothing is changed automatically."
        ),
        threshold=threshold,
    )


def summarise_decisions(decisions: Sequence[HealDecision]) -> dict[str, int]:
    return {
        "auto_applied": sum(1 for d in decisions if d.auto_apply),
        "needs_human_review": sum(1 for d in decisions if d.needs_human_review),
        "routed_to_bugs": sum(1 for d in decisions if d.action == "route_to_bug_packager"),
        "quarantined": sum(1 for d in decisions if d.action == "quarantine_environment"),
    }
