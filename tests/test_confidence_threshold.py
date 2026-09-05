"""The 0.60 auto-apply threshold and the branch it gates.

This is the safety-critical logic in the system. Two properties matter most and
both are asserted here directly:

* a GENUINE_DEFECT is never auto-"fixed", at any confidence;
* a patch that would remove or weaken an assertion is rejected, at any
  confidence.
"""

from __future__ import annotations

import pytest

from agents.healer import patch_weakens_assertions
from config import CONFIDENCE_AUTO_APPLY_THRESHOLD
from differentiation.confidence_scorer import (
    BASE_CONFIDENCE,
    ConfidenceSignals,
    blend,
    decide,
    score_signals,
    should_auto_apply,
    summarise_decisions,
)
from graph.state import DefectClass


class TestThresholdConstant:
    def test_threshold_is_the_documented_value(self):
        assert CONFIDENCE_AUTO_APPLY_THRESHOLD == 0.6


class TestShouldAutoApply:
    def test_script_issue_at_threshold_applies(self):
        assert should_auto_apply(DefectClass.SCRIPT_ISSUE, 0.6) is True

    def test_script_issue_above_threshold_applies(self):
        assert should_auto_apply(DefectClass.SCRIPT_ISSUE, 0.95) is True

    def test_script_issue_just_below_threshold_does_not_apply(self):
        assert should_auto_apply(DefectClass.SCRIPT_ISSUE, 0.59) is False

    def test_the_reported_low_confidence_case_does_not_apply(self):
        # "Healer: 0.41 confidence - NOT auto-applied, queued for human review"
        assert should_auto_apply(DefectClass.SCRIPT_ISSUE, 0.41) is False

    @pytest.mark.parametrize("confidence", [0.6, 0.8, 0.99, 1.0])
    def test_genuine_defect_never_auto_applies(self, confidence):
        assert should_auto_apply(DefectClass.GENUINE_DEFECT, confidence) is False

    @pytest.mark.parametrize(
        "classification", [DefectClass.ENVIRONMENT, DefectClass.UNKNOWN]
    )
    def test_non_script_classes_never_auto_apply(self, classification):
        assert should_auto_apply(classification, 1.0) is False

    def test_no_proposed_fix_means_nothing_to_apply(self):
        assert should_auto_apply(DefectClass.SCRIPT_ISSUE, 0.9, has_fix=False) is False

    def test_accepts_a_string_classification(self):
        assert should_auto_apply("SCRIPT_ISSUE", 0.7) is True

    @pytest.mark.parametrize("bad", [None, "high", float("nan")])
    def test_unusable_confidence_never_applies(self, bad):
        assert should_auto_apply(DefectClass.SCRIPT_ISSUE, bad) is False


class TestDecide:
    def test_high_confidence_script_issue_takes_the_patch_branch(self):
        decision = decide(DefectClass.SCRIPT_ISSUE, 0.82)
        assert decision.auto_apply is True
        assert decision.needs_human_review is False
        assert decision.action == "apply_patch_and_rerun"

    def test_low_confidence_script_issue_takes_the_review_branch(self):
        decision = decide(DefectClass.SCRIPT_ISSUE, 0.41)
        assert decision.auto_apply is False
        assert decision.needs_human_review is True
        assert decision.action == "queue_for_review"
        assert "0.41" in decision.reason and "0.60" in decision.reason

    def test_genuine_defect_routes_to_the_bug_packager(self):
        decision = decide(DefectClass.GENUINE_DEFECT, 0.9)
        assert decision.auto_apply is False
        assert decision.action == "route_to_bug_packager"
        assert "never" in decision.reason.lower()

    def test_environment_failure_is_quarantined_for_a_human(self):
        decision = decide(DefectClass.ENVIRONMENT, 0.9)
        assert decision.action == "quarantine_environment"
        assert decision.needs_human_review is True

    def test_unknown_changes_nothing_automatically(self):
        decision = decide(DefectClass.UNKNOWN, 0.9)
        assert decision.auto_apply is False
        assert decision.needs_human_review is True

    def test_script_issue_without_a_fix_is_queued(self):
        decision = decide(DefectClass.SCRIPT_ISSUE, 0.99, has_fix=False)
        assert decision.auto_apply is False
        assert "no concrete" in decision.reason

    def test_summarise_counts_the_branches(self):
        decisions = [
            decide(DefectClass.SCRIPT_ISSUE, 0.9),
            decide(DefectClass.SCRIPT_ISSUE, 0.2),
            decide(DefectClass.GENUINE_DEFECT, 0.8),
            decide(DefectClass.ENVIRONMENT, 0.8),
        ]
        summary = summarise_decisions(decisions)
        assert summary["auto_applied"] == 1
        assert summary["routed_to_bugs"] == 1
        assert summary["quarantined"] == 1
        # The low-confidence script issue and the environment failure need a
        # human. The confidently-classified genuine defect does not: it becomes
        # a filed bug, which is an answer rather than an open question.
        assert summary["needs_human_review"] == 2

    def test_a_low_confidence_genuine_defect_still_wants_a_human(self):
        assert decide(DefectClass.GENUINE_DEFECT, 0.3).needs_human_review is True


class TestSignalScoring:
    def test_no_evidence_stays_at_the_base(self):
        assert score_signals(ConfidenceSignals()).value == pytest.approx(BASE_CONFIDENCE)

    def test_selector_present_plus_clear_timeout_raises_above_threshold(self):
        score = score_signals(
            ConfidenceSignals(selector_present_in_dom=True, failure_kind="timeout")
        )
        assert score.value >= CONFIDENCE_AUTO_APPLY_THRESHOLD

    def test_captcha_drags_confidence_well_below_threshold(self):
        score = score_signals(ConfidenceSignals(captcha_or_bot_wall=True))
        assert score.value < CONFIDENCE_AUTO_APPLY_THRESHOLD

    def test_auth_wall_drags_confidence_down(self):
        assert score_signals(ConfidenceSignals(auth_wall=True)).value < BASE_CONFIDENCE

    def test_spa_race_drags_confidence_down(self):
        assert score_signals(ConfidenceSignals(spa_race_suspected=True)).value < BASE_CONFIDENCE

    def test_reproduced_twice_raises_confidence(self):
        assert score_signals(ConfidenceSignals(reproduced_twice=True)).value > BASE_CONFIDENCE

    def test_score_is_clamped_to_the_unit_interval(self):
        maxed = score_signals(
            ConfidenceSignals(
                selector_present_in_dom=True,
                failure_kind="timeout",
                locator_named_in_error=True,
                reproduced_twice=True,
                console_errors_present=True,
            )
        )
        floored = score_signals(
            ConfidenceSignals(
                spa_race_suspected=True,
                captcha_or_bot_wall=True,
                network_flaky=True,
                ambiguous_expected_text=True,
                auth_wall=True,
            )
        )
        assert 0.0 <= floored.value <= maxed.value <= 1.0

    def test_reasons_are_recorded_for_the_report(self):
        score = score_signals(ConfidenceSignals(captcha_or_bot_wall=True))
        assert score.reasons and "captcha" in score.rationale().lower()


class TestBlend:
    def test_evidence_carries_the_larger_weight(self):
        # 0.4 model / 0.6 evidence, so the grounded half wins a disagreement.
        assert blend(1.0, 0.0) == pytest.approx(0.4)
        assert blend(0.0, 1.0) == pytest.approx(0.6)

    def test_agreement_leaves_the_value_alone(self):
        assert blend(0.7, 0.7) == pytest.approx(0.7)

    def test_missing_model_confidence_leaves_the_evidence_alone(self):
        assert blend(None, 0.42) == pytest.approx(0.42)

    def test_unparsable_model_confidence_leaves_the_evidence_alone(self):
        assert blend("very sure", 0.42) == pytest.approx(0.42)

    def test_a_confident_model_cannot_override_damning_evidence(self):
        # Model says 1.0, evidence says 0.2 -> still below the threshold.
        assert blend(1.0, 0.2) < CONFIDENCE_AUTO_APPLY_THRESHOLD

    def test_result_is_clamped(self):
        assert 0.0 <= blend(5.0, 5.0) <= 1.0
        assert 0.0 <= blend(-5.0, -5.0) <= 1.0


class TestAssertionWeakeningGuard:
    BASE = (
        "async def test_flow(page, ctx):\n"
        "    await page.goto('https://example.com/')\n"
        "    await expect(page.get_by_role('heading')).to_contain_text('Welcome')\n"
        "    await expect(page.get_by_test_id('total')).to_have_text('10.00')\n"
    )

    def test_a_pure_locator_swap_is_allowed(self):
        patched = self.BASE.replace("get_by_test_id('total')", "get_by_test_id('order-total')")
        assert patch_weakens_assertions(self.BASE, patched) is False

    def test_deleting_an_assertion_is_blocked(self):
        patched = "\n".join(
            line for line in self.BASE.splitlines() if "to_have_text" not in line
        )
        assert patch_weakens_assertions(self.BASE, patched) is True

    def test_commenting_out_an_assertion_is_blocked(self):
        patched = self.BASE.replace(
            "    await expect(page.get_by_test_id('total')).to_have_text('10.00')",
            "    # await expect(page.get_by_test_id('total')).to_have_text('10.00')",
        )
        assert patch_weakens_assertions(self.BASE, patched) is True

    def test_downgrading_a_text_assertion_to_visibility_is_blocked(self):
        patched = self.BASE.replace(
            "to_have_text('10.00')", "to_be_visible()"
        )
        assert patch_weakens_assertions(self.BASE, patched) is True

    def test_adding_an_assertion_is_allowed(self):
        patched = self.BASE + "    await expect(page.locator('body')).to_be_visible()\n"
        assert patch_weakens_assertions(self.BASE, patched) is False

    def test_empty_inputs_are_not_treated_as_weakening(self):
        assert patch_weakens_assertions("", self.BASE) is False
        assert patch_weakens_assertions(self.BASE, "") is False
