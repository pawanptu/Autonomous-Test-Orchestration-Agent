"""Bounded re-runs, flakiness reporting, and visual baseline identity.

Two related judgements are pinned here.

*When to re-run.* Re-running every failure doubles the length of every red
build and trains people to dismiss the "flaky" label. Re-running nothing leaves
a genuine defect and a slow network indistinguishable. The rule is: re-run when
the failure mode is one that genuinely varies between attempts, or when the flow
carries enough risk that certainty is worth the wall-clock.

*What a passing re-run means.* A flow that fails then passes is unstable, and is
reported as flaky rather than as green - an intermittent failure is one a user
will eventually meet.
"""

from __future__ import annotations

import pytest

from browser.runner import classify_rerun, should_rerun_for_flake
from config import Settings
from differentiation.visual_diff import BaselineIdentity, baseline_path_for, mask_signature
from graph.state import RiskLevel, TestResult, TestStatus


@pytest.fixture
def cfg() -> Settings:
    return Settings()


def _failure(error_type: str, message: str = "", attempts: int = 1) -> TestResult:
    return TestResult(
        flow_id="F1",
        status=TestStatus.FAILED,
        error_type=error_type,
        error_message=message,
        attempts=attempts,
    )


class TestRerunDecision:
    @pytest.mark.parametrize(
        "error_type,message",
        [
            ("TimeoutError", "Timeout 8000ms exceeded"),
            ("TestTimeout", "the test did not finish"),
            ("Error", "net::ERR_CONNECTION_RESET"),
            ("Error", "element is not attached to the DOM"),
            ("Error", "Execution context was destroyed"),
            ("Error", "waiting for locator to be visible"),
        ],
    )
    def test_timing_dependent_failures_are_re_run(self, error_type, message, cfg):
        should, why = should_rerun_for_flake(_failure(error_type, message), RiskLevel.LOW, cfg)
        assert should is True
        assert "timing" in why

    def test_deterministic_failure_on_a_low_risk_flow_is_not_re_run(self, cfg):
        """An assertion failure reproduces by construction; a re-run buys nothing."""
        should, why = should_rerun_for_flake(
            _failure("AssertionError", "expected 'Welcome'"), RiskLevel.LOW, cfg
        )
        assert should is False
        assert "conclusive" in why

    def test_high_risk_failure_is_confirmed_even_when_deterministic(self, cfg):
        """Before filing a defect on a critical flow, confirm it."""
        should, why = should_rerun_for_flake(
            _failure("AssertionError", "expected 'Welcome'"), RiskLevel.HIGH, cfg
        )
        assert should is True
        assert "high-risk" in why

    def test_a_passing_test_is_never_re_run(self, cfg):
        result = TestResult(flow_id="F1", status=TestStatus.PASSED)
        assert should_rerun_for_flake(result, RiskLevel.HIGH, cfg)[0] is False

    def test_rerun_is_capped_at_one(self, cfg):
        """The cap is what keeps a run finite when a target is simply down."""
        already = _failure("TimeoutError", "timeout", attempts=2)
        should, why = should_rerun_for_flake(already, RiskLevel.HIGH, cfg)
        assert should is False
        assert "already re-run" in why

    def test_configuration_can_disable_reruns(self, cfg):
        disabled = cfg.with_overrides(rerun_failed_flows=False)
        assert should_rerun_for_flake(_failure("TimeoutError"), RiskLevel.HIGH, disabled)[0] is False


class TestRerunClassification:
    def test_fail_then_pass_is_reported_as_flaky_not_as_a_pass(self):
        """Reporting it green would hide instability a user will hit."""
        first = _failure("TimeoutError", "timeout")
        second = TestResult(flow_id="F1", status=TestStatus.PASSED)
        result, verdict = classify_rerun(first, second)
        assert result.flaky is True
        assert result.status is TestStatus.PASSED
        assert "flaky" in verdict
        assert result.notes

    def test_reproduced_failure_is_treated_as_a_real_defect(self):
        first = _failure("AssertionError", "expected 'Welcome'")
        second = _failure("AssertionError", "expected 'Welcome'")
        result, verdict = classify_rerun(first, second)
        assert result.flaky is False
        assert "confirmed" in verdict
        assert any("reproduced" in note for note in result.notes)


class TestBaselineIdentity:
    def test_environment_changes_the_baseline_key(self):
        """A staging capture is not a valid reference for production."""
        base = Settings()
        default = BaselineIdentity.from_settings(base)
        staging = BaselineIdentity.from_settings(base.with_overrides(visual_environment="staging"))
        assert default.key() != staging.key()

    def test_locale_changes_the_baseline_key(self):
        base = Settings()
        english = BaselineIdentity.from_settings(base)
        german = BaselineIdentity.from_settings(base.with_overrides(visual_locale="de-DE"))
        assert english.key() != german.key()

    def test_theme_changes_the_baseline_key(self):
        base = Settings()
        light = BaselineIdentity.from_settings(base, theme="light")
        dark = BaselineIdentity.from_settings(base, theme="dark")
        assert light.key() != dark.key()

    def test_viewport_changes_the_baseline_key(self):
        base = Settings()
        wide = BaselineIdentity.from_settings(base, viewport="1280x900")
        narrow = BaselineIdentity.from_settings(base, viewport="390x844")
        assert wide.key() != narrow.key()

    def test_browser_major_version_changes_the_baseline_key(self):
        base = Settings()
        old = BaselineIdentity.from_settings(base, browser_version="128.0.6613.120")
        new = BaselineIdentity.from_settings(base, browser_version="131.0.1.1")
        assert old.key() != new.key()

    def test_browser_patch_version_does_not_change_the_key(self):
        """Chromium repaints subtly between patches without invalidating a reference.

        Keying on the patch version would discard every baseline on every
        browser update, which makes the whole feature useless.
        """
        base = Settings()
        a = BaselineIdentity.from_settings(base, browser_version="128.0.6613.120")
        b = BaselineIdentity.from_settings(base, browser_version="128.0.9999.1")
        assert a.key() == b.key()

    def test_mask_rules_change_the_baseline_key(self):
        """A masked capture and an unmasked one are different images by construction."""
        base = Settings()
        plain = BaselineIdentity.from_settings(base)
        masked = BaselineIdentity.from_settings(
            base.with_overrides(visual_mask_selectors=(".price",))
        )
        assert plain.key() != masked.key()

    def test_build_id_is_recorded_but_does_not_key_the_baseline(self):
        """A baseline exists precisely to be compared across builds."""
        base = Settings()
        first = BaselineIdentity.from_settings(base.with_overrides(visual_build_id="1.0.0"))
        second = BaselineIdentity.from_settings(base.with_overrides(visual_build_id="2.0.0"))
        assert first.key() == second.key()
        assert first.describe()["build_id"] == "1.0.0"

    def test_identity_is_reflected_in_the_baseline_path(self):
        identity = BaselineIdentity.from_settings(Settings(), viewport="1280x900")
        path = baseline_path_for("https://ex.com", "F1", "1280x900", identity)
        assert identity.key() in str(path)

    def test_legacy_path_is_preserved_when_no_identity_is_given(self):
        """Baselines already on disk from earlier runs must stay reachable."""
        path = baseline_path_for("https://ex.com", "F1", "1280x900")
        assert path.name == "F1__1280x900.png"
        assert path.parent.name == "ex.com"

    def test_describe_exposes_provenance(self):
        described = BaselineIdentity.from_settings(
            Settings(), browser_version="128.0.1.1"
        ).describe()
        for field in ("environment", "locale", "theme", "viewport", "browser_major", "key"):
            assert field in described


class TestMaskSignature:
    def test_order_does_not_matter(self):
        assert mask_signature([".a", ".b"]) == mask_signature([".b", ".a"])

    def test_empty_is_a_stable_sentinel(self):
        assert mask_signature([]) == "none"
        assert mask_signature(["  "]) == "none"

    def test_different_rules_differ(self):
        assert mask_signature([".a"]) != mask_signature([".b"])
