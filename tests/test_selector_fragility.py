"""Selector strategy priority and fragility reporting.

A locator that resolves today is not the same as a locator that will still
resolve after the next release. The priority order decides which locator the
agent picks; the fragility grade tells the reader which of the resulting tests
are going to need maintenance. Both are reported, because a suite whose
maintenance cost is invisible is a suite that gets deleted.
"""

from __future__ import annotations

import pytest

from browser.selectors import (
    STRATEGY_FRAGILITY,
    STRATEGY_PRIORITY,
    assess_fragility,
    expression_for,
    fallback_order,
    fragility_summary,
    strategy_rank,
)


class TestStrategyPriority:
    def test_required_order_is_enforced(self):
        """test attributes, then role+name, then labels, then css/text.

        This is the whole reason generated tests survive a re-skin: a
        ``data-testid`` exists to be selected on, while a CSS path is an
        accident of the current markup.
        """
        assert STRATEGY_PRIORITY[0] == "testid"
        assert STRATEGY_PRIORITY[1] == "role"
        assert STRATEGY_PRIORITY[2] == "label"
        assert STRATEGY_PRIORITY.index("css") > STRATEGY_PRIORITY.index("label")
        assert STRATEGY_PRIORITY.index("text") > STRATEGY_PRIORITY.index("role")

    def test_rank_orders_stable_before_fragile(self):
        assert strategy_rank("testid") < strategy_rank("role") < strategy_rank("css")

    def test_unknown_strategy_sorts_last(self):
        assert strategy_rank("made-up") == len(STRATEGY_PRIORITY)

    def test_form_actions_lead_with_label(self):
        """A form control is labelled; a button is best addressed by role."""
        order = fallback_order("fill")
        assert order[0] == "testid"
        assert order.index("label") < order.index("role")

    def test_every_strategy_has_a_fragility_grade(self):
        for strategy in STRATEGY_PRIORITY:
            assert strategy in STRATEGY_FRAGILITY


class TestExpressionRendering:
    @pytest.mark.parametrize(
        "strategy,value,expected_fragment",
        [
            ("testid", "submit", "get_by_test_id"),
            ("label", "Email", "get_by_label"),
            ("placeholder", "Search", "get_by_placeholder"),
            ("text", "Continue", "get_by_text"),
            ("css", ".btn", "locator"),
        ],
    )
    def test_expressions_use_the_matching_playwright_api(
        self, strategy, value, expected_fragment
    ):
        assert expected_fragment in expression_for(strategy, value)

    def test_role_includes_the_accessible_name(self):
        rendered = expression_for("role", "Sign in", role="button")
        assert 'get_by_role("button"' in rendered and "Sign in" in rendered

    def test_quotes_in_a_value_are_escaped(self):
        """An unescaped quote would produce a syntactically invalid test file."""
        rendered = expression_for("text", 'He said "hi"')
        assert rendered.count('"') % 2 == 0
        assert "\\" in rendered

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            expression_for("telepathy", "x")


class TestFragilityAssessment:
    @pytest.mark.parametrize("strategy", ["testid", "role", "label"])
    def test_durable_strategies_grade_stable(self, strategy):
        assessment = assess_fragility(strategy, "page.x()")
        assert assessment["grade"] == "stable"
        assert assessment["recommendation"] == ""

    @pytest.mark.parametrize("strategy", ["text", "css"])
    def test_brittle_strategies_grade_fragile(self, strategy):
        assessment = assess_fragility(strategy, "page.x()")
        assert assessment["grade"] == "fragile"
        assert "data-testid" in assessment["recommendation"]

    @pytest.mark.parametrize("strategy", ["placeholder", "alt", "title"])
    def test_copy_based_strategies_grade_moderate(self, strategy):
        assert assess_fragility(strategy, "page.x()")["grade"] == "moderate"

    @pytest.mark.parametrize(
        "expression",
        [
            'page.locator(".css-1a2b3c4")',
            'page.locator(".sc-bdVaJa")',
            'page.locator("div > p:nth-child(3)")',
        ],
    )
    def test_generated_class_names_and_positions_are_flagged(self, expression):
        """These change on literally every build of the target application."""
        assessment = assess_fragility("css", expression)
        assert assessment["grade"] == "fragile"
        assert assessment["notes"]

    def test_ambiguous_match_is_flagged_even_for_a_good_strategy(self):
        """Multiple matches means the element is chosen by position.

        The locator resolves, so nothing fails today; it will silently target a
        different element the moment the order changes.
        """
        assessment = assess_fragility("role", 'page.get_by_role("button")', match_count=4)
        assert assessment["grade"] == "fragile"
        assert any("4 elements" in note for note in assessment["notes"])

    def test_unknown_strategy_is_treated_as_fragile(self):
        """Unknown must fail towards *reporting* risk, not towards hiding it."""
        assert assess_fragility("mystery", "x")["grade"] == "fragile"


class TestFragilitySummary:
    def test_all_stable_reads_as_durable(self):
        summary = fragility_summary(
            [assess_fragility("testid", "a"), assess_fragility("role", "b")]
        )
        assert summary["verdict"] == "durable"
        assert summary["fragile_ratio"] == 0.0

    def test_mixed_suite_reports_a_ratio(self):
        summary = fragility_summary(
            [
                assess_fragility("testid", "a"),
                assess_fragility("css", 'page.locator(".x")'),
                assess_fragility("text", "b"),
                assess_fragility("role", "c"),
            ]
        )
        assert summary["counts"]["fragile"] == 2
        assert summary["fragile_ratio"] == 0.5
        assert summary["verdict"] != "durable"

    def test_empty_suite_does_not_divide_by_zero(self):
        summary = fragility_summary([])
        assert summary["total"] == 0
        assert summary["fragile_ratio"] == 0.0
        assert summary["verdict"] == "no selectors resolved"
